"""Validate and post the AI first-pass review produced by `analyse.py`.

Runs in the `publish` workflow job with `pull-requests: write` and no LLM API
keys. Verifies `HEAD_SHA`, restricts inline comments to right-side lines inside
diff hunks, strips `@mentions` and external links, and posts `COMMENT` only.
"""

# Standard library
import argparse
import http.client
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REVIEW_MARKER = "<!-- neural-lam-ai-review -->"
AI_TRIAGED_LABEL = "ai-triaged"
NEEDS_TRIAGE_LABEL = "needs-triage"
ALLOWED_URL_PREFIX = "https://github.com/mllam/"
MAX_COMMENTS = 10
MAX_COMMENT_CHARS = 1500
MAX_SUMMARY_CHARS = 4000
MAX_SUGGESTION_CHARS = 2000
FOOTER = (
    f"\n\n---\n*Automated first-pass review; a maintainer will still review "
    f"and merge.*\n{REVIEW_MARKER}"
)

_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_REF_LINK_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+).*$", re.MULTILINE)
_BARE_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>)\]]+", re.IGNORECASE)
_MAILTO_RE = re.compile(r"\bmailto:[^\s<>)\]]+", re.IGNORECASE)
_EMAIL_RE = re.compile(
    r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"
)
_MENTION_RE = re.compile(
    r"(?<![\w`])@"
    r"([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:/[A-Za-z0-9_-]+)?)"
)


def gh_api(
    path: str,
    token: str,
    *,
    method: str = "GET",
    accept: str = "application/vnd.github+json",
    payload: dict[str, Any] | None = None,
) -> str:
    """Call the GitHub REST API over HTTPS and raise on non-2xx responses."""
    headers = {
        "Accept": accept,
        "User-Agent": "neural-lam-ai-review",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    conn = http.client.HTTPSConnection("api.github.com", timeout=60)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        text = resp.read().decode("utf-8", errors="replace")
    finally:
        conn.close()

    if not 200 <= resp.status < 300:
        print(
            f"GitHub API {method} {path} failed ({resp.status}): {text}",
            file=sys.stderr,
        )
        raise RuntimeError(
            f"GitHub API {method} {path} returned HTTP {resp.status}"
        )
    return text


def right_side_lines(diff_text: str) -> dict[str, set[int]]:
    """Return `{path: right_side_line_numbers}` inside unified diff hunks."""
    allowed: dict[str, set[int]] = {}
    current_path: str | None = None
    right_line = rem_old = rem_new = 0

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("diff --git "):
            current_path = None
            rem_old = rem_new = 0
        elif raw_line.startswith("+++ b/") and rem_old == 0 and rem_new == 0:
            current_path = raw_line[6:]
            allowed.setdefault(current_path, set())
        elif (
            raw_line.startswith("+++ /dev/null")
            and rem_old == 0
            and rem_new == 0
        ):
            current_path = None
        elif raw_line.startswith("@@ ") and rem_old == 0 and rem_new == 0:
            match = _HUNK_RE.match(raw_line)
            if match is not None and current_path is not None:
                rem_old = int(match.group(1)) if match.group(1) else 1
                right_line = int(match.group(2))
                rem_new = int(match.group(3)) if match.group(3) else 1
        elif current_path is not None and (rem_old > 0 or rem_new > 0):
            if raw_line.startswith("\\"):
                continue
            if raw_line.startswith("+") and rem_new > 0:
                allowed[current_path].add(right_line)
                right_line += 1
                rem_new -= 1
            elif raw_line.startswith("-") and rem_old > 0:
                rem_old -= 1
            elif (
                (raw_line.startswith(" ") or raw_line == "")
                and rem_old > 0
                and rem_new > 0
            ):
                allowed[current_path].add(right_line)
                right_line += 1
                rem_old -= 1
                rem_new -= 1
            else:
                rem_old = rem_new = 0
    return allowed


def _is_allowed_url(url: str) -> bool:
    return (
        url.startswith(ALLOWED_URL_PREFIX)
        and ".." not in url
        and "\\" not in url
    )


def clean(text: str, *, max_chars: int = MAX_COMMENT_CHARS) -> str:
    """Sanitize prose: escape HTML, strip outside links and @mentions."""
    # Escape entities and HTML tags first so no entities or raw tags can form
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Remove markdown images
    text = _MD_IMAGE_RE.sub("", text)
    # Allowed markdown links are preserved; others collapse to link text
    text = _MD_LINK_RE.sub(
        lambda m: m.group(0) if _is_allowed_url(m.group(2)) else m.group(1),
        text,
    )
    # Strip external reference link definitions
    text = _REF_LINK_DEF_RE.sub(
        lambda m: m.group(0) if _is_allowed_url(m.group(1)) else "",
        text,
    )
    # Neutralize mailto: and emails
    text = _MAILTO_RE.sub("", text)
    text = _EMAIL_RE.sub(r"\1 at \2", text)
    # Remove bare external URLs and autolinks
    text = _BARE_URL_RE.sub(
        lambda m: m.group(0) if _is_allowed_url(m.group(0)) else "",
        text,
    )
    # Neutralize @mentions
    text = _MENTION_RE.sub(r"\1", text)
    return text[:max_chars].rstrip()


def format_suggestion(suggestion: str) -> str:
    """Format raw replacement code into a GitHub suggestion block.

    Drops the block if the code contains backtick or tilde fences, or if it
    exceeds `MAX_SUGGESTION_CHARS`.
    """
    code = suggestion.strip("\r\n")
    if not code:
        return ""
    if "```" in code or "~~~" in code or len(code) > MAX_SUGGESTION_CHARS:
        return ""
    return f"\n```suggestion\n{code}\n```"


def add_label(repo: str, pr_number: int, label: str, token: str) -> None:
    """Add a label to the pull request via GitHub REST API."""
    gh_api(
        f"/repos/{repo}/issues/{pr_number}/labels",
        token,
        method="POST",
        payload={"labels": [label]},
    )


def already_reviewed(reviews: list[dict[str, Any]]) -> bool:
    """Return True if `github-actions[bot]` already posted an AI review."""
    return any(
        (r.get("user") or {}).get("login") == "github-actions[bot]"
        and REVIEW_MARKER in (r.get("body") or "")
        for r in reviews
    )


def build_review_payload(
    result: dict[str, Any],
    diff_text: str,
    expected_head_sha: str,
) -> dict[str, Any] | None:
    """Validate `result` against `diff_text` and build the review payload."""
    if result.get("status") != "review":
        return None
    if result.get("head_sha") != expected_head_sha:
        raise RuntimeError("Artifact head_sha does not match workflow HEAD_SHA")

    review = result.get("review")
    if not isinstance(review, dict):
        raise ValueError("Missing `review` object in result JSON")
    raw_summary, raw_comments = review.get("summary"), review.get("comments")
    if not isinstance(raw_summary, str) or not isinstance(raw_comments, list):
        raise ValueError("Invalid `review` fields in result JSON")

    changed = right_side_lines(diff_text)
    comments: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for entry in raw_comments:
        if len(comments) >= MAX_COMMENTS:
            break
        if not isinstance(entry, dict):
            continue
        path, line, body, suggestion = (
            entry.get("path"),
            entry.get("line"),
            entry.get("body"),
            entry.get("suggestion", ""),
        )
        if (
            not isinstance(path, str)
            or isinstance(line, bool)
            or not isinstance(line, int)
            or not isinstance(body, str)
            or not isinstance(suggestion, str)
            or line not in changed.get(path, set())
            or (path, line) in seen
        ):
            continue
        cleaned_body = clean(body, max_chars=MAX_COMMENT_CHARS)
        suggestion_block = format_suggestion(suggestion)
        full_comment = f"{cleaned_body}{suggestion_block}".strip()
        if full_comment:
            seen.add((path, line))
            comments.append(
                {
                    "path": path,
                    "line": line,
                    "side": "RIGHT",
                    "body": full_comment,
                }
            )

    summary = clean(raw_summary, max_chars=MAX_SUMMARY_CHARS)
    if not summary and not comments:
        return None
    return {
        "commit_id": expected_head_sha,
        "event": "COMMENT",
        "body": f"{summary}{FOOTER}" if summary else FOOTER.lstrip(),
        "comments": comments,
    }


def format_dry_run(payload: dict[str, Any] | None, reason: str = "") -> str:
    """Render a plain-text preview of the review payload for `--dry-run`."""
    if payload is None:
        return f"No review to post ({reason or 'skipped'}).\n"
    lines = [
        f"Event: {payload['event']} (commit {payload['commit_id']})",
        "",
        payload["body"],
    ]
    for item in payload["comments"]:
        lines.extend(
            [
                "",
                f"--- {item['path']}:{item['line']} ({item['side']}) ---",
                item["body"],
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `.github/ai_review/publish.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("--repo", help="GitHub repository as owner/name")
    parser.add_argument("--pr", type=int, help="Pull request number")
    parser.add_argument("--diff-file", type=Path, help="Local diff file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = json.loads(args.result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result.json must contain a JSON object")

    head_sha = (os.environ.get("HEAD_SHA") or "").strip()
    is_live = args.diff_file is None
    token = (os.environ.get("GH_TOKEN") or "") if is_live else ""

    if is_live:
        if not args.repo or args.pr is None:
            parser.error("either --diff-file or --repo and --pr are required")
        base = f"/repos/{args.repo}/pulls/{args.pr}"
        pr = json.loads(gh_api(base, token))
        labels = {
            lbl.get("name")
            for lbl in (pr.get("labels") or [])
            if isinstance(lbl, dict)
        }

        live_sha = str(pr["head"]["sha"])
        head_sha = head_sha or live_sha
        if live_sha != head_sha:
            print("PR moved on since the analysis, not posting.")
            return 0
        if NEEDS_TRIAGE_LABEL in labels:
            print(f"PR has the {NEEDS_TRIAGE_LABEL} label, not posting.")
            return 0
        reviews = json.loads(gh_api(f"{base}/reviews?per_page=100", token))
        if already_reviewed(reviews):
            print("PR already has an AI first-pass review, not posting.")
            return 0

        # Mark PR as ai-triaged so subsequent toggles skip calling the model
        if result.get("triage") is not None and AI_TRIAGED_LABEL not in labels:
            try:
                add_label(args.repo, args.pr, AI_TRIAGED_LABEL, token)
            except Exception as exc:
                print(
                    f"Warning: could not add {AI_TRIAGED_LABEL} label: {exc}",
                    file=sys.stderr,
                )

        diff_text = (
            gh_api(base, token, accept="application/vnd.github.diff")
            if result.get("status") == "review"
            else ""
        )
    else:
        if not args.dry_run:
            parser.error("--diff-file requires --dry-run")
        diff_text = args.diff_file.read_text(encoding="utf-8")
        head_sha = head_sha or str(result.get("head_sha") or "local")

    if result.get("status") != "review":
        reason = str(result.get("reason") or "skipped")
        print(
            format_dry_run(None, reason)
            if args.dry_run
            else f"Skipping publish: {reason}",
            end="" if args.dry_run else "\n",
        )
        return 0

    payload = build_review_payload(result, diff_text, head_sha)
    if args.dry_run:
        print(format_dry_run(payload, "empty review after filtering"), end="")
        return 0
    if payload is None:
        print("No valid review content remained after sanitization, skipping.")
        return 0

    gh_api(
        f"/repos/{args.repo}/pulls/{args.pr}/reviews",
        token,
        method="POST",
        payload=payload,
    )
    print(f"Posted COMMENT review with {len(payload['comments'])} comment(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
