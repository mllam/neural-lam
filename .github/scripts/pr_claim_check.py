"""
Automated triage check for PRs without a linked issue or competing with others.

Run by `.github/workflows/pr-claim-check.yml` on `pull_request_target`, so it
always executes from the default branch and never runs code from the PR itself.
"""

# Standard library
import argparse
import json
import re
import subprocess
from typing import Any

MARKER = "<!-- pr-claim-check -->"
LABEL = "needs-triage"
CONTRIBUTING_URL = (
    "https://github.com/mllam/neural-lam/blob/main/CONTRIBUTING.md"
    "#getting-started"
)
CLOSING_KEYWORDS = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|solve[sd]?)"

OPEN_PRS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: 100, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        body
        createdAt
        author { login __typename }
        labels(first: 20) { nodes { name } }
        closingIssuesReferences(first: 20) {
          nodes { number repository { nameWithOwner } }
        }
      }
    }
  }
}
"""


def gh(*args: str) -> str:
    """Run `gh api` with the given arguments and return its stdout."""
    try:
        return subprocess.run(
            ["gh", "api", *args], check=True, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError as err:
        print(err.stderr)
        raise


def login(pr: dict[str, Any]) -> str:
    """Return the PR author's login (deleted accounts have no author)."""
    return (pr.get("author") or {}).get("login", "ghost")


def is_bot(pr: dict[str, Any]) -> bool:
    """Return True if the PR author is a bot."""
    author = pr.get("author") or {}
    return (
        author.get("__typename") == "Bot"
        or login(pr).endswith("[bot]")
        or login(pr) == "dependabot"
    )


def template_problems(body: str) -> list[str]:
    """Check if PR description is empty or has unfilled placeholders."""
    if not body or not body.strip():
        return [
            "The PR description is empty. Please fill out the pull request "
            "template describing the changes and motivation."
        ]

    placeholders = [
        "< Summary of the changes.>",
        "< Summary of the changes>",
        "< Please also include relevant motivation and context. >",
        "< List any dependencies that are required for this change. >",
        "< Link to the relevant issue",
    ]
    if any(p in body for p in placeholders):
        return [
            "The PR description contains placeholder text from the template. "
            "Please describe the changes and motivation."
        ]
    return []


def fetch_open_prs(repo: str) -> list[dict[str, Any]]:
    """Fetch all open pull requests of ``repo`` with the fields used here."""
    owner, name = repo.split("/")
    prs: list[dict[str, Any]] = []
    cursor = None
    while True:
        args = ["graphql", "-f", f"query={OPEN_PRS_QUERY}"]
        args += ["-f", f"owner={owner}", "-f", f"name={name}"]
        if cursor:
            args += ["-f", f"cursor={cursor}"]
        page = json.loads(gh(*args))["data"]["repository"]["pullRequests"]
        prs.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            return prs
        cursor = page["pageInfo"]["endCursor"]


def issue_ref(repo: str) -> str:
    """
    Regex capturing N in `#N`, `<repo>#N` or an issue URL of ``repo``. Skips
    other repos' `owner/name#N` and colours like `#27ae60`.
    """
    own = re.escape(repo)
    return (
        rf"(?:(?<![\w/])(?:{own})?#|https://github\.com/{own}/issues/)(\d+)\b"
    )


def claimed_numbers(pr: dict[str, Any], repo: str) -> set[int]:
    """
    Return the numbers a pull request claims to close: GitHub's closing
    references plus `closes/fixes/resolves/solves #N` in the body. Mentions
    such as "part of #N" or "refs #N" are not claims.
    """
    ref = issue_ref(repo)
    closing_refs = (pr.get("closingIssuesReferences") or {}).get("nodes", [])
    numbers = {
        n["number"]
        for n in closing_refs
        if n.get("repository", {}).get("nameWithOwner") == repo
    }
    numbers |= {
        int(n)
        for n in re.findall(
            rf"\b{CLOSING_KEYWORDS}:?\s+(?:issue\s+)?{ref}",
            pr.get("body") or "",
            re.IGNORECASE,
        )
    }
    # `#00` is the placeholder in the PR template
    numbers.discard(0)
    return numbers


def mentioned_numbers(pr: dict[str, Any], repo: str) -> set[int]:
    """Return every issue/PR number the pull request mentions or claims."""
    numbers = {
        int(n) for n in re.findall(issue_ref(repo), pr.get("body") or "")
    }
    numbers |= claimed_numbers(pr, repo)
    numbers.discard(0)
    return numbers


def lookup_issue(repo: str, number: int) -> dict[str, Any] | None:
    """Return issue ``number``, or None if it is a pull request or unknown."""
    try:
        issue = json.loads(gh(f"repos/{repo}/issues/{number}"))
    except subprocess.CalledProcessError as err:
        # Only a missing or deleted issue is "unknown", anything else (rate
        # limit, 5xx) must not turn into a "no issue linked" comment
        if "HTTP 404" not in err.stderr and "HTTP 410" not in err.stderr:
            raise
        return None
    return None if "pull_request" in issue else issue


def find_problems(
    repo: str, pr: dict[str, Any], open_prs: list[dict[str, Any]]
) -> list[str]:
    """List what a maintainer needs to sort out before reviewing ``pr``."""
    problems = list(template_problems(pr.get("body") or ""))

    issues = {
        number: issue
        for number in sorted(mentioned_numbers(pr, repo))
        if (issue := lookup_issue(repo, number))
    }
    if not issues:
        problems.append(
            "This PR does not reference an issue. Please open or find one "
            f"first ([CONTRIBUTING]({CONTRIBUTING_URL})) and link it in the "
            "description with `closes #<N>` or `refs #<N>`."
        )
        return problems

    for number in sorted(claimed_numbers(pr, repo) & issues.keys()):
        assignees = [a["login"] for a in issues[number].get("assignees", [])]
        if issues[number].get("state") == "closed":
            problems.append(
                f"#{number} is already closed, please link an open issue."
            )
        elif assignees and login(pr) not in assignees:
            names = ", ".join(f"`{a}`" for a in assignees)
            problems.append(
                f"#{number} is assigned to {names}. Please coordinate on the "
                "issue before continuing here."
            )
        for other in open_prs:
            if (
                other.get("number") != pr.get("number")
                and other["createdAt"] < pr["createdAt"]
                and login(other) != login(pr)
                and number in claimed_numbers(other, repo)
            ):
                problems.append(
                    f"#{other['number']} by `{login(other)}` (opened "
                    f"{other['createdAt'][:10]}) already addresses #{number}. "
                    "Please collaborate there instead of opening a competing "
                    "PR."
                )
    return problems


def render(problems: list[str]) -> str:
    """Build the bot comment for ``problems``."""
    if not problems:
        return f"{MARKER}\nThe issue-link check passes now, thanks!"
    items = "\n".join(f"- {p}" for p in problems)
    return (
        f"{MARKER}\nThanks for the PR! An automated check found something to "
        f"sort out before review:\n\n{items}\n\nThis comment updates when the "
        "PR is edited or pushed to. If a point does not apply, say so in a "
        "comment and a maintainer will have a look."
    )


def sync(repo: str, pr: dict[str, Any], problems: list[str]) -> None:
    """Create or update the bot comment and add or remove the label."""
    number = pr["number"]
    body = render(problems)
    existing = gh(
        f"repos/{repo}/issues/{number}/comments",
        "--paginate",
        "--jq",
        f'.[] | select(.user.login == "github-actions[bot]" and '
        f'(.body | startswith("{MARKER}"))) | .id',
    ).split()
    if existing:
        gh(
            "--method",
            "PATCH",
            f"repos/{repo}/issues/comments/{existing[0]}",
            "-f",
            f"body={body}",
        )
    elif problems:
        gh(
            "--method",
            "POST",
            f"repos/{repo}/issues/{number}/comments",
            "-f",
            f"body={body}",
        )

    labels_nodes = (pr.get("labels") or {}).get("nodes", [])
    has_label = LABEL in {label["name"] for label in labels_nodes}
    if problems and not has_label:
        try:
            gh(
                "--method",
                "POST",
                f"repos/{repo}/labels",
                "-f",
                f"name={LABEL}",
                "-f",
                "color=fbca04",
                "-f",
                "description=Issue link, assignment or overlap needs a look",
            )
        except subprocess.CalledProcessError:
            pass  # label already exists
        gh(
            "--method",
            "POST",
            f"repos/{repo}/issues/{number}/labels",
            "-f",
            f"labels[]={LABEL}",
        )
    elif not problems and has_label:
        try:
            gh(
                "--method",
                "DELETE",
                f"repos/{repo}/issues/{number}/labels/{LABEL}",
            )
        except subprocess.CalledProcessError as err:
            if "HTTP 404" not in err.stderr:  # label already removed
                raise


def main() -> None:
    """Check one pull request and report the result on it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr", type=int, required=True, help="PR number")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the comment instead of posting it",
    )
    args = parser.parse_args()

    open_prs = fetch_open_prs(args.repo)
    pr = next((p for p in open_prs if p["number"] == args.pr), None)
    if pr is None or is_bot(pr):
        return

    problems = find_problems(args.repo, pr, open_prs)
    if args.dry_run:
        print(render(problems))
    else:
        sync(args.repo, pr, problems)


if __name__ == "__main__":
    main()
