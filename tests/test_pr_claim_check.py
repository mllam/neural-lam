"""Tests for the pull request claim check script."""

# Standard library
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Third-party
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "pr_claim_check.py"
)
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "pull_request_template.md"
)

spec = importlib.util.spec_from_file_location("pr_claim_check", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
pr_claim_check = importlib.util.module_from_spec(spec)
sys.modules["pr_claim_check"] = pr_claim_check
spec.loader.exec_module(pr_claim_check)

REPO = "mllam/neural-lam"


def make_pr(
    number: int = 1,
    author: str = "alice",
    created_at: str = "2026-01-02T00:00:00Z",
    body: str = "",
    labels: list[str] | None = None,
    closing_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Helper to construct a PR dict fixture."""
    return {
        "number": number,
        "author": {"login": author, "__typename": "User"},
        "createdAt": created_at,
        "body": body,
        "labels": {"nodes": [{"name": name} for name in (labels or [])]},
        "closingIssuesReferences": {"nodes": closing_refs or []},
    }


def test_login():
    """Test extracting login handles or ghost fallback."""
    assert pr_claim_check.login({"author": {"login": "alice"}}) == "alice"
    assert pr_claim_check.login({"author": None}) == "ghost"
    assert pr_claim_check.login({}) == "ghost"


def test_is_bot():
    """Test identifying bot authors."""
    assert pr_claim_check.is_bot(
        {"author": {"login": "bot", "__typename": "Bot"}}
    )
    assert pr_claim_check.is_bot(
        {"author": {"login": "app[bot]", "__typename": "User"}}
    )
    assert pr_claim_check.is_bot(
        {"author": {"login": "dependabot", "__typename": "User"}}
    )
    assert not pr_claim_check.is_bot(
        {"author": {"login": "alice", "__typename": "User"}}
    )


def test_template_problems_real_template():
    """Verify that unmodified PR template is detected as unfilled."""
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    problems = pr_claim_check.template_problems(template_text)
    assert len(problems) == 1
    assert "placeholder text" in problems[0]


def test_template_problems():
    """Test detection of empty bodies or unfilled template placeholders."""
    assert pr_claim_check.template_problems("") == [
        "The PR description is empty. Please fill out the pull request "
        "template describing the changes and motivation."
    ]
    assert pr_claim_check.template_problems("   \n\t  ") == [
        "The PR description is empty. Please fill out the pull request "
        "template describing the changes and motivation."
    ]

    body_with_summary = (
        "## Describe your changes\n\n< Summary of the changes.>\n\ncloses #12"
    )
    assert len(pr_claim_check.template_problems(body_with_summary)) == 1

    clean_body = (
        "This fixes the memory leak by releasing CUDA cache.\n\nCloses #12"
    )
    assert pr_claim_check.template_problems(clean_body) == []


def test_issue_ref_regex():
    """Test matching issue references and skipping hex colors/other repos."""
    ref_pattern = pr_claim_check.issue_ref(REPO)
    # Standard library
    import re

    # Direct issue and full repo issue
    assert re.findall(ref_pattern, "Fixes #12 and mllam/neural-lam#34") == [
        "12",
        "34",
    ]
    assert re.findall(
        ref_pattern, "See https://github.com/mllam/neural-lam/issues/56"
    ) == ["56"]

    # Hex colors with letters (e.g. mermaid stroke:#27ae60) must NOT match
    assert re.findall(ref_pattern, "stroke:#27ae60 fill:#e74c3c") == []

    # Other repo refs must NOT match
    assert (
        re.findall(
            ref_pattern,
            "mllam/weather-model-graphs#88 or other/neural-lam#99",
        )
        == []
    )


def test_claimed_and_mentioned_numbers():
    """Test issue claim extraction vs generic mentions."""
    pr = make_pr(
        body=(
            "Fixes #101, also resolves "
            "https://github.com/mllam/neural-lam/issues/102. closes #00. "
            "Refs #201 and mentions mllam/weather-model-graphs#88. "
            "Color is #27ae60."
        ),
        closing_refs=[
            {"number": 103, "repository": {"nameWithOwner": REPO}},
            {"number": 999, "repository": {"nameWithOwner": "other/repo"}},
        ],
    )
    assert pr_claim_check.claimed_numbers(pr, REPO) == {101, 102, 103}
    assert pr_claim_check.mentioned_numbers(pr, REPO) == {101, 102, 103, 201}


def test_gh_calls_and_error_handling(capsys):
    """Test that gh() prints stderr before re-raising CalledProcessError."""
    err = subprocess.CalledProcessError(
        1, "gh", stderr="GraphQL error: timeout"
    )
    with patch("subprocess.run", side_effect=err):
        with pytest.raises(subprocess.CalledProcessError):
            pr_claim_check.gh("api", "graphql")
        captured = capsys.readouterr().out
        assert "GraphQL error: timeout" in captured


def test_lookup_issue():
    """Test looking up issue metadata, filtering PRs, and error branches."""
    with patch("pr_claim_check.gh") as mock_gh:
        mock_gh.return_value = json.dumps({"number": 12, "title": "Bug"})
        assert pr_claim_check.lookup_issue(REPO, 12) == {
            "number": 12,
            "title": "Bug",
        }

    with patch("pr_claim_check.gh") as mock_gh:
        mock_gh.return_value = json.dumps(
            {"number": 14, "title": "PR", "pull_request": {"url": "..."}}
        )
        assert pr_claim_check.lookup_issue(REPO, 14) is None

    # 404 / 410 -> None
    err_404 = subprocess.CalledProcessError(
        1, "gh", stderr="HTTP 404: Not Found"
    )
    with patch("pr_claim_check.gh", side_effect=err_404):
        assert pr_claim_check.lookup_issue(REPO, 999) is None

    err_410 = subprocess.CalledProcessError(1, "gh", stderr="HTTP 410: Gone")
    with patch("pr_claim_check.gh", side_effect=err_410):
        assert pr_claim_check.lookup_issue(REPO, 999) is None

    # 403 / 500 -> Must raise
    err_500 = subprocess.CalledProcessError(
        1, "gh", stderr="HTTP 500: Server Error"
    )
    with patch("pr_claim_check.gh", side_effect=err_500):
        with pytest.raises(subprocess.CalledProcessError):
            pr_claim_check.lookup_issue(REPO, 999)


@pytest.mark.parametrize(
    ("test_id", "pr", "open_prs", "issues", "expected_substrings"),
    [
        (
            "no_issue_referenced",
            make_pr(body="Typo fix"),
            [],
            {},
            ["This PR does not reference an issue."],
        ),
        (
            "only_pr_referenced",
            make_pr(body="Depends on #50"),
            [],
            {},  # #50 is not an issue (lookup_issue returns None)
            ["This PR does not reference an issue."],
        ),
        (
            "template_unfilled_and_no_issue",
            make_pr(body="< Summary of the changes.>"),
            [],
            {},
            [
                "placeholder text from the template",
                "This PR does not reference an issue.",
            ],
        ),
        (
            "clean_pr_passes",
            make_pr(body="Added feature. Closes #10"),
            [],
            {10: {"number": 10, "state": "open", "assignees": []}},
            [],
        ),
        (
            "closed_issue_claimed",
            make_pr(body="Fixes #10"),
            [],
            {10: {"number": 10, "state": "closed", "assignees": []}},
            ["#10 is already closed, please link an open issue."],
        ),
        (
            "closed_issue_mentioned_only",
            make_pr(body="Refs #10 for context"),
            [],
            {10: {"number": 10, "state": "closed", "assignees": []}},
            [],  # Mentions alone do not fail on closed
        ),
        (
            "assigned_issue_claimed_other_author",
            make_pr(author="alice", body="Closes #20"),
            [],
            {
                20: {
                    "number": 20,
                    "state": "open",
                    "assignees": [{"login": "bob"}, {"login": "carol"}],
                }
            },
            ["#20 is assigned to `bob`, `carol`"],
        ),
        (
            "assigned_issue_claimed_same_author",
            make_pr(author="alice", body="Closes #20"),
            [],
            {
                20: {
                    "number": 20,
                    "state": "open",
                    "assignees": [{"login": "alice"}],
                }
            },
            [],
        ),
        (
            "assigned_issue_mentioned_only",
            make_pr(author="alice", body="Refs #20"),
            [],
            {
                20: {
                    "number": 20,
                    "state": "open",
                    "assignees": [{"login": "bob"}],
                }
            },
            [],
        ),
        (
            "competing_older_pr_claims",
            make_pr(
                number=50,
                author="alice",
                created_at="2026-01-02T12:00:00Z",
                body="Closes #30",
            ),
            [
                make_pr(
                    number=40,
                    author="bob",
                    created_at="2026-01-01T10:00:00Z",
                    body="Fixes #30",
                ),
            ],
            {30: {"number": 30, "state": "open", "assignees": []}},
            ["#40 by `bob` (opened 2026-01-01) already addresses #30"],
        ),
        (
            "competing_older_pr_same_author",
            make_pr(
                number=50,
                author="alice",
                created_at="2026-01-02T12:00:00Z",
                body="Closes #30",
            ),
            [
                make_pr(
                    number=40,
                    author="alice",
                    created_at="2026-01-01T10:00:00Z",
                    body="Fixes #30",
                ),
            ],
            {30: {"number": 30, "state": "open", "assignees": []}},
            [],  # Same author updating PR does not conflict
        ),
        (
            "competing_newer_pr",
            make_pr(
                number=50,
                author="alice",
                created_at="2026-01-01T10:00:00Z",
                body="Closes #30",
            ),
            [
                make_pr(
                    number=60,
                    author="bob",
                    created_at="2026-01-02T12:00:00Z",
                    body="Fixes #30",
                ),
            ],
            {30: {"number": 30, "state": "open", "assignees": []}},
            [],  # Newer PR does not flag earlier PR
        ),
        (
            "competing_older_pr_only_mentions",
            make_pr(
                number=50,
                author="alice",
                created_at="2026-01-02T12:00:00Z",
                body="Closes #30",
            ),
            [
                make_pr(
                    number=40,
                    author="bob",
                    created_at="2026-01-01T10:00:00Z",
                    body="Refs #30",  # Not a claim!
                ),
            ],
            {30: {"number": 30, "state": "open", "assignees": []}},
            [],  # Older PR only referenced it, so not competing claim
        ),
    ],
)
def test_find_problems_table(
    test_id, pr, open_prs, issues, expected_substrings
):
    """Parametrized verification of find_problems across all edge cases."""
    all_open_prs = [pr, *open_prs]
    with patch(
        "pr_claim_check.lookup_issue", side_effect=lambda _, n: issues.get(n)
    ):
        problems = pr_claim_check.find_problems(REPO, pr, all_open_prs)
        if not expected_substrings:
            assert problems == [], f"Expected no problems for {test_id}"
        else:
            for expected in expected_substrings:
                assert any(expected in p for p in problems), (
                    f"Missing '{expected}' in problems: {problems} for "
                    f"{test_id}"
                )


def test_fetch_open_prs_pagination():
    """Test paginated GraphQL requests in fetch_open_prs."""
    page1 = {
        "data": {
            "repository": {
                "pullRequests": {
                    "pageInfo": {"hasNextPage": True, "endCursor": "cursor1"},
                    "nodes": [{"number": 1}],
                }
            }
        }
    }
    page2 = {
        "data": {
            "repository": {
                "pullRequests": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                    "nodes": [{"number": 2}],
                }
            }
        }
    }
    with patch(
        "pr_claim_check.gh", side_effect=[json.dumps(page1), json.dumps(page2)]
    ) as mock_gh:
        prs = pr_claim_check.fetch_open_prs(REPO)
        assert len(prs) == 2
        assert [p["number"] for p in prs] == [1, 2]
        assert mock_gh.call_count == 2


def test_render():
    """Test markdown rendering of problems and pass status."""
    pass_msg = pr_claim_check.render([])
    assert pr_claim_check.MARKER in pass_msg
    assert "passes now" in pass_msg

    fail_msg = pr_claim_check.render(["Issue missing", "Template unedited"])
    assert pr_claim_check.MARKER in fail_msg
    assert "- Issue missing" in fail_msg
    assert "- Template unedited" in fail_msg
    assert "If a point does not apply" in fail_msg


def test_sync_new_problem_adds_comment_and_label():
    """Test creating comment and label on new failure."""
    pr = make_pr(number=77, labels=[])
    with patch("pr_claim_check.gh") as mock_gh:
        mock_gh.return_value = ""  # No existing bot comments
        pr_claim_check.sync(REPO, pr, ["Problem 1"])

        calls = [c.args for c in mock_gh.call_args_list]
        # Check comment creation
        assert any(
            c[:3] == ("--method", "POST", f"repos/{REPO}/issues/77/comments")
            for c in calls
        )
        # Check label added
        assert any(
            c[:3] == ("--method", "POST", f"repos/{REPO}/issues/77/labels")
            for c in calls
        )


def test_sync_resolved_updates_comment_and_removes_label():
    """Test updating existing comment and removing label on resolution."""
    pr = make_pr(number=77, labels=[pr_claim_check.LABEL])
    with patch("pr_claim_check.gh") as mock_gh:

        def gh_side_effect(*args, **kwargs):
            if args[0] == f"repos/{REPO}/issues/77/comments":
                return "12345\n"
            return ""

        mock_gh.side_effect = gh_side_effect
        pr_claim_check.sync(REPO, pr, [])

        calls = [c.args for c in mock_gh.call_args_list]
        # Check existing comment patched
        assert any(
            c[:3]
            == ("--method", "PATCH", f"repos/{REPO}/issues/comments/12345")
            for c in calls
        )
        # Check label deleted
        assert any(
            c[:3]
            == (
                "--method",
                "DELETE",
                f"repos/{REPO}/issues/77/labels/{pr_claim_check.LABEL}",
            )
            for c in calls
        )


def test_sync_label_delete_error_handling():
    """Test 404 on label DELETE is ignored while other errors are raised."""
    pr = make_pr(number=77, labels=[pr_claim_check.LABEL])
    err_404 = subprocess.CalledProcessError(1, "gh", stderr="HTTP 404")
    err_500 = subprocess.CalledProcessError(1, "gh", stderr="HTTP 500")

    # 404 is ignored
    with patch("pr_claim_check.gh") as mock_gh:

        def gh_side_effect(*args, **kwargs):
            if args[0] == f"repos/{REPO}/issues/77/comments":
                return "12345\n"
            if args[1] == "DELETE":
                raise err_404
            return ""

        mock_gh.side_effect = gh_side_effect
        pr_claim_check.sync(REPO, pr, [])  # Should not raise

    # 500 raises
    with patch("pr_claim_check.gh") as mock_gh:

        def gh_side_effect(*args, **kwargs):
            if args[0] == f"repos/{REPO}/issues/77/comments":
                return "12345\n"
            if args[1] == "DELETE":
                raise err_500
            return ""

        mock_gh.side_effect = gh_side_effect
        with pytest.raises(subprocess.CalledProcessError):
            pr_claim_check.sync(REPO, pr, [])


def test_main_bot_skip():
    """Test that bots are skipped in main."""
    open_prs = [
        {
            "number": 99,
            "author": {"login": "dependabot[bot]", "__typename": "Bot"},
        }
    ]
    with (
        patch("pr_claim_check.fetch_open_prs", return_value=open_prs),
        patch(
            "sys.argv",
            ["pr_claim_check.py", "--repo", REPO, "--pr", "99"],
        ),
        patch("pr_claim_check.sync") as mock_sync,
    ):
        pr_claim_check.main()
        mock_sync.assert_not_called()


def test_main_dry_run(capsys):
    """Test dry-run prints comment without syncing."""
    pr = make_pr(number=100, body="Fixes #10")
    issues = {10: {"number": 10, "state": "open", "assignees": []}}
    with (
        patch("pr_claim_check.fetch_open_prs", return_value=[pr]),
        patch(
            "pr_claim_check.lookup_issue",
            side_effect=lambda _, n: issues.get(n),
        ),
        patch("pr_claim_check.sync") as mock_sync,
        patch(
            "sys.argv",
            [
                "pr_claim_check.py",
                "--repo",
                REPO,
                "--pr",
                "100",
                "--dry-run",
            ],
        ),
    ):
        pr_claim_check.main()
        mock_sync.assert_not_called()
        captured = capsys.readouterr().out
        assert pr_claim_check.MARKER in captured
        assert "passes now" in captured
