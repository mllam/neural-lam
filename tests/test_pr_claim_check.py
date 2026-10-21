"""Tests for the pull request claim check script."""

# Standard library
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

# Third-party

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "pr_claim_check.py"
)
spec = importlib.util.spec_from_file_location("pr_claim_check", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
pr_claim_check = importlib.util.module_from_spec(spec)
sys.modules["pr_claim_check"] = pr_claim_check
spec.loader.exec_module(pr_claim_check)


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
    assert (
        "placeholder text"
        in pr_claim_check.template_problems(body_with_summary)[0]
    )

    body_with_link_placeholder = (
        "Fixed the issue.\n\n< Link to the relevant issue or task > closes #12"
    )
    assert (
        len(pr_claim_check.template_problems(body_with_link_placeholder)) == 1
    )

    clean_body = (
        "This fixes the memory leak by releasing CUDA cache.\n\nCloses #12"
    )
    assert pr_claim_check.template_problems(clean_body) == []


def test_claimed_numbers():
    """Test issue claim extraction from keywords, URLs, and GraphQL nodes."""
    repo = "mllam/neural-lam"
    issue_url = "https://github.com/mllam/neural-lam/issues/102"
    pr = {
        "body": f"Fixes #101, also resolves {issue_url}. closes #00.",
        "closingIssuesReferences": {"nodes": [{"number": 103}]},
    }
    claims = pr_claim_check.claimed_numbers(pr, repo)
    assert claims == {101, 102, 103}

    # Mentions should not be treated as claims
    pr_mention = {
        "body": "This is part of #201 and refs #202. solves #203.",
        "closingIssuesReferences": {"nodes": []},
    }
    assert pr_claim_check.claimed_numbers(pr_mention, repo) == {203}


def test_mentioned_numbers():
    """Test extracting all referenced issues including mentions and claims."""
    repo = "mllam/neural-lam"
    pr = {
        "body": "Working on #42 and refs #43. Closes #44 and #00.",
        "closingIssuesReferences": {"nodes": []},
    }
    assert pr_claim_check.mentioned_numbers(pr, repo) == {42, 43, 44}


def test_lookup_issue():
    """Test looking up issue metadata and filtering out pull requests."""
    repo = "mllam/neural-lam"

    with patch("pr_claim_check.gh") as mock_gh:
        mock_gh.return_value = json.dumps({"number": 12, "title": "Bug"})
        issue = pr_claim_check.lookup_issue(repo, 12)
        assert issue == {"number": 12, "title": "Bug"}

    with patch("pr_claim_check.gh") as mock_gh:
        mock_gh.return_value = json.dumps(
            {"number": 14, "title": "PR", "pull_request": {"url": "..."}}
        )
        assert pr_claim_check.lookup_issue(repo, 14) is None

    with patch("pr_claim_check.gh") as mock_gh:
        mock_gh.side_effect = subprocess.CalledProcessError(1, "gh")
        assert pr_claim_check.lookup_issue(repo, 999) is None


def test_find_problems_missing_issue():
    """Test reporting error when no issue is referenced."""
    repo = "mllam/neural-lam"
    pr = {
        "number": 50,
        "author": {"login": "alice"},
        "createdAt": "2026-01-02T00:00:00Z",
        "body": "Fixed a typo in the documentation.",
        "closingIssuesReferences": {"nodes": []},
    }
    with patch("pr_claim_check.lookup_issue", return_value=None):
        problems = pr_claim_check.find_problems(repo, pr, [pr])
        assert len(problems) == 1
        assert "This PR does not reference an issue" in problems[0]


def test_find_problems_closed_issue():
    """Test reporting error when claimed issue is closed."""
    repo = "mllam/neural-lam"
    pr = {
        "number": 51,
        "author": {"login": "alice"},
        "createdAt": "2026-01-02T00:00:00Z",
        "body": "Implemented feature. Closes #10.",
        "closingIssuesReferences": {"nodes": []},
    }
    issues = {10: {"number": 10, "state": "closed", "assignees": []}}
    with patch(
        "pr_claim_check.lookup_issue", side_effect=lambda _, n: issues.get(n)
    ):
        problems = pr_claim_check.find_problems(repo, pr, [pr])
        assert "#10 is already closed." in problems


def test_find_problems_assigned_issue():
    """Test reporting error when claimed issue is assigned to someone else."""
    repo = "mllam/neural-lam"
    pr = {
        "number": 52,
        "author": {"login": "alice"},
        "createdAt": "2026-01-02T00:00:00Z",
        "body": "Implemented feature. Closes #20.",
        "closingIssuesReferences": {"nodes": []},
    }
    issues = {
        20: {
            "number": 20,
            "state": "open",
            "assignees": [{"login": "bob"}, {"login": "carol"}],
        }
    }
    with patch(
        "pr_claim_check.lookup_issue", side_effect=lambda _, n: issues.get(n)
    ):
        problems = pr_claim_check.find_problems(repo, pr, [pr])
        assert any("#20 is assigned to `bob`, `carol`" in p for p in problems)

    # When PR author is an assignee, no problem
    issues[20]["assignees"].append({"login": "alice"})
    with patch(
        "pr_claim_check.lookup_issue", side_effect=lambda _, n: issues.get(n)
    ):
        problems = pr_claim_check.find_problems(repo, pr, [pr])
        assert not any("assigned" in p for p in problems)


def test_find_problems_competing_older_pr():
    """Test reporting error when an earlier open PR claims the same issue."""
    repo = "mllam/neural-lam"
    earlier_pr = {
        "number": 40,
        "author": {"login": "bob"},
        "createdAt": "2026-01-01T10:00:00Z",
        "body": "Resolves #30",
        "closingIssuesReferences": {"nodes": []},
    }
    current_pr = {
        "number": 53,
        "author": {"login": "alice"},
        "createdAt": "2026-01-02T12:00:00Z",
        "body": "Fixes #30",
        "closingIssuesReferences": {"nodes": []},
    }
    issues = {30: {"number": 30, "state": "open", "assignees": []}}
    with patch(
        "pr_claim_check.lookup_issue", side_effect=lambda _, n: issues.get(n)
    ):
        problems = pr_claim_check.find_problems(
            repo, current_pr, [earlier_pr, current_pr]
        )
        assert any(
            "#40 by `bob` (opened 2026-01-01) already addresses #30" in p
            for p in problems
        )

    # Same author should not trigger a competing PR conflict (stacked PR)
    earlier_pr_same_author = {
        "number": 41,
        "author": {"login": "alice"},
        "createdAt": "2026-01-01T10:00:00Z",
        "body": "Resolves #30",
        "closingIssuesReferences": {"nodes": []},
    }
    with patch(
        "pr_claim_check.lookup_issue", side_effect=lambda _, n: issues.get(n)
    ):
        problems = pr_claim_check.find_problems(
            repo, current_pr, [earlier_pr_same_author, current_pr]
        )
        assert not any("already addresses" in p for p in problems)


def test_render():
    """Test markdown rendering of problems and pass status."""
    pass_msg = pr_claim_check.render([])
    assert pr_claim_check.MARKER in pass_msg
    assert "passes now" in pass_msg

    fail_msg = pr_claim_check.render(["Issue missing", "Template unedited"])
    assert pr_claim_check.MARKER in fail_msg
    assert "- Issue missing" in fail_msg
    assert "- Template unedited" in fail_msg


def test_sync():
    """Test synchronizing comments and labels via gh calls."""
    repo = "mllam/neural-lam"
    pr = {
        "number": 77,
        "labels": {"nodes": []},
    }

    # Case 1: New failure -> post comment and add label
    with patch("pr_claim_check.gh") as mock_gh:
        mock_gh.side_effect = lambda *args, **kwargs: ""
        pr_claim_check.sync(repo, pr, ["Problem 1"])

        calls = [c.args for c in mock_gh.call_args_list]
        # Querying comments
        assert calls[0][0] == f"repos/{repo}/issues/77/comments"
        # Creating comment
        assert calls[1][0:3] == (
            "--method",
            "POST",
            f"repos/{repo}/issues/77/comments",
        )
        # Creating label if needed
        assert calls[2][0:3] == ("--method", "POST", f"repos/{repo}/labels")
        # Adding label to PR
        assert calls[3][0:3] == (
            "--method",
            "POST",
            f"repos/{repo}/issues/77/labels",
        )

    # Case 2: Existing comment updated, label already present
    pr_with_label = {
        "number": 77,
        "labels": {"nodes": [{"name": pr_claim_check.LABEL}]},
    }
    with patch("pr_claim_check.gh") as mock_gh:

        def gh_side_effect(*args, **kwargs):
            if args[0] == f"repos/{repo}/issues/77/comments":
                return "12345\n"
            return ""

        mock_gh.side_effect = gh_side_effect
        pr_claim_check.sync(repo, pr_with_label, ["Still broken"])

        calls = [c.args for c in mock_gh.call_args_list]
        assert calls[1][0:3] == (
            "--method",
            "PATCH",
            f"repos/{repo}/issues/comments/12345",
        )

    # Case 3: Problems resolved -> remove label, update comment
    with patch("pr_claim_check.gh") as mock_gh:

        def gh_side_effect(*args, **kwargs):
            if args[0] == f"repos/{repo}/issues/77/comments":
                return "12345\n"
            return ""

        mock_gh.side_effect = gh_side_effect
        pr_claim_check.sync(repo, pr_with_label, [])

        calls = [c.args for c in mock_gh.call_args_list]
        # Comment updated with success
        assert calls[1][0:3] == (
            "--method",
            "PATCH",
            f"repos/{repo}/issues/comments/12345",
        )
        # Label deleted
        assert calls[2][0:3] == (
            "--method",
            "DELETE",
            f"repos/{repo}/issues/77/labels/{pr_claim_check.LABEL}",
        )


def test_main_bot_skip(capsys):
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
            ["pr_claim_check.py", "--repo", "mllam/neural-lam", "--pr", "99"],
        ),
        patch("pr_claim_check.sync") as mock_sync,
    ):
        pr_claim_check.main()
        mock_sync.assert_not_called()


def test_main_dry_run(capsys):
    """Test dry-run prints comment without syncing."""
    open_prs = [
        {
            "number": 100,
            "author": {"login": "alice", "__typename": "User"},
            "createdAt": "2026-01-01T00:00:00Z",
            "body": "Fixes #10",
            "labels": {"nodes": []},
            "closingIssuesReferences": {"nodes": []},
        }
    ]
    issues = {10: {"number": 10, "state": "open", "assignees": []}}
    with (
        patch("pr_claim_check.fetch_open_prs", return_value=open_prs),
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
                "mllam/neural-lam",
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
