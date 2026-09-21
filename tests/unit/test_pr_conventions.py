"""Tests for the PR convention check in .github/scripts/check_pr_conventions.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "check_pr_conventions.py"
_spec = importlib.util.spec_from_file_location("check_pr_conventions", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
conventions = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = conventions
_spec.loader.exec_module(conventions)

PullRequest = conventions.PullRequest
check = conventions.check

CHANGELOG_0_7_0 = """# Changelog

## [0.7.0] - 2026-09-21

### Added
- Something (#64, KG-21).

## [0.6.1] - 2026-09-18
"""


def _pr(
    title="KG-21: Stream the column checks", branch="streaming", labels=(), author="me"
):
    return PullRequest(
        title=title, branch=branch, author=author, labels=frozenset(labels)
    )


def _check(pr, base="0.6.1", head="0.7.0", lock="0.7.0", changelog=CHANGELOG_0_7_0):
    return check(
        pr, base_version=base, head_version=head, lock_version=lock, changelog=changelog
    )


def test_a_ticketed_release_passes():
    assert _check(_pr()) == []


def test_the_jira_key_may_come_from_the_branch_name():
    assert _check(_pr(title="Stream the column checks", branch="KG-21-streaming")) == []


def test_a_missing_jira_key_is_reported():
    problems = _check(_pr(title="Stream the column checks", branch="streaming"))
    assert len(problems) == 1
    assert "No Jira key" in problems[0]


def test_a_lookalike_key_does_not_count():
    assert _check(_pr(title="XKG-21 streaming", branch="streaming")) != []


def test_no_ticket_label_waives_the_jira_key():
    assert _check(_pr(title="Tidy docs", labels={"no-ticket"})) == []


@pytest.mark.parametrize("head", ["0.6.2", "0.7.0", "1.0.0"])
def test_any_next_version_is_a_valid_release(head):
    changelog = CHANGELOG_0_7_0.replace("0.7.0", head)
    assert _check(_pr(), head=head, lock=head, changelog=changelog) == []


@pytest.mark.parametrize(
    "head", ["0.6.1", "0.6.3", "0.8.0", "0.7.1", "0.6.0", "0.7.0.1"]
)
def test_an_unchanged_skipped_or_backwards_version_is_refused(head):
    problems = _check(_pr(), head=head, lock=head)
    assert len(problems) == 1
    assert "no-release" in problems[0]


def test_a_stale_uv_lock_is_reported():
    problems = _check(_pr(), lock="0.6.1")
    assert problems == ["uv.lock records agentic-kg 0.6.1, not 0.7.0: run `uv lock`."]


def test_an_entry_left_under_unreleased_is_reported():
    changelog = CHANGELOG_0_7_0.replace("## [0.7.0] - 2026-09-21", "## [Unreleased]")
    problems = _check(_pr(), changelog=changelog)
    assert len(problems) == 1
    assert "topmost section is [Unreleased]" in problems[0]


def test_an_undated_release_heading_is_reported():
    changelog = CHANGELOG_0_7_0.replace(" - 2026-09-21", "")
    assert _check(_pr(), changelog=changelog) == [
        "CHANGELOG.md's [0.7.0] heading needs a date."
    ]


def test_no_release_label_passes_when_the_version_is_unchanged():
    assert _check(_pr(labels={"no-release"}), head="0.6.1", lock="0.6.1") == []


def test_no_release_label_refuses_a_bump():
    problems = _check(_pr(labels={"no-release"}))
    assert len(problems) == 1
    assert "drop the label or the bump" in problems[0]


def test_dependabot_is_exempt():
    pr = _pr(title="chore(deps): bump idna", author="dependabot[bot]")
    assert _check(pr, head="0.6.1", lock="0.6.1") == []


def test_both_problems_are_reported_together():
    problems = _check(
        _pr(title="Stream", branch="streaming"), head="0.6.1", lock="0.6.1"
    )
    assert len(problems) == 2
