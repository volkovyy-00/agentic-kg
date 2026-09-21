#!/usr/bin/env python3
"""Check a pull request against the conventions in CONTRIBUTING.md.

Two rules, each with an explicit opt-out label:

- A Jira key (``KG-123``) in the PR title or branch name, unless labelled ``no-ticket``.
- A release: ``pyproject.toml``'s version is the next patch, minor or major version after
  the base branch's, ``uv.lock`` agrees, and ``CHANGELOG.md``'s topmost section is that
  version — unless labelled ``no-release``, in which case the version must not move.

Dependabot's PRs are exempt from both: it can neither name a ticket nor bump a version.

Run by ``.github/workflows/pr-conventions.yml``, which passes the PR's details through
the environment. The checking itself is the pure ``check()``, covered by
``tests/unit/test_pr_conventions.py``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

JIRA_KEY = re.compile(r"\bKG-\d+\b")
VERSION_HEADING = re.compile(r"^## \[([^\]]+)\](?: - (\d{4}-\d{2}-\d{2}))?\s*$", re.M)
EXEMPT_AUTHORS = {"dependabot[bot]"}
NO_TICKET = "no-ticket"
NO_RELEASE = "no-release"

Version = tuple[int, int, int]


@dataclass(frozen=True)
class PullRequest:
    title: str
    branch: str
    author: str
    labels: frozenset[str]


def parse_version(text: str) -> Version | None:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text.strip())
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def next_versions(base: Version) -> set[Version]:
    major, minor, patch = base
    return {(major, minor, patch + 1), (major, minor + 1, 0), (major + 1, 0, 0)}


def check(
    pr: PullRequest,
    base_version: str,
    head_version: str,
    lock_version: str | None,
    changelog: str,
) -> list[str]:
    """Return every convention the PR breaks, as messages; empty means it passes."""
    if pr.author in EXEMPT_AUTHORS:
        return []
    problems = []
    if NO_TICKET not in pr.labels and not (
        JIRA_KEY.search(pr.title) or JIRA_KEY.search(pr.branch)
    ):
        problems.append(
            "No Jira key: put the ticket (e.g. KG-21) in the PR title or branch name, "
            f"or add the '{NO_TICKET}' label if there is no ticket."
        )
    if NO_RELEASE in pr.labels:
        if head_version != base_version:
            problems.append(
                f"Labelled '{NO_RELEASE}' but pyproject.toml's version moved "
                f"{base_version} -> {head_version}: drop the label or the bump."
            )
        return problems
    problems.extend(
        _release_problems(base_version, head_version, lock_version, changelog)
    )
    return problems


def _release_problems(
    base_version: str, head_version: str, lock_version: str | None, changelog: str
) -> list[str]:
    opt_out = (
        f" (or add the '{NO_RELEASE}' label if this PR changes nothing user-visible)"
    )
    base, head = parse_version(base_version), parse_version(head_version)
    if base is None:
        return [f"The base branch's version {base_version!r} is not MAJOR.MINOR.PATCH."]
    if head is None or head not in next_versions(base):
        allowed = ", ".join(".".join(map(str, v)) for v in sorted(next_versions(base)))
        return [
            f"pyproject.toml's version is {head_version}; a release from {base_version} "
            f"must be one of {allowed}{opt_out}."
        ]
    problems = []
    if lock_version != head_version:
        problems.append(
            f"uv.lock records agentic-kg {lock_version}, not {head_version}: run `uv lock`."
        )
    headings = VERSION_HEADING.findall(changelog)
    if not headings or headings[0][0] != head_version:
        found = headings[0][0] if headings else "none"
        problems.append(
            f"CHANGELOG.md's topmost section is [{found}]; it must be "
            f"'## [{head_version}] - YYYY-MM-DD' holding this PR's entries."
        )
    elif not headings[0][1]:
        problems.append(f"CHANGELOG.md's [{head_version}] heading needs a date.")
    return problems


def _pyproject_version(text: str) -> str:
    return tomllib.loads(text)["project"]["version"]


def _lock_version(text: str) -> str | None:
    for package in tomllib.loads(text).get("package", []):
        if package.get("name") == "agentic-kg":
            return package.get("version")
    return None


def main() -> int:
    env = os.environ
    pr = PullRequest(
        title=env["PR_TITLE"],
        branch=env["PR_BRANCH"],
        author=env["PR_AUTHOR"],
        labels=frozenset(json.loads(env.get("PR_LABELS") or "[]")),
    )
    base_pyproject = subprocess.run(
        ["git", "show", f"{env['BASE_SHA']}:pyproject.toml"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    problems = check(
        pr,
        base_version=_pyproject_version(base_pyproject),
        head_version=_pyproject_version(Path("pyproject.toml").read_text()),
        lock_version=_lock_version(Path("uv.lock").read_text()),
        changelog=Path("CHANGELOG.md").read_text(),
    )
    for problem in problems:
        print(f"::error::{problem}")
    if not problems:
        print("PR follows the ticket and release conventions.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
