"""The agent-context files stay small, scoped and pointing at real files.

CLAUDE.md regrew a paragraph per ticket after KG-36 cut it, because plans,
review rounds and the PR template each asked for one more (KG-55). These
checks make the budget and the homes mechanical:

- the root CLAUDE.md is at most 150 lines and names every rule file and
  skill in its first 40 lines: a session that never read a matching file,
  wrote a new one (a Write does not load a rule) or lost the rule to
  compaction still sees the pointer;
- a rule file is at most 30 lines and path-scoped in the strict format
  below: an unscoped rule loads at launch and saves nothing, and a glob
  matching no listed file is a rule that never loads;
- every repo path an agent-context file names exists, written from the
  repo root;
- no agent-context file carries a ticket key: the why lives in docstrings.

The file list is git's -- tracked plus untracked, minus ignored -- so a file
created but not yet committed counts and an ignored one does not. Outside a
git checkout this fails rather than skips: a skip is not a pass.
"""

import glob
import re
import subprocess
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT_FILE = REPO / "CLAUDE.md"
ROOT_BUDGET = 150
RULE_BUDGET = 30
POINTER_WINDOW = 40
MIN_ROOT_PATH_CLAIMS = 10

_FENCE = re.compile(r"^```.*?^```[^\n]*$", re.MULTILINE | re.DOTALL)
_SPAN = re.compile(r"`([^`\n]+)`")
_PATH_CHARS = re.compile(r"[\w.\-/]+")
_PATH_SUFFIXES = (".py", ".md", ".toml", ".yml", ".yaml", ".json", ".lock")
_TICKET_KEY = re.compile(r"\bKG-\d+\b")
_FRONTMATTER = re.compile(r"\A---\npaths:\n((?:  - \"[^\"\n]+\"\n)+)---\n")
_PATTERN = re.compile(r'^  - "([^"\n]+)"$', re.MULTILINE)


@cache
def _listed_files() -> frozenset[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(
        path for path in result.stdout.split("\0") if path and (REPO / path).is_file()
    )


@cache
def _listed_directories() -> frozenset[str]:
    directories = set()
    for path in _listed_files():
        parts = path.split("/")
        for end in range(1, len(parts)):
            directories.add("/".join(parts[:end]) + "/")
    return frozenset(directories)


def _rule_files() -> list[Path]:
    return sorted((REPO / ".claude" / "rules").rglob("*.md"))


def _skill_files() -> list[Path]:
    return sorted((REPO / ".claude" / "skills").rglob("SKILL.md"))


def _agent_context_files() -> list[Path]:
    return [ROOT_FILE, *_rule_files(), *_skill_files()]


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _prose(text: str) -> str:
    """The text outside fenced blocks, where backticks pair within a line."""
    return _FENCE.sub("", text)


def _path_claims(text: str) -> list[str]:
    """Backticked spans that claim to be a repo file or directory.

    A span counts only if it is made of path characters (so no URL scheme,
    `KEY=value` or call), contains a slash without leading with one (so no
    API route), and ends in a slash or a known file suffix (so no dotted
    Python name such as `pkg/mod.function`).
    """
    return [
        span
        for span in _SPAN.findall(_prose(text))
        if _PATH_CHARS.fullmatch(span)
        and "/" in span
        and not span.startswith("/")
        and (span.endswith("/") or span.endswith(_PATH_SUFFIXES))
    ]


def test_root_claude_md_is_within_budget():
    lines = ROOT_FILE.read_text().splitlines()
    assert len(lines) <= ROOT_BUDGET, (
        f"CLAUDE.md is {len(lines)} lines, budget {ROOT_BUDGET}. Move area rules to "
        ".claude/rules/, reasons to docstrings (see CONTRIBUTING.md)."
    )


def test_there_are_rule_files_to_check():
    """Guards the checks below: with no rule files they would all pass."""
    assert _rule_files()


def test_every_rule_file_is_path_scoped_and_within_budget():
    wrong = []
    for rule in _rule_files():
        text = rule.read_text()
        if len(text.splitlines()) > RULE_BUDGET:
            wrong.append(f"{_relative(rule)}: over {RULE_BUDGET} lines")
        if not _FRONTMATTER.match(text):
            wrong.append(
                f"{_relative(rule)}: frontmatter must be exactly '---', 'paths:', "
                "one or more '  - \"<glob>\"' lines, '---'"
            )
    assert not wrong, "\n".join(wrong)


def test_every_rule_glob_matches_a_listed_file():
    wrong = []
    for rule in _rule_files():
        frontmatter = _FRONTMATTER.match(rule.read_text())
        if frontmatter is None:
            continue  # reported by the format test
        for pattern in _PATTERN.findall(frontmatter.group(1)):
            if "{" in pattern:
                wrong.append(f"{_relative(rule)}: {pattern!r} uses braces")
                continue
            matched = set(glob.glob(pattern, root_dir=REPO, recursive=True))
            if not matched & _listed_files():
                wrong.append(f"{_relative(rule)}: {pattern!r} matches no listed file")
    assert not wrong, "\n".join(wrong)


def test_no_backtick_span_crosses_a_line():
    """A span split over two lines shifts the pairing of every later backtick,
    so the path check below would silently read the wrong spans."""
    wrong = [
        f"{_relative(path)}:{number}"
        for path in _agent_context_files()
        for number, line in enumerate(_prose(path.read_text()).splitlines(), 1)
        if line.count("`") % 2
    ]
    assert not wrong, "odd number of backticks on: " + ", ".join(wrong)


def test_the_path_reader_finds_the_root_claims():
    """Guards the reader: one that found nothing would pass the check below."""
    assert len(_path_claims(ROOT_FILE.read_text())) >= MIN_ROOT_PATH_CLAIMS


def test_every_named_repo_path_exists():
    known = _listed_files() | _listed_directories()
    missing = [
        f"{_relative(path)}: `{claim}`"
        for path in _agent_context_files()
        for claim in _path_claims(path.read_text())
        if claim not in known
    ]
    assert not missing, (
        "write paths from the repo root, and name gitignored places in prose:\n"
        + "\n".join(missing)
    )


def test_no_agent_context_file_carries_a_ticket_key():
    found = [
        f"{_relative(path)}: {key}"
        for path in _agent_context_files()
        for key in _TICKET_KEY.findall(path.read_text())
    ]
    assert not found, "ticket keys belong in docstrings:\n" + "\n".join(found)


def test_root_points_at_every_rule_and_skill_early():
    head = "\n".join(ROOT_FILE.read_text().splitlines()[:POINTER_WINDOW])
    expected = [_relative(path) for path in [*_rule_files(), *_skill_files()]]
    missing = [path for path in expected if path not in head]
    assert not missing, (
        f"not named in CLAUDE.md's first {POINTER_WINDOW} lines: {missing}"
    )
