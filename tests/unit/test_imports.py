"""Every module in the package must import cleanly.

This guards against the class of defect where a module is broken for months
because nothing imports it.

Modules are discovered from the filesystem, NOT via pkgutil.walk_packages.
`src/agentic_kg/coordinators/` has no __init__.py, so pkgutil treats it as a
namespace directory and never descends into it — walk_packages sees 28 of the
52 modules here and skips the entire coordinators tree, which is precisely the
code `adk web` loads. A test that cannot see the deliverable is worse than no
test, because it reports success.
"""
import importlib
from pathlib import Path

import agentic_kg

PACKAGE_ROOT = Path(agentic_kg.__file__).parent
SRC_ROOT = PACKAGE_ROOT.parent


def _module_names():
    """Every importable module name under the package, from the filesystem."""
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(SRC_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        yield ".".join(parts)


def test_module_discovery_covers_the_coordinators():
    """Guard the guard: if discovery silently stops finding coordinators,
    the import test above would pass while checking nothing that matters."""
    names = list(_module_names())
    assert any(name.startswith("agentic_kg.coordinators.") for name in names)
    assert len(names) > 40, f"only discovered {len(names)} modules"


def test_every_module_imports():
    failures = []
    for name in _module_names():
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - we want to report every failure
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)
