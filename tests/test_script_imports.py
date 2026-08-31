"""The scripts package must resolve to THIS repo, not something on sys.path.

`scripts/` is imported by every diagnostic (`from scripts.diagnose_failure
import split_indices`). Without an __init__.py it is a namespace package, and a
namespace portion is discarded as soon as the path scan finds a regular
`scripts` package further down sys.path -- which some environments provide.
The failure is environment-dependent, so it passed locally and broke the whole
diagnostic suite on the training machine.

Deliberately torch-free so it runs anywhere.
"""

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_scripts_is_a_regular_package():
    assert (REPO / "scripts" / "__init__.py").exists(), (
        "scripts/__init__.py is required; without it `scripts` is a namespace "
        "package that any installed `scripts` package will shadow"
    )


def test_scripts_resolves_inside_this_repo():
    spec = importlib.util.find_spec("scripts")
    assert spec is not None and spec.origin is not None, "scripts did not resolve"
    assert pathlib.Path(spec.origin).resolve().is_relative_to(REPO), (
        f"`import scripts` resolved to {spec.origin}, outside the repo -- "
        "something on sys.path is shadowing it"
    )


def test_every_cross_script_import_target_exists():
    """Each `from scripts.X import Y` must name a module that is really there."""
    import re

    missing = []
    for path in sorted((REPO / "scripts").glob("*.py")):
        for module in re.findall(r"^from scripts\.(\w+) import", path.read_text(), re.M):
            if not (REPO / "scripts" / f"{module}.py").exists():
                missing.append(f"{path.name} -> scripts.{module}")
    assert not missing, f"imports naming absent modules: {missing}"
