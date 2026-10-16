"""Golden Dafny output: `lsc gen` on each construct fixture and each example must
match the committed `.dfy.gen` byte for byte.

Regenerate after an intended change with ``LSC_UPDATE_GOLDEN=1 pytest tests/test_golden.py``
and review the diff.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from conftest import REPO, run_lsc

HERE = Path(__file__).resolve().parent
CASES = sorted((HERE / "golden").glob("*.py")) + sorted((REPO / "python" / "examples").glob("*.py"))


@pytest.mark.lsc
@pytest.mark.parametrize("src", CASES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_golden_dfy_gen(src: Path, tmp_path: Path) -> None:
    work = tmp_path / src.name
    shutil.copy(src, work)
    r = run_lsc(["gen", "--backend=dafny", work.name], tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    got = (tmp_path / f"{src.stem}.dfy.gen").read_text()
    golden = src.with_suffix(".dfy.gen")
    if os.environ.get("LSC_UPDATE_GOLDEN") == "1":
        golden.write_text(got)
    assert golden.exists(), f"missing golden file {golden} (run with LSC_UPDATE_GOLDEN=1)"
    assert got == golden.read_text(), f"{golden.name} drifted; rerun with LSC_UPDATE_GOLDEN=1 and review the diff"


@pytest.mark.lsc
def test_extract_command_prints_the_envelope(tmp_path: Path) -> None:
    src = REPO / "python" / "examples" / "clamp.py"
    r = run_lsc(["extract", str(src)], tmp_path)
    assert r.returncode == 0, r.stderr
    import json
    env = json.loads(r.stdout)
    assert env["functions"][0]["name"] == "clamp"  # `extract` prints the Raw IR module, as for .ts


@pytest.mark.lsc
def test_unsupported_construct_fails_lsc_with_location(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("def f(x: float) -> int:\n    return 1\n")
    r = run_lsc(["gen", "--backend=dafny", "bad.py"], tmp_path)
    assert r.returncode != 0
    assert "bad.py:1:" in r.stderr and "'float' is not supported" in r.stderr
    assert not (tmp_path / "bad.dfy.gen").exists()
