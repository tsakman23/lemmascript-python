"""Every golden construct fixture also verifies, so each M1 construct has a
verified use. (Examples are verified by `lsc check` over LemmaScript-files.txt.)"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import has_dafny

GOLDEN = sorted((Path(__file__).resolve().parent / "golden").glob("*.dfy.gen"))


@pytest.mark.dafny
@pytest.mark.skipif(not has_dafny(), reason="dafny not on PATH")
@pytest.mark.parametrize("gen", GOLDEN, ids=lambda p: p.name.removesuffix(".dfy.gen"))
def test_golden_fixture_verifies(gen: Path, tmp_path: Path) -> None:
    dfy = tmp_path / gen.name.removesuffix(".gen")
    shutil.copy(gen, dfy)
    # As lsc does: Dafny's standard library is needed when the text uses `Std.`.
    std = ["--standard-libraries"] if "Std." in gen.read_text() else []
    r = subprocess.run(["dafny", "verify", *std, str(dfy)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
