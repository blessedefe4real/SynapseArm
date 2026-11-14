"""Runs the browser demo's JavaScript checks (docs/test_core.js) when Node is available.

They compare the JS port of the kinematics, IK and path tools with values from the Python code.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_demo_core_matches_python():
    out = subprocess.run(["node", str(ROOT / "docs" / "test_core.js")], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "all checks passed" in out.stdout
