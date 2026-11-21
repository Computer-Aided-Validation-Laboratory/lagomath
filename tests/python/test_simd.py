"""SIMD reuses frozen independent oracles, plus native boundary/alias checks."""
import subprocess
import sys
from validate_cases import ROOT


def test_manual_simd_shared_cases():
    result=subprocess.run([sys.executable,ROOT/'tools/run_simd_cases.py'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_simd_function_names_are_explicit():
    import re
    paths=list((ROOT/'src').glob('*_simd.zig'))+list((ROOT/'src/mojo').glob('*_simd.mojo'))
    assert paths
    for path in paths:
        names=re.findall(r'\b(?:fn|def)\s+(\w+)',path.read_text())
        assert all(name.endswith('SIMD') for name in names),path
