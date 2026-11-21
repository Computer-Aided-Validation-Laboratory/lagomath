import subprocess
import sys
import pytest
from validate_cases import ROOT

@pytest.fixture(scope='session')
def libraries():
    subprocess.run(['zig','build','cabi','-Doptimize=Debug'],cwd=ROOT,check=True)
    subprocess.run(['zig','build','cabi','-Doptimize=ReleaseFast','--prefix','build/cabi-release'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,ROOT/'tools/build_mojo.py'],cwd=ROOT,check=True,capture_output=True)
    return {'zig':ROOT/'zig-out/lib/liblagomath_zig.so','zig_release':ROOT/'build/cabi-release/lib/liblagomath_zig.so','mojo':ROOT/'build/liblagomath_mojo.so','mojo_debug':ROOT/'build/liblagomath_mojo_debug.so'}

