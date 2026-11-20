import os
from pathlib import Path
import shutil
import subprocess
import sys
from validate_cases import ROOT


def test_relocated_mojo_runtime(libraries,tmp_path):
    shutil.copy2(libraries['mojo'],tmp_path/'liblagomath_mojo.so')
    shutil.copytree(ROOT/'build/.libs',tmp_path/'.libs')
    # Standard library only, isolated imports and no Mojo environment variables.
    code='import ctypes; lib=ctypes.CDLL('+repr(str(tmp_path/'liblagomath_mojo.so'))+'); assert lib.lm_abi_version()==1'
    subprocess.run([sys.executable,'-I','-S','-c',code],env={'PATH':'/usr/bin:/bin','HOME':str(tmp_path)},check=True)


def test_api_manifest():
    subprocess.run([sys.executable,ROOT/'tools/audit_api.py','--check'],cwd=ROOT,check=True)


def test_mojo_consumer():
    subprocess.run([ROOT/'.venv/bin/mojo','-I',ROOT/'src/mojo',ROOT/'integration/mojo-consumer/main.mojo'],cwd=ROOT,check=True,capture_output=True)
