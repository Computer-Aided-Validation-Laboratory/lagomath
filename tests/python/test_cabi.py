import json
import subprocess
import sys
from pathlib import Path
import pytest
from validate_cases import ROOT

@pytest.mark.parametrize('backend',['zig','zig_release','mojo','mojo_debug'])
def test_cabi(libraries,backend):
    proc=subprocess.run([sys.executable,ROOT/'tests/python/cabi_worker.py',libraries[backend]],cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    records=json.loads(proc.stdout)
    report=ROOT/'build'/f'coverage-{backend}-cabi.json'
    report.write_text(json.dumps(records,indent=2)+'\n')
    assert any(r['status']=='pass' for r in records)
