import json
import subprocess
import numpy as np
import pytest
from validate_cases import ROOT, load_cases, validate

def decoded(values):
    return [float(v["special"]) if isinstance(v,dict) else v for v in values]

def verify(records):
    cases=load_cases();index={r['id']:r for r in records}
    assert len(index)==len(records),'Duplicate backend results'
    assert set(index)=={c['id'] for c in cases},'Missing or unexpected case IDs'
    for c in cases:
        r=index[c['id']]
        assert r['representation']==c['representation'],c['id']
        if 'expected_error' in c:assert r['status']=='error' and r['error']==c['expected_error'],(c['id'],r)
        else:
            assert r['status']=='ok',(c['id'],r)
            assert len(r['result'])==len(c['expected']),c['id']
            np.testing.assert_allclose(decoded(r['result']),decoded(c['expected']),equal_nan=True,atol=c['tolerance']['atol'],rtol=c['tolerance']['rtol'],err_msg=c['id'])
            if 'residual_tolerance' in c:
                a=np.asarray(decoded(c['inputs']['a']),dtype=float).reshape(c['inputs']['shape'])
                b=np.asarray(decoded(c['inputs']['b']),dtype=float)
                x=np.asarray(decoded(r['result']),dtype=float)
                residual=np.linalg.norm(a@x-b,np.inf)
                scale=np.linalg.norm(a,np.inf)*np.linalg.norm(x,np.inf)+np.linalg.norm(b,np.inf)
                assert residual<=c['residual_tolerance']*scale,c['id']


def test_manifest():assert validate()>0

@pytest.mark.parametrize('mode',['Debug','ReleaseFast'])
def test_zig(mode):
    p=subprocess.run(['zig','build','cases',f'-Doptimize={mode}','--','--jsonl'],cwd=ROOT,capture_output=True,text=True,check=True)
    records=[json.loads(s) for s in p.stderr.splitlines() if s.startswith('{')]
    verify(records)
    (ROOT/'build').mkdir(exist_ok=True)
    (ROOT/'build'/f'coverage-zig-native-{mode}.json').write_text(json.dumps(records))

@pytest.mark.parametrize('artifact',['mojo-cases','mojo-cases-debug'])
def test_mojo(libraries,artifact):
    p=subprocess.run([ROOT/'build'/artifact,'--jsonl'],cwd=ROOT,capture_output=True,text=True,check=True)
    records=[json.loads(s) for s in p.stdout.splitlines()]
    verify(records)
    (ROOT/'build'/f'coverage-mojo-native-{artifact}.json').write_text(json.dumps(records))

def test_reference_regeneration(tmp_path):
    import generate_regression as generator
    generator.OUT=tmp_path
    generator.generate()
    for path in tmp_path.glob('*.json'):
        assert path.read_bytes()==(ROOT/'tests/cases'/path.name).read_bytes(),path.name
