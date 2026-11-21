import hashlib
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

def assert_reference_tree(actual, frozen, *, dtype=None, numerical=False, path='root'):
    if isinstance(frozen,dict):
        assert isinstance(actual,dict) and actual.keys()==frozen.keys(),path
        if frozen.get('source')=='hand':
            assert actual==frozen,path
            return
        dtype=frozen.get('dtype',dtype)
        for key in frozen:
            assert_reference_tree(actual[key],frozen[key],dtype=dtype,
                                  numerical=key in ('inputs','expected') or numerical,
                                  path=f'{path}.{key}')
    elif isinstance(frozen,list):
        assert isinstance(actual,list) and len(actual)==len(frozen),path
        if numerical and frozen and all(isinstance(v,(int,float)) for v in frozen) and any(isinstance(v,float) for v in frozen):
            eps=np.finfo(np.float32 if dtype=='f32' else np.float64).eps
            # Use array scale: matrix products can cancel in individual entries.
            scale=max(abs(v) for v in frozen)
            np.testing.assert_allclose(actual,frozen,rtol=32*eps,atol=32*eps*scale,err_msg=path)
            return
        for i,(a,b) in enumerate(zip(actual,frozen)):
            assert_reference_tree(a,b,dtype=dtype,numerical=numerical,path=f'{path}[{i}]')
    elif numerical and isinstance(frozen,float):
        # BLAS/LAPACK dispatch varies by CPU even with identical locked wheels.
        # Allow only reference roundoff, independently of kernel tolerances.
        eps=np.finfo(np.float32 if dtype=='f32' else np.float64).eps
        assert isinstance(actual,(int,float)),path
        assert abs(actual-frozen)<=32*eps*max(1,abs(frozen)),path
    else:
        assert actual==frozen,path


def assert_reference_manifest(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    for group in ('files','derived_files','provenance_files'):
        for name,digest in manifest[group].items():
            source=ROOT/'tests/cases'/name if group=='provenance_files' else directory/name
            assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
    # Each tree has its own valid hashes; numerical roundoff changes those hashes.
    return {key:value for key,value in manifest.items() if key not in ('files','derived_files')}


def test_reference_regeneration(tmp_path,monkeypatch):
    import generate_regression as generator
    monkeypatch.setattr(generator,'OUT',tmp_path)
    generator.generate()
    frozen=ROOT/'tests/cases'
    # Historical provenance is an input archive, not regenerated output.
    assert {p.name for p in tmp_path.glob('*.json')}=={p.name for p in frozen.glob('*.json') if p.name not in ('historical_inputs.json','schema.json')}
    assert assert_reference_manifest(tmp_path)==assert_reference_manifest(frozen)
    generated_manifest=json.loads((tmp_path/'manifest.json').read_text())
    frozen_manifest=json.loads((frozen/'manifest.json').read_text())
    for group in ('files','derived_files'):
        assert generated_manifest[group].keys()==frozen_manifest[group].keys()
    for path in tmp_path.glob('*.json'):
        if path.name!='manifest.json':
            assert_reference_tree(json.loads(path.read_text()),json.loads((frozen/path.name).read_text()),path=path.name)


@pytest.mark.parametrize('field',['inputs','expected','tolerance','operation'])
def test_reference_comparison_rejects_changes(field):
    frozen=dict(dtype='f64',inputs=dict(a=[1.0]),expected=[2.0],
                tolerance=dict(atol=2e-12,rtol=2e-11),operation='dot')
    actual=json.loads(json.dumps(frozen))
    if field=='inputs':actual['inputs']['a'][0]+=1e-6
    elif field=='expected':actual['expected'][0]+=1e-6
    elif field=='tolerance':actual['tolerance']['atol']*=2
    else:actual['operation']='sum'
    with pytest.raises(AssertionError):assert_reference_tree(actual,frozen)


def test_reference_comparison_accepts_roundoff():
    for dtype in ('f32','f64'):
        frozen=dict(dtype=dtype,inputs=dict(a=[1.0]),expected=[2.0])
        actual=json.loads(json.dumps(frozen))
        eps=float(np.finfo(np.float32 if dtype=='f32' else np.float64).eps)
        actual['inputs']['a'][0]+=eps
        actual['expected'][0]-=eps
        assert_reference_tree(actual,frozen)
