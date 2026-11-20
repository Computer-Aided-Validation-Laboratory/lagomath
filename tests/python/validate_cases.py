import hashlib
import json
from pathlib import Path
import jsonschema
ROOT=Path(__file__).resolve().parents[2]

def reject_constant(value):raise AssertionError("Non-standard JSON number: "+value)

def load_json(path):return json.loads(path.read_text(),parse_constant=reject_constant)

def load_cases():return load_json(ROOT/'tests/cases/all.json')['cases']

def validate():
    jsonschema.validate(load_json(ROOT/'tests/cases/all.json'),json.loads((ROOT/'tests/cases/schema.json').read_text()))
    cases=load_cases();ids=[c['id'] for c in cases]
    assert len(ids)==len(set(ids)), 'Duplicate case IDs'
    manifest=json.loads((ROOT/'tests/cases/manifest.json').read_text())
    assert len(cases)==manifest['case_count']
    for name,digest in manifest['files'].items():assert hashlib.sha256((ROOT/'tests/cases'/name).read_bytes()).hexdigest()==digest,name
    reconstructed=[]
    for name in manifest['files']:
        if name!='contracts.json':reconstructed+=json.loads((ROOT/'tests/cases'/name).read_text())['cases']
    contracts=json.loads((ROOT/'tests/cases/contracts.json').read_text())['cases']
    assert len(contracts)==manifest['contract_count']
    assert {c['id']:c for c in reconstructed}=={c['id']:c for c in cases}
    for case in cases:
        assert ('expected' in case)!=('expected_error' in case)
        assert case['dtype'] in ['f32','f64','i32']
        assert case['inputs']['shape'] and all(n>=0 for n in case['inputs']['shape'])
    for group in ['provenance_files','derived_files']:
        for name,digest in manifest[group].items():
            assert hashlib.sha256((ROOT/'tests/cases'/name).read_bytes()).hexdigest()==digest,name
    archive=load_json(ROOT/'tests/cases/historical_inputs.json')['tests']
    registry=load_json(ROOT/'tests/cases/legacy_registry.json')['tests']
    assert len(registry)==len(archive)==140
    assert {c['id'] for c in registry}=={c['id'] for c in archive}
    bundle=load_json(ROOT/'tests/cases/legacy_cases.json')['cases']
    selected={i for row in registry for i in row['case_ids']}
    assert selected=={c['id'] for c in bundle}
    index={c['id']:c for c in cases}
    assert all(c==index[c['id']] for c in bundle)
    for row in registry:
        assert row['case_ids']
        assert f'verifyLegacy("{row["id"]}")' in (ROOT/row['file']).read_text()
    path=ROOT/'benchmarks/cases'
    jsonschema.validate(json.loads((path/'manifest.json').read_text()),json.loads((path/'schema.json').read_text()))
    return len(cases)

if __name__=='__main__': print(f'{validate()} shared cases validated')
