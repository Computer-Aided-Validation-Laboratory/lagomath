"""Combine validated native and C ABI records without hiding N/A cases."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def report():
    cases=json.loads((ROOT/'tests/cases/all.json').read_text())['cases']
    names=['zig-native-Debug','zig-native-ReleaseFast','mojo-native-mojo-cases','mojo-native-mojo-cases-debug','zig-cabi','zig_release-cabi','mojo-cabi','mojo_debug-cabi']
    records={}
    for name in names:
        path=ROOT/'build'/f'coverage-{name}.json'
        entries=json.loads(path.read_text())
        index={r['id']:r for r in entries}
        if len(index)!=len(entries) or set(index)!={c['id'] for c in cases}:raise RuntimeError(f'Incomplete coverage: {name}')
        records[name]=index
    matrix=[]
    for case in cases:
        row={'id':case['id']}
        for name,index in records.items():
            record=index[case['id']]
            if record['status']=='N/A':
                assert record.get('reason'),(name,case['id'])
                row[name]={'status':'N/A','reason':record['reason']}
            else:row[name]={'status':'pass'}
        matrix.append(row)
    (ROOT/'build/coverage-matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    return len(matrix)

if __name__=='__main__':print(f'{report()} cases across eight backend/build configurations')
