"""Build the API/case registry and reject unacknowledged canonical operation gaps."""
import argparse
import json
import re
from api_surface import method_catalog
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

# Explicit mapping is intentional: fixture operations need not invent public methods.
VECTOR={'dot':'dot','normL1':'normL1','normL2':'normL2','normL2Square':'normL2Square','maxAbs':'maxAbs','allFinite':'allFinite','axpy':'axpy'}
MATRIX={'matvec':'mulVec','matmul':'mulMat','outer':'outer','outerAdd':'outerAdd','frobeniusNorm':'frobeniusNorm','maxAbs':'maxAbs','allFinite':'allFinite'}
SOLVERS={'gauss':'solveGauss','lu':'solveLU','cholesky':'solveCholesky','luReuse':'solveLUFactored','choleskyReuse':'solveCholeskyFactored'}

def catalog():
    cases=json.loads((ROOT/'tests/cases/all.json').read_text())['cases']
    benches=json.loads((ROOT/'benchmarks/cases/manifest.json').read_text())['cases']
    entries=[]
    for suite,module,operations,reps in [('vecops','vecops',VECTOR,['Vec','VecView']),('matops','matops',MATRIX,['Mat','MatView']),('linsolve','linsolve',SOLVERS,['Mat','MatView'])]:
      for op,prefix in operations.items():
       for rep in reps:
        name=prefix+rep
        zig=(ROOT/f'src/{module}.zig').read_text()
        mojo=(ROOT/f'src/mojo/{module}.mojo').read_text()
        ids=[c['id'] for c in cases if c['suite']==suite and c['operation']==op and c['representation']==rep]
        benchids=[c['id'] for c in benches if c['operation']==op and c['representation']==rep]
        entries.append(dict(name=name,module=module,representation=rep,zig=name in zig,mojo=name in mojo,dtypes=['f32','f64'],shared_case_ids=ids,benchmark_ids=benchids,cabi=op in ['dot','axpy','matvec','matmul','gauss','cholesky']))
    for rep in ['Mat','MatView']:
      for prefix,op in [('factorLU','luReuse'),('factorCholesky','choleskyReuse')]:
        name=prefix+rep
        entries.append(dict(name=name,module='linsolve',representation=rep,zig=name in (ROOT/'src/linsolve.zig').read_text(),mojo=name in (ROOT/'src/mojo/linsolve.mojo').read_text(),dtypes=['f32','f64'],shared_case_ids=[c['id'] for c in cases if c['suite']=='linsolve' and c['operation']==op and c['representation']==rep],benchmark_ids=[c['id'] for c in benches if c['operation']==op and c['representation']==rep],cabi=False))
    def entry(name,module,rep,operations,suite=None,kind='numerical'):
        ids=[c['id'] for c in cases if c['representation']==rep and c['operation'] in operations and (suite is None or c['suite']==suite)]
        entries.append(dict(name=name,module=module,representation=rep,kind=kind,zig=name in (ROOT/f'src/{module}.zig').read_text(),mojo=name in (ROOT/f'src/mojo/{module}.mojo').read_text(),dtypes=['f32','f64'],shared_case_ids=ids,benchmark_ids=[c['id'] for c in benches if c['representation']==rep and c['operation'] in operations],cabi=False))
    for rep in ['Vec','VecView']:
        entry('norm'+rep,'vecops',rep,['normL2','legacyNorms'])
        entry('squaredNorm'+rep,'vecops',rep,['normL2Square','legacyNorms'])
        entry('applyPermutation'+rep,'rowops',rep,['permutation'])
    for rep in ['Mat','MatView']:
        for prefix,op in [('swapRows','swapRows'),('scaleRow','scaleRow'),('addScaledRow','addScaledRow')]:
            entry(prefix+rep,'rowops',rep,[op+'Method'] if rep=='Mat' else [op])
        for prefix,op in SOLVERS.items():
            entry(op+rep+'Checked','linsolve',rep,[prefix]+(['luMultiple'] if prefix=='luReuse' else ['choleskyMultiple'] if prefix=='choleskyReuse' else []))
        for prefix,op in [('factorLU','luReuse'),('factorCholesky','choleskyReuse')]:
            entry(prefix+rep+'Checked','linsolve',rep,[op])
    entry("matchArrayDims","ndarraytype","NDArray",["matchDims"])
    # Full source inventories make the retained compatibility surface reviewable.
    inventory={}
    for language,paths in [('zig',sorted((ROOT/'src').glob('*.zig'))),('mojo',sorted((ROOT/'src/mojo').glob('*.mojo')))]:
        inventory[language]={str(p.relative_to(ROOT)):re.findall(r'(?:pub fn|\bdef|\bstruct)\s+(\w+)',p.read_text()) for p in paths if p.stem not in ['benchmark','case_runner','extended_cases','cabi','contracts']}
    return dict(schema_version=2,methods=method_catalog(ROOT,cases,benches),canonical_operations=entries,source_inventory=inventory,compatibility_policy='Historical Zig aliases and convenience namespaces retained; canonical parity is audited independently.',coverage_policy='Every canonical operation must exist in both languages and have shared numerical cases; no absent backend counts as passing.',numeric_domains={'f32':'canonical','f64':'canonical','integers':'shared i32 constructor/access/arithmetic subset; floating kernels require f32/f64'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    result=catalog()
    gaps=[e['name'] for e in result['canonical_operations']+result['methods'] if not e['zig'] or not e['mojo'] or (e.get('kind')!='mechanical' and not e['shared_case_ids'])]
    if gaps:raise SystemExit('API gaps: '+', '.join(gaps))
    path=ROOT/'docs/api_manifest.json';text=json.dumps(result,indent=2)+'\n'
    if args.check:
        if path.read_text()!=text:raise SystemExit('API inventory changed: run tools/audit_api.py and review it')
    else:path.write_text(text)
    print(f"{len(result['canonical_operations'])} canonical operations and {len(result['methods'])} methods verified")

if __name__=='__main__':main()
