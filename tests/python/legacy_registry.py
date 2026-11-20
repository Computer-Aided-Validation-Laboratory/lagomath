"""Map the original named unit checks onto their shared execution cases."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
# Function names in old test titles map to actual shared operations, never to a
# passing placeholder. Multiple operations retain composite-test coverage.
RULES={
'getRowVec':['row'],'getColVec':['column'],'getSubMat':['submatrix'],
'insertRowVec':['insertRow'],'insertColVec':['insertColumn'],
'insertSubMat':['insertSubmatrix'],'insertMat22':['insertMat22'],
'multiplies':['matmul'],'multiply':['mulInPlace'],'divide':['divInPlace'],'mulVec3':['mulVec3'],'mulVec':['matvec'],'mulMatRect':['matmul'],
'mulMat':['matmul'],'mulSquare':['matmul'],'mulElemWise':['mulOut'],
'divElemWise':['divOut'],'mulScal':['scale'],'addInPlace':['add'],
'subInPlace':['sub'],'mulInPlace':['mulInPlace'],'divInPlace':['divInPlace'],
'apply':['applySqrt','applyAtan','applyExp'],'.max':['max','extremaIndices'],
'.min':['min','extremaIndices'],'sum':['sum'],'mean':['mean'],
'axpy':['axpy'],'cross':['cross'],'norm':['normL1','normL2','normL2Square','legacyNorms'],
'length':['normL2'],'dot':['dot'],'add':['add'],'sub':['sub'],
'.mul':['mul'],'div':['div'],'trace':['trace'],'transpose':['transpose','transposeSquare'],
'.det':['det'],'.inv':['inv','inverseIdentity'],'adjugate':['adj'],
'getFlatIdx':['getFlatIdx'],'calcFlatStride':['calcFlatStride'],
'getSlice':['getSlice','row'],'getPlaneSlice':['getPlaneSlice','planeTail'],
'fixedPrefixView':['fixedPrefixView','emptyPrefix'],'offset':['offset2','offset3','offset4'],
'matchArrayDims':['matchDims'],'shape mismatch':['matchDims'],
'flat allocation':['initFlat','dupe','planeTail'],'NDArrayOps':['addOut','subOut','mulOut','divOut','extractMat','emptyPrefix'],
'swapRows':['swapRows'],'scaleRow':['scaleRow'],'addScaledRow':['addScaledRow'],
'Permutation':['permutation'],'outer':['outer','outerAdd'],
'frobeniusNorm':['frobeniusNorm'],'maxAbs':['maxAbs'],'allFinite':['allFinite'],
'factorisation reconstruction':['luReconstruction','choleskyReconstruction'],
'factorisation reuse':['luMultiple','choleskyMultiple'],
'cross-solver':['gauss','lu','cholesky'],'DIC':['gauss','lu','cholesky'],
'gauss.':['gauss'],'lu.':['lu','luReuse'],'cholesky.':['cholesky','choleskyReuse'],
'constructors':['zeros','ones','fill','identity','diag','initRows','initVec2','initVec3','coordinates'],
'identity':['identity','transpose'],'flat helpers':['flatAccess'],
'column extrema':['columnExtrema'],'field access':['valueCopy','flatAccess'],
'row slice':['rowMutation','row'],'initSlice':['initRows','valueCopy'],
'near-singular':['invThreshold'],'reductions':['extremaIndices','normL1','normL2','normL2Square','rangeLen'],
'partial apply':['partialApply'],'checked slice':['addChecked','subChecked','mulChecked','divChecked','mulScalChecked'],
}


def generate(cases,output):
    source=json.loads((ROOT/'tests/cases/historical_inputs.json').read_text())['tests']
    rows=[]
    for test in source:
        operations=set()
        for text,ops in RULES.items():
            if text.lower() in test['name'].lower():operations.update(ops)
        # Composite titles need the extra contracts they describe.
        if test['file'].endswith('vectype.zig') and 'minimum' in test['name']:operations.add('componentExtrema')
        if 'VecSliceOps' in test['name']:operations.add('componentExtrema')
        if 'integer' in test['name']:operations.add('integerArithmetic')
        if 'cleanup' in test['name']:operations.add('dupe')
        if 'allocation' in test['name']:operations.update(['fill','initFlat'])
        relevant=[c for c in cases if c['operation'] in operations]
        if not relevant:raise RuntimeError('Unmapped historical test: '+test['name'])
        # Prefer extracted original data; retain every operation in composites.
        selected=[]
        for operation in sorted(operations):
            matches=[c for c in relevant if c['operation']==operation]
            if not matches:continue
            for dtype in ['f32','f64','i32']:
                typed=[c for c in matches if c['dtype']==dtype]
                extracted=[c for c in typed if test['id'] in c['id']]
                selected+=(extracted or typed)[:2]
        rows.append(dict(id=test['id'],file=test['file'],name=test['name'],original_assertion_count=test['assertions'],case_ids=sorted({c['id'] for c in selected})))
    selected_ids={i for row in rows for i in row['case_ids']}
    (output/'legacy_registry.json').write_text(json.dumps(dict(schema_version=1,tests=rows),indent=2)+'\n')
    (output/'legacy_cases.json').write_text(json.dumps(dict(cases=[c for c in cases if c['id'] in selected_ids]),separators=(',',':'))+'\n')
