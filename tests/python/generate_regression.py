"""Freeze shared numerical inputs and independent references. No backend computes truth."""
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy
import scipy.linalg
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tests/cases'

def generate():
    suites={}
    case_ids=set()
    def add(suite,operation,dtype,inputs,expected=None,error=None,representation='View',label='basic',source='numpy'):
        case=dict(id=f'{suite}.{operation}.{representation}.{label}.{dtype}',suite=suite,category='property' if source=='mathematical_identity' else 'regression' if source!='hand' else 'analytical',operation=operation,dtype=dtype,representation=representation,inputs=inputs,source=source)
        if error: case['expected_error']=error
        else: case.update(expected=[({"special":"nan" if np.isnan(v) else "-inf" if v<0 else "inf"} if not np.isfinite(v) else v) for v in np.asarray(expected).reshape(-1).tolist()],tolerance=dict(atol=2e-5 if dtype=='f32' else 2e-12,rtol=2e-4 if dtype=='f32' else 2e-11))
        if source=='hand' and not error:case['tolerance']=dict(atol=0,rtol=0)
        if suite=='types':
            case['inputs'].update(mutate_index=1,mutate_value=7)
            if representation=='VecView':case['inputs']['view_start']=1
        if suite=='rowops':case['inputs'].update(row_a=0,row_b=case['inputs']['shape'][0]-1)
        if case["id"] in case_ids:raise ValueError("Duplicate case ID: "+case["id"])
        case_ids.add(case["id"])
        suites.setdefault(suite,[]).append(case)
    for dtype in ['f32','f64']:
      dt=np.float32 if dtype=='f32' else np.float64
      for n in [1,2,3,4,6,8,12,16,31,257,1003,4096]:
        a=np.asarray([((i%13)-6)/8 for i in range(n)],dtype=dt);b=np.asarray([((i%7)+1)/4 for i in range(n)],dtype=dt)
        inputs=dict(a=a.tolist(),b=b.tolist(),shape=[n],alpha=-0.5)
        ops=dict(dot=np.dot(a,b),normL1=np.sum(np.abs(a)),normL2=np.linalg.norm(a),normL2Square=np.dot(a,a),maxAbs=np.max(np.abs(a)),allFinite=1,sum=np.sum(a),mean=np.mean(a),min=np.min(a),max=np.max(a),add=a+b,sub=a-b,mul=a*b,div=a/b,axpy=-0.5*a+b)
        for rep in ['Vec','VecView','slice'] if n<=16 else ['VecView','slice']:
          for op,expected in ops.items():
            if rep=='Vec' and op in ['mul','div']:continue
            if rep=='slice' and op in ['maxAbs','allFinite','axpy']:continue
            add('vecops',op,dtype,inputs,expected,representation=rep,label=str(n))
      for n in [0,1]:
        for op in ['dot','normL1','normL2','normL2Square']:
          add('vecops',op,dtype,dict(a=[0.]*n,b=[0.]*n,shape=[n]),[0],representation='VecView',label=f'zero{n}',source='hand')
      for rows,inner,cols in [(1,1,1),(2,2,2),(3,3,3),(4,4,4),(6,6,6),(8,8,8),(12,12,12),(2,3,4),(6,12,8),(12,6,3)]:
        a=(np.arange(rows*inner,dtype=dt).reshape(rows,inner)%11-5)/8;b=(np.arange(inner*cols,dtype=dt).reshape(inner,cols)%7-3)/4;x=np.arange(inner,dtype=dt)/4
        inp=dict(a=a.reshape(-1).tolist(),b=b.reshape(-1).tolist(),x=x.tolist(),shape=[rows,inner,cols])
        for rep in ['MatView','Mat']:
          for op,expected in dict(matmul=a@b,matvec=a@x,transpose=a.T,trace=np.trace(a),frobeniusNorm=np.linalg.norm(a),maxAbs=np.max(np.abs(a)),allFinite=1).items():
            add('matops',op,dtype,inp,expected,representation=rep,label=f'{rows}x{inner}x{cols}')
      for n in [1,2,3,4,6,8,12]:
        rng=np.random.default_rng(314159+n);j=rng.normal(size=(n+3,n));a=np.asarray(j.T@j+np.eye(n),dtype=dt);b=np.arange(1,n+1,dtype=dt)
        for scale in ([1.,1e-25,1e25] if dtype=='f32' else [1.,1e-250,1e250]):
          aa=np.asarray(a*scale,dtype=dt);bb=np.asarray(b*scale,dtype=dt)
          # Reference solves unscaled represented matrix to avoid LAPACK scaling artefacts.
          x=np.linalg.solve(aa.astype(np.float64)/scale,bb.astype(np.float64)/scale)
          for op in ['gauss','lu','cholesky','luReuse','choleskyReuse']:
            for rep in ['Mat','MatView']:
              add('linsolve',op,dtype,dict(a=aa.reshape(-1).tolist(),b=bb.tolist(),shape=[n,n]),x,representation=rep,label=f'{n}.scale{scale}')
        for op,error,aa in [('gauss','SingularMatrix',np.zeros((n,n))),('lu','SingularMatrix',np.zeros((n,n))),('cholesky','NotPositiveDefinite',-np.eye(n))]:
          add('linsolve',op,dtype,dict(a=aa.reshape(-1).tolist(),b=b.tolist(),shape=[n,n]),error=error,representation='MatView',label=f'{n}.failure',source='hand')
      for shape in [[1],[3],[17],[257],[2,3],[3,5],[2,3,4],[3,2,5],[2,3,4,5],[2,2,2,2,2]]:
        index=[s-1 for s in shape];flat=int(np.ravel_multi_index(tuple(index),tuple(shape)))
        add('ndarray','getFlatIdx',dtype,dict(shape=shape,index=index,a=list(range(int(np.prod(shape))))),[flat],representation='NDArray',label='x'.join(map(str,shape)),source='hand')
    for dtype in ['f32','f64']:
      for rows,cols in [(1,1),(2,3),(3,2),(6,6)]:
        a=np.arange(rows*cols,dtype=float).reshape(rows,cols)/4
        for op in ['swapRows','scaleRow','addScaledRow']:
          expected=a.copy()
          if op=='swapRows':expected[[0,rows-1]]=expected[[rows-1,0]]
          elif op=='scaleRow':expected[0]*=-0.5
          else:expected[rows-1]+=expected[0]*-0.5
          add('rowops',op,dtype,dict(a=a.reshape(-1).tolist(),shape=[rows,cols],alpha=-0.5),expected,representation='MatView',label=f'{rows}x{cols}')
      for n in [1,3,6,12]:
        a=np.arange(n,dtype=float)/4;b=np.arange(n,dtype=float)/2-1
        for op in ['outer','outerAdd']:
          initial=np.full((n,n),2.)
          expected=np.outer(a,b) if op=='outer' else initial-0.5*np.outer(a,b)
          for rep in ['Mat','MatView']:
            add('matops',op,dtype,dict(a=initial.reshape(-1).tolist(),b=b.tolist(),x=a.tolist(),shape=[n,n,n],alpha=-0.5),expected,representation=rep,label=str(n))
      for op,error in [('gauss','SingularMatrix'),('lu','SingularMatrix'),('cholesky','NotPositiveDefinite')]:
        add('linsolve',op,dtype,dict(a=[0,1,1,0] if op=='cholesky' else [1,2,2,4],b=[1,1],shape=[2,2]),error=error,representation='MatView',label='denseFailure',source='hand')
        add('linsolve',op,dtype,dict(a=[1,0,0,1,0,0],b=[1,1],shape=[2,3]),error='ShapeMismatch',representation='MatView',label='badShape',source='hand')
        add('linsolve',op,dtype,dict(a=[{'special':'nan'},0,0,1],b=[1,1],shape=[2,2]),error='NonFiniteInput',representation='MatView',label='nan',source='hand')
        add('linsolve',op,dtype,dict(a=[1,0,0,1],b=[{'special':'inf'},1],shape=[2,2]),error='NonFiniteInput',representation='MatView',label='infRhs',source='hand')
      for shape in [[3],[2,3],[2,3,4],[2,3,4,5],[2,2,2,2,2]]:
        strides=[int(np.prod(shape[i+1:])) for i in range(len(shape))]
        inp=dict(a=list(range(int(np.prod(shape)))),shape=shape,index=[n-1 for n in shape])
        label='x'.join(map(str,shape))
        for op,expected in [('calcFlatStride',strides),('planeBase',[strides[0]*(shape[0]-1)]),('getPlaneSlice',list(range(strides[0]*(shape[0]-1),int(np.prod(shape)))) )]:
          add('ndarray',op,dtype,inp,expected,representation='NDArray',label=label,source='hand')
        if len(shape)>=2:
          add('ndarray','subBase2',dtype,inp,[sum((shape[i]-1)*strides[i] for i in range(2))],representation='NDArray',label=label,source='hand')
          start=(shape[0]-1)*strides[0]
          add('ndarray','fixedPrefixView',dtype,inp,list(range(start,start+strides[0])),representation='NDArray',label=label,source='hand')
        if len(shape) in [2,3,4]:
          add('ndarray','offset'+str(len(shape)),dtype,inp,[int(np.prod(shape))-1],representation='NDArray',label=label,source='hand')
    for dtype in ['f32','f64']:
      for n in [2,3,4]:
        rng=np.random.default_rng(2718+n)
        a=rng.normal(size=(n,n))+np.eye(n)*3
        a=a.astype(np.float32 if dtype=='f32' else np.float64)
        for op,expected in [('det',np.linalg.det(a)),('inv',np.linalg.inv(a))]:
          add('matops',op,dtype,dict(a=a.reshape(-1).tolist(),b=[],x=[],shape=[n,n,n]),expected,representation='Mat',label=f'{n}.dense')
        add('matops','inv',dtype,dict(a=[0]*(n*n),b=[],x=[],shape=[n,n,n]),error='SingularMatrix',representation='Mat',label=f'{n}.singular',source='hand')
    for dtype in ['f32','f64']:
      for rep in ['Vec','VecView']:
        add('types','valueCopy' if rep=='Vec' else 'viewMutation',dtype,dict(a=[1,2,3] if rep=='Vec' else [99,1,2,3,99],b=[1,2,3],shape=[3]),[1,2,3,1,7,3] if rep=='Vec' else [99,1,7,3,99],representation=rep,label='ownership',source='hand')
      add('types','valueCopy',dtype,dict(a=[1,2,3,4],b=[1,2,3,4],x=[],shape=[2,2,2]),[1,2,3,4,1,7,3,4],representation='Mat',label='ownership',source='hand')
      for n in [2,3,4]:
        a=np.eye(n)*3+np.ones((n,n))/8
        add('matops','inverseIdentity',dtype,dict(a=a.reshape(-1).tolist(),b=[],x=[],shape=[n,n,n]),np.eye(n),representation='Mat',label=str(n),source='mathematical_identity')
        suites['matops'][-1]['category']='property'
    for dtype in ['f32','f64']:
      a=np.array([[0.,2.,1.],[3.,-1.,2.],[4.,-1.,5.]])
      b=np.array([5.,7.,15.])
      for op in ['gauss','lu']:
        for rep in ['Mat','MatView']:
          add('linsolve',op,dtype,dict(a=a.reshape(-1).tolist(),b=b.tolist(),shape=[3,3]),np.linalg.solve(a,b),representation=rep,label='pivotRequired')
    for dtype in ['f32','f64']:
      for rep in ['Vec','VecView','slice']:
        add('vecops','dot',dtype,dict(a=[{'special':'nan'},1,2],b=[1,1,1],shape=[3]),[np.nan],representation=rep,label='nanOutput',source='hand')
        add('vecops','dot',dtype,dict(a=[{'special':'-inf'},1,2],b=[1,1,1],shape=[3]),[-np.inf],representation=rep,label='negativeInfOutput',source='hand')
        if rep!='Vec':
          add('vecops','div',dtype,dict(a=[0,1,-1],b=[0,0,0],shape=[3]),[np.nan,np.inf,-np.inf],representation=rep,label='zeroDivisor',source='hand')
    for dtype in ['f32','f64']:
      dt=np.float32 if dtype=='f32' else np.float64
      eps=np.finfo(dt).eps
      for n in [2,3,4,6,8,12]:
        families={'identity':np.eye(n),'diagonal':np.diag(np.arange(1,n+1)),'triangular':np.triu(np.ones((n,n)))+np.eye(n),'permutation':np.eye(n)[::-1], 'nearSingular':np.diag([1.]*(n-1)+[eps/8])}
        if n in [2,3,6,12]:
            rng=np.random.default_rng(9876+n)
            families['denseSigned']=rng.normal(size=(n,n))+np.eye(n)*n
            families['conditionedDiagonal']=np.diag(np.geomspace(1,1e-4 if dtype=='f32' else 1e-10,n))
        families['zeroRhs']=np.eye(n)
        if n in [2,3,6,12]:
            for condition in ([1e2,1e4] if dtype=='f32' else [1e2,1e4,1e6,1e8,1e12]):
                families['condition'+str(condition)]=np.diag(np.geomspace(1,1/condition,n))
        for label,raw in families.items():
            a=raw.astype(dt);b=(np.zeros(n,dtype=dt) if label=='zeroRhs' else (a@np.arange(1,n+1,dtype=dt)).astype(dt))
            for op in ['gauss','lu','cholesky']:
                if op=='cholesky' and (not np.allclose(a,a.T) or np.linalg.eigvalsh(a).min()<=0):continue
                for rep in ['Mat','MatView']:
                    error='SingularMatrix' if label=='nearSingular' and op!='cholesky' else None
                    expected=None if error else (np.linalg.solve(a.astype(float),b.astype(float)) if np.array_equal(a,np.diag(np.diag(a))) else scipy.linalg.solve(a.astype(float),b.astype(float)))
                    add('linsolve',op,dtype,dict(a=a.ravel().tolist(),b=b.tolist(),shape=[n,n]),expected,error=error,representation=rep,label=f'{n}.{label}',source='scipy')
                    if not error:
                        suites['linsolve'][-1]['residual_tolerance']=2e-5 if dtype=='f32' else 2e-12
    for dtype in ['f32','f64']:
      for tag in ['nan','inf','-inf']:
        for rep in ['Vec','VecView']:
          add('vecops','allFinite',dtype,dict(a=[{'special':tag},1,2],b=[0,0,0],shape=[3]),[0],representation=rep,label=tag,source='hand')
        for rep in ['Mat','MatView']:
          add('matops','allFinite',dtype,dict(a=[1,0,0,{'special':tag}],b=[],x=[],shape=[2,2,2]),[0],representation=rep,label=tag,source='hand')
      for op in ['dot','normL1','normL2','normL2Square']:
        add('vecops',op,dtype,dict(a=[],b=[],shape=[0]),[0],representation='slice',label='empty',source='hand')
    from extended_cases import generate as extended
    extended(add)
    from historical_cases import generate as historical
    historical(add)
    for suite,cases in suites.items(): (OUT/f'{suite}.json').write_text(json.dumps(dict(schema_version=1,cases=cases),indent=2,allow_nan=False)+'\n')
    contracts=[dict(id='ndarray.init.'+label,suite='ndarray',category='contract',operation='init',dtype='f64',representation='NDArray',inputs=dict(a=[0],shape=shape),expected_error='AssertionFailed',source='hand',execution='native-child-process') for label,shape in [('zero_dimension',[2,0,3]),('rank_zero',[])]]
    for label,op,shape,index in [('index_bounds','get',[2,3],[2,0]),('wrong_offset_rank','offset3',[2,3],[0,0,0]),('view_length','matrixInit',[2,3],[])]:
        contracts.append(dict(id='contract.'+label,suite='ndarray' if op!='matrixInit' else 'types',category='contract',operation=op,dtype='f64',representation='NDArray' if op!='matrixInit' else 'MatView',inputs=dict(a=[0]*6,shape=shape,index=index),expected_error='AssertionFailed',source='hand',execution='native-child-process'))
    (OUT/'contracts.json').write_text(json.dumps(dict(schema_version=1,cases=contracts),indent=2)+'\n')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.json')) if p.stem in suites or p.stem=='contracts'}
    manifest=dict(schema_version=1,generator='generate_regression.py/v2',numpy=np.__version__,scipy=scipy.__version__,files=files,case_count=sum(map(len,suites.values())),contract_count=len(contracts),abi_output_guard=dict(prefix=2,suffix=2,value=123))
    all_cases=[c for cases in suites.values() for c in cases]
    (OUT/'all.json').write_text(json.dumps(dict(schema_version=1,cases=all_cases),separators=(',',':'))+'\n')
    from legacy_registry import generate as registry
    registry(all_cases,OUT)
    manifest['provenance_files']={'historical_inputs.json':hashlib.sha256((ROOT/'tests/cases/historical_inputs.json').read_bytes()).hexdigest()}
    manifest['derived_files']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['all.json','legacy_registry.json','legacy_cases.json']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{len(all_cases)} shared cases')

if __name__=='__main__':generate()
