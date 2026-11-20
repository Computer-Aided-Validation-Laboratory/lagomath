"""Re-exercise extracted historical inputs using independent numerical references.

The archive is immutable migration provenance; it is never read from Zig source
while regenerating. Named native smoke tests refer to the migration registry.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def generate(add):
    tests=json.loads((ROOT/'tests/cases/historical_inputs.json').read_text())['tests']
    from legacy_registry import operations_for_test
    emit=add
    for test in tests:
        operations=operations_for_test(test)
        seen=set()
        def add(suite,operation,dtype,inputs,expected=None,**kwargs):
            # Historical inputs only exercise the behaviour named by that test.
            # One original input per operation/representation/precision suffices;
            # RHS and scratch arrays must not create unrelated test families.
            key=(operation,kwargs.get('representation'),dtype)
            if operation not in operations or key in seen:return
            seen.add(key)
            emit(suite,operation,dtype,inputs,expected,**kwargs)
        for index,array in enumerate(test['arrays']):
            if array['role']!='input' or not array['values']:continue
            for dtype in ['f32','f64']:
                dt=np.float32 if dtype=='f32' else np.float64
                a=np.asarray(array['values'],dtype=dt)
                label=test['id']+'.'+str(index)
                if len(array['shape'])==1:
                    n=len(a);b=np.ones(n,dtype=dt)
                    for rep in (['Vec','VecView','slice'] if n in [1,2,3,4,6,8,12,16] else ['VecView','slice']):
                        for op,expected in [('dot',np.dot(a,b)),('normL1',np.sum(abs(a))),('normL2',np.linalg.norm(a)),('normL2Square',np.dot(a,a)),('min',np.min(a)),('max',np.max(a)),('sum',np.sum(a)),('mean',np.mean(a)),('add',a+b),('sub',a-b)]:
                            add('vecops',op,dtype,dict(a=a.tolist(),b=b.tolist(),shape=[n],alpha=2),expected,representation=rep,label=label,source='numpy')
                else:
                    r,c=array['shape'];a=a.reshape(r,c)
                    for rep in ['Mat','MatView'] if r==c and r in [2,3,4] else ['MatView']:
                        inp=dict(a=a.ravel().tolist(),b=np.ones((c,c)).ravel().tolist(),x=np.ones(c).tolist(),shape=[r,c,c])
                        for op,expected in [('matmul',a@np.ones((c,c))),('matvec',a@np.ones(c)),('transpose',a.T),('trace',np.trace(a)),('frobeniusNorm',np.linalg.norm(a)),('maxAbs',np.max(abs(a)))]:
                            add('matops',op,dtype,inp,expected,representation=rep,label=label,source='numpy')
                        if rep=='Mat':
                            add('matops','det',dtype,inp,np.linalg.det(a),representation=rep,label=label,source='numpy')
                            if np.linalg.cond(a)<1e4:
                                add('matops','inv',dtype,inp,np.linalg.inv(a),representation=rep,label=label,source='numpy')
                    if r==c and r in [2,3,4,6,8,12] and np.linalg.cond(a)<1e4:
                        inp=dict(a=a.ravel().tolist(),b=np.arange(1,r+1,dtype=float).tolist(),shape=[r,r])
                        ops=['gauss','lu','luReuse']
                        if np.allclose(a,a.T) and np.linalg.eigvalsh(a).min()>0:ops+=['cholesky','choleskyReuse']
                        for rep in ['Mat','MatView']:
                            for op in ops:add('linsolve',op,dtype,inp,np.linalg.solve(a,np.arange(1,r+1,dtype=float)),representation=rep,label=label,source='numpy')
