"""Isolated cffi ABI validation. Every numerical reference comes from shared JSON."""
import json
import re
import sys
from pathlib import Path
import cffi
import numpy as np
from validate_cases import ROOT,load_cases


def declarations():
    text=(ROOT/'include/lagomath.h').read_text()
    return '\n'.join(line for line in text.splitlines() if not line.startswith('#') and line not in ['extern "C" {','}'])

def run(library):
    ffi=cffi.FFI();ffi.cdef(declarations());lib=ffi.dlopen(str(Path(library).resolve()))
    codes={k:int(v) for k,v in re.findall(r'#define LM_(\w+) (\d+)',(ROOT/'include/lagomath.h').read_text())}
    assert lib.lm_abi_version()==codes['ABI_VERSION']
    for symbol in re.findall(r'\b(lm_\w+)\(',declarations()):getattr(lib,symbol)
    import subprocess
    probe=ROOT/'build/c-layout-probe.c'
    probe.write_text('#include <stddef.h>\n#include <stdio.h>\n#include "lagomath.h"\nint main(void){printf("%zu %zu %zu",sizeof(lm_mat22_f64),_Alignof(lm_mat22_f64),offsetof(lm_mat22_f64,data));}\n')
    exe=ROOT/'build/c-layout-probe'
    subprocess.run(['cc','-I',str(ROOT/'include'),str(probe),'-o',str(exe)],check=True)
    size,alignment,offset=map(int,subprocess.check_output([exe],text=True).split())
    assert lib.lm_mat22_size_f64()==size==ffi.sizeof('lm_mat22_f64')
    assert lib.lm_mat22_align_f64()==alignment==ffi.alignof('lm_mat22_f64')
    assert offset==ffi.offsetof('lm_mat22_f64','data')
    fixture=next(c for c in load_cases() if c['id']=='matops.transpose.Mat.2x2x2.f64')
    src=ffi.new('lm_mat22_f64 *');dst=ffi.new('lm_mat22_f64 *')
    for i,v in enumerate(fixture['inputs']['a']):src.data[i]=v
    assert lib.lm_mat22_transpose_f64(src,dst)==codes['OK']
    np.testing.assert_allclose(list(dst.data),fixture['expected'],**fixture['tolerance'])
    guard=json.loads((ROOT/'tests/cases/manifest.json').read_text())['abi_output_guard']
    prefix,suffix=guard['prefix'],guard['suffix']
    report=[]
    for case in load_cases():
        if case['id']==fixture['id']:
            report.append(dict(id=case['id'],status='pass'));continue
        op=case['operation'];suite=case['suite'];dtype=case['dtype'];dt=np.float32 if dtype=='f32' else np.float64;ct='float' if dtype=='f32' else 'double';inp=case['inputs'];shape=inp['shape']
        if not ((suite=='vecops' and op in ['dot','axpy']) or (suite=='matops' and op in ['matmul','matvec']) or (suite=='linsolve' and op in ['gauss','cholesky'])):
            report.append(dict(id=case['id'],status='N/A',reason='outside narrow C ABI'));continue
        decode=lambda values: [float(v['special']) if isinstance(v,dict) else v for v in values]
        if suite=='linsolve' and len(shape)>1 and shape[0]!=shape[1]:
            report.append(dict(id=case['id'],status='N/A',reason='ABI takes a square matrix dimension'));continue
        a=np.asarray(decode(inp['a']),dtype=dt);b=np.asarray(decode(inp.get('b',[])),dtype=dt)
        if 'expected_error' not in case and (not np.isfinite(a).all() or not np.isfinite(b).all()):
            report.append(dict(id=case['id'],status='N/A',reason='C ABI validates finite inputs; native standard kernels propagate nonfinite values'))
            continue
        def pointer(array):
            assert array.dtype==dt and array.flags.c_contiguous and array.flags.aligned
            return ffi.from_buffer(f'{ct}[]',array)
        if op=='dot':size=1
        elif op=='axpy':size=len(b)
        elif op=='matmul':size=shape[0]*shape[2]
        else:size=shape[0]
        parent=np.full(size+prefix+suffix,guard['value'],dtype=dt);out=parent[prefix:prefix+size]
        if op=='dot':status=getattr(lib,f'lm_dot_{dtype}')(pointer(a),pointer(b),len(a),pointer(out))
        elif op=='axpy':
            out[:]=b;status=getattr(lib,f'lm_axpy_{dtype}')(inp['alpha'],pointer(a),pointer(out),len(a))
        elif op=='matmul':status=getattr(lib,f'lm_matmul_{dtype}')(pointer(a),pointer(b),pointer(out),*shape)
        elif op=='matvec':
            x=np.asarray(inp['x'],dtype=dt);status=getattr(lib,f'lm_matvec_{dtype}')(pointer(a),shape[0],shape[1],pointer(x),pointer(out))
        else:
            scratch=np.empty(size*size+size,dtype=dt);status=getattr(lib,f'lm_solve_{op}_{dtype}')(pointer(a),pointer(b),pointer(out),size,pointer(scratch),len(scratch))
        np.testing.assert_array_equal(parent[:prefix],guard['value'])
        np.testing.assert_array_equal(parent[prefix+size:],guard['value'])
        if 'expected_error' in case:assert status=={'SingularMatrix':codes['SINGULAR'],'NotPositiveDefinite':codes['NOT_POSITIVE_DEFINITE'],'ShapeMismatch':codes['SHAPE_MISMATCH'],'NonFiniteInput':codes['NONFINITE']}[case['expected_error']],(case['id'],status)
        else:
            assert status==codes['OK'],(case['id'],status)
            np.testing.assert_allclose(out,case['expected'],**case['tolerance'],err_msg=case['id'])
        report.append(dict(id=case['id'],status='pass'))
    # Mechanical ABI argument validation; numerical expectations stay in fixtures.
    for dtype in ['f32','f64']:
        ct='float' if dtype=='f32' else 'double';r=ffi.new(f'{ct} *');dot=getattr(lib,f'lm_dot_{dtype}')
        assert dot(ffi.NULL,ffi.NULL,-1,r)==codes['INVALID_ARGUMENT']
        assert dot(ffi.NULL,ffi.NULL,1,r)==codes['INVALID_ARGUMENT']
        assert dot(ffi.NULL,ffi.NULL,0,r)==codes['OK']
        assert dot(ffi.NULL,ffi.NULL,0,ffi.NULL)==codes['INVALID_ARGUMENT']
        buf=ffi.new(f'{ct}[]',20)
        assert getattr(lib,f'lm_matmul_{dtype}')(buf,buf,buf,2,2,2)==codes['INVALID_ALIASING']
        assert getattr(lib,f'lm_solve_gauss_{dtype}')(buf,buf+4,buf+6,2,buf+8,1)==codes['SHAPE_MISMATCH']
        buf[0]=float('nan');buf[1]=1
        assert dot(buf,buf+1,1,r)==codes['NONFINITE']
    return report

if __name__=='__main__':
    report=run(sys.argv[1]);print(json.dumps(report))
