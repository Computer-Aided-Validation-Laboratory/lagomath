"""Generate static SIMD test/benchmark adapters without changing scalar APIs.

Only dispatch glue is generated. All numerical implementations live in the
reviewable *_simd source modules. Shared oracle fixtures are never rewritten.
"""
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODULES=('sliceops','vecops','matops','rowops','linsolve')
BENCH_OPERATIONS={'dot','normL2','axpy','sum','max','add','mul','matmul','matvec',
                  'outer','outerAdd','transpose','rowops','gauss','lu','cholesky',
                  'factorLU','factorCholesky','luReuse','choleskyReuse','luBatch16',
                  'choleskyBatch16','det','inv','matmulRect','matvecRect',
                  'transposeRect','rowopsRect','addMatrix','stream','normL1','normL2Square','min','mean','maxAbs','allFinite','sub','div','scale','fill','permute','dotBatch3','subMatrix','mulMatrix','divMatrix','scaleMatrix','fillMatrix','matrixNorm','matrixMaxAbs','matrixAllFinite','trace','scaleRow','addScaledRow'}


def supported_benchmark(case):
    return case['operation'] in BENCH_OPERATIONS


def supported_case(case):
    return case['suite'] in ('vecops','matops','rowops','linsolve')


def transform(source,language,*,benchmark=False,locality=False):
    if language=='zig':
        source=source.replace('@import("root.zig")','@import("lagomath")')
        # Extended/metadata fixtures are outside this numerical adapter.
        source=source.replace('@import("extended_cases.zig").run(T, c, a, b)','error.UnsupportedSIMDCase')
    # Statically bind the canonical dispatch calls to explicit SIMD symbols.
    for module in MODULES:
        source=re.sub(r'lm\.'+module+r'\.(\w+)',lambda m:f'lm.{module}_simd.{m[1]}SIMD',source)
    if language=='zig':
        # Fixed and borrowed vector methods in the shared fixture adapter.
        if not benchmark:
            start=source.index('fn vectorFixed(');end=source.index('fn matrixFixed(')
            part=source[start:end]
            for op in ('normL1','normL2','normL2Square','sum','mean','min','max'):
                part=part.replace(f'v.{op}()',f'lm.vecops_simd.{op}VecSIMD(N,T,v)')
            for op in ('add','sub'):part=part.replace(f'v.{op}(w)',f'lm.vecops_simd.{op}VecSIMD(N,T,v,w)')
            source=source[:start]+part+source[end:]
            start=source.index('fn run(')
            part=source[start:]
            for op in ('normL1','normL2','normL2Square','sum','mean','min','max'):
                part=part.replace(f'v.{op}()',f'lm.vecops_simd.{op}VecViewSIMD(T,v)')
            for op in ('add','sub','mul','div'):
                part=part.replace(f'destination.{op}InPlace(&w)',f'lm.vecops_simd.{op}VecViewSIMD(T,destination,w,destination)')
            source=source[:start]+part
        source=source.replace('matrix.trace()','lm.matops_simd.traceMatViewSIMD(T,matrix)')
        # The fixed matrix section has compile-time dimensions.
        if not benchmark:
            start=source.index('fn matrixFixed(');end=source.index('fn solveFixed(')
            part=source[start:end].replace('lm.matops_simd.traceMatViewSIMD(T,matrix)','lm.matops_simd.traceMatSIMD(R,C,T,matrix)')
            part=part.replace('matrix.transpose()','lm.matops_simd.transposeMatSIMD(R,C,T,matrix)')
            part=part.replace('ops.det(T, matrix)','lm.smallmatops_simd.detMatSIMD(R,T,matrix)').replace('ops.invChecked(T, matrix, 0)','lm.smallmatops_simd.invMatCheckedSIMD(R,T,matrix,0)')
            part=re.sub(r'            const ops = [^\n]+;\n','',part)
            source=source[:start]+part+source[end:]
            source=source.replace('matrix.transpose(&dest)','lm.matops_simd.transposeMatViewSIMD(T,matrix,dest)')
        else:
            source=source.replace('a.transpose()','lm.matops_simd.transposeMatSIMD(N,N,T,a.*)')
            source=source.replace('matrix.transpose(&dest)','lm.matops_simd.transposeMatViewSIMD(T,matrix,dest)')
            source=source.replace('a.transpose()','lm.matops_simd.transposeMatSIMD(R,C,T,a)')
            # The rectangular fixed branch uses a value, not the kernel pointer.
            source=source.replace('lm.matops_simd.transposeMatSIMD(N,N,T,a.*) else av.transpose(&tv)','lm.matops_simd.transposeMatSIMD(R,C,T,a) else lm.matops_simd.transposeMatViewSIMD(T,av,tv)')
            source=source.replace('ops.det(T, a.*)','lm.smallmatops_simd.detMatSIMD(N,T,a.*)').replace('ops.inv(T, a.*)','lm.smallmatops_simd.invMatSIMD(N,T,a.*)')
            source=re.sub(r'        const ops = [^\n]+;\n','',source)
            source=source.replace('lm.MatViewOps(T).add(&matrix, &matrix2, &dest)','lm.matops_simd.addMatViewSIMD(T,matrix,matrix2,dest)')
            source=source.replace('''                for (array.slice) |v| checksum += v;''','''                checksum = lm.sliceops_simd.sumSIMD(T,array.slice);''')
        if benchmark:
            source=source.replace('batchDotScalar(T,a,b,out[0..c.shape[0]])','lm.vecops_simd.dotBatchSIMD(3,T,a,b,out[0..c.shape[0]])').replace('batchDotScalar(T, a, b, out[0..c.shape[0]])','lm.vecops_simd.dotBatchSIMD(3,T,a,b,out[0..c.shape[0]])')
            source=source.replace('@memset(out, 2)','lm.sliceops_simd.fillSIMD(T,out,2)')
            source=source.replace('dest.fill(2)','lm.matops_simd.fillMatViewSIMD(T,dest,2)')
            for method,name in [('sub','sub'),('mulElemWise','mulElemWise'),('divElemWise','divElemWise')]:
                source=source.replace(f'lm.MatViewOps(T).{method}(&matrix, &matrix2, &dest)',f'lm.matops_simd.{name}MatViewSIMD(T,matrix,matrix2,dest)')
            source=source.replace('lm.MatViewOps(T).mulScal(&matrix, 0.5, &dest)','lm.matops_simd.mulScalMatViewSIMD(T,matrix,0.5,dest)')
        if locality:
            source=source.replace('lm.matops_simd.mulMatMatViewSIMD(T, matrix, matrix2, dest)','matmulLocal(T,matrix,matrix2,dest)').replace('lm.matops_simd.mulMatMatViewSIMD(T, av, bv, ov)','matmulLocal(T,av,bv,ov)')
            source+='''
fn matmulLocal(comptime T:type,a:lm.MatView(T),b:lm.MatView(T),out:lm.MatView(T)) void {
    const W=32/@sizeOf(T);
    var row:usize=0;
    while(row<a.rows_num) : (row+=4) {
        var col:usize=0;
        while(b.cols_num-col>=W) : (col+=W) {
            var acc:[4][W]T=@splat(@splat(0));
            for(0..a.cols_num) |k| {
                inline for(0..4) |r| if(row+r<a.rows_num) {
                    inline for(0..W) |j| acc[r][j]+=a.get(row+r,k)*b.get(k,col+j);
                };
            }
            inline for(0..4) |r| if(row+r<a.rows_num) {
                inline for(0..W) |j| out.set(row+r,col+j,acc[r][j]);
            };
        }
        while(col<b.cols_num) : (col+=1) {
            for(row..@min(row+4,a.rows_num)) |r| {var v:T=0;for(0..a.cols_num) |k| v+=a.get(r,k)*b.get(k,col);out.set(r,col,v);}
        }
    }
}
'''
    else:
        for v in ('v','av'):
            suffix='Vec' if v=='v' else 'VecView'
            for op in ('sum','mean','min','max'):
                source=re.sub(r'\b'+v+r'\.'+op+r'\(\)',f'lm.vecops_simd.{op}{suffix}SIMD({v})',source)
        for op in ('add','sub'):source=source.replace(f'v.{op}(w)',f'lm.vecops_simd.{op}VecSIMD(v,w)')
        for op in ('add','sub','mul','div'):
            source=source.replace(f'dst.{op}InPlace(bv)',f'lm.vecops_simd.{op}VecViewSIMD(av,bv,dst)')
        source=source.replace('matrix.transpose()','lm.matops_simd.transposeMatSIMD(matrix)')
        source=source.replace('matrix.transpose(lm.MatView(output, shape[1], shape[0]))','lm.matops_simd.transposeMatViewSIMD(matrix,lm.MatView(output,shape[1],shape[0]))')
        source=source.replace('matrix.trace()','lm.matops_simd.traceMatSIMD(matrix)')
        if not benchmark:
            start=source.index('def run[')
            source=source[:start]+source[start:].replace('lm.matops_simd.traceMatSIMD(matrix)','lm.matops_simd.traceMatViewSIMD(matrix)')
        for n in (2,3,4):
            source=source.replace(f'lm.Mat{n}{n}Ops.det(', 'lm.smallmatops_simd.detMatSIMD(').replace(f'lm.Mat{n}{n}Ops.invChecked(', 'lm.smallmatops_simd.invMatCheckedSIMD(').replace(f'lm.Mat{n}{n}Ops.inv(', 'lm.smallmatops_simd.invMatSIMD(')
        if benchmark:
            source=source.replace('''                for j in range(total):
                    checksum += array.getFlat(j)''','''                checksum = lm.sliceops_simd.sumSIMD(array.data.as_imm())''')
            source=source.replace('a.transpose()','lm.matops_simd.transposeMatSIMD(a)').replace('av.transpose(tv)','lm.matops_simd.transposeMatViewSIMD(av,tv)').replace('matrix.transpose(wm)','lm.matops_simd.transposeMatViewSIMD(matrix,wm)')
            source=source.replace('lm.MatViewOps[dtype].add(matrix, matrix2, wm)','lm.matops_simd.addMatViewSIMD(matrix,matrix2,wm)')
        if benchmark:
            source=source.replace('batchDotScalar(av.data.as_imm(),bv.data.as_imm(),dst[0:n])','lm.vecops_simd.dotBatchSIMD[3](av.data.as_imm(),bv.data.as_imm(),dst[0:n])')
            source=source.replace('dst.fill(2)','lm.sliceops_simd.fillSIMD(dst,Scalar[dtype](2))').replace('wm.fill(2)','lm.matops_simd.fillMatViewSIMD(wm,Scalar[dtype](2))')
            for method in ('sub','mulElemWise','divElemWise'):
                source=source.replace(f'lm.MatViewOps[dtype].{method}(matrix,matrix2,wm)',f'lm.matops_simd.{method}MatViewSIMD(matrix,matrix2,wm)')
            source=source.replace('lm.MatViewOps[dtype].mulScal(matrix,Scalar[dtype](0.5),wm)','lm.matops_simd.mulScalMatViewSIMD(matrix,Scalar[dtype](0.5),wm)')
            source=source.replace('lm.matops_simd.traceMatSIMD(matrix)','lm.matops_simd.traceMatViewSIMD(matrix)')
        if locality:
            source='from std.collections import Array\n'+source
            source=source.replace('lm.matops_simd.mulMatMatViewSIMD(matrix, matrix2, wm)','matmulLocal(matrix,matrix2,wm)').replace('lm.matops_simd.mulMatMatViewSIMD(av, bv, ov)','matmulLocal(av,bv,ov)')
            source+='''

def matmulLocal[dtype: DType,oa: Origin,ob: Origin,od: Origin[mut=True]](a: lm.MatView[dtype,oa],b: lm.MatView[dtype,ob],dst: lm.MatView[dtype,od]):
    comptime W = 8 if dtype==DType.float32 else 4
    var row = 0
    while row<a.rows_num:
        var col = 0
        while col+W<=b.cols_num:
            var acc = Array[Scalar[dtype],4*W](fill=0)
            for k in range(a.cols_num):
                comptime for r in range(4):
                    if row+r<a.rows_num:
                        comptime for j in range(W):
                            acc[r*W+j] += a.get(row+r,k)*b.get(k,col+j)
            comptime for r in range(4):
                if row+r<a.rows_num:
                    comptime for j in range(W):
                        dst.set(row+r,col+j,acc[r*W+j])
            col += W
        while col<b.cols_num:
            for r in range(row,min(row+4,a.rows_num)):
                var value: Scalar[dtype] = 0
                for k in range(a.cols_num):
                    value += a.get(r,k)*b.get(k,col)
                dst.set(r,col,value)
            col += 1
        row += 4
'''
    if language=='zig':
        for name in ('matrix','dest','tv'):
            source=source.replace(f'var {name} = lm.MatView',f'const {name} = lm.MatView')
    return source
