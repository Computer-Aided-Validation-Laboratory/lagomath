"""Regenerate the medium/large borrowed-matrix benchmark definitions."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    path=ROOT/'benchmarks/cases/manifest.json'
    manifest=json.loads(path.read_text())
    additions=[]
    for dtype in ('f32','f64'):
        for n in (16,32,64,127,128,256,512,1024):
            operations=['matvec','transpose','addMatrix']
            if n<=512:operations+=['matmul']
            if n in (16,32,64,128,256,512):
                operations+=['lu','cholesky','factorLU','factorCholesky','luReuse','choleskyReuse']
            for op in operations:
                checksum={'matvec':n+1+(n-1)/8,'transpose':n+1,'addMatrix':2*(n+1),
                          'matmul':(n+1)**2+(n-1)/64,'factorLU':n+1,
                          'factorCholesky':math.sqrt(n+1)}.get(op,1/(n+1+(n-1)/8))
                additions.append(dict(id=f'bench.{op}.MatView.{n}.{dtype}',operation=op,dtype=dtype,
                    representation='MatView',shape=[n,n],warmup=3,samples=10,runs_per_sample=5,
                    expected_checksum=checksum,input_family='deterministic',
                    input_rule='matrix diagonal=N+1, off-diagonal=1/8; RHS ones',
                    aliasing='disjoint borrowed input/output buffers; setup outside timed region',
                    optional=n>=512))
        for r,k,c in ((32,128,16),(128,32,64)):
            additions.append(dict(id=f'bench.matmulRect.MatView.{r}x{k}x{c}.{dtype}',
                operation='matmulRect',dtype=dtype,representation='MatView',shape=[r,k,c],
                warmup=3,samples=10,runs_per_sample=5,expected_checksum=k,input_family='deterministic',
                input_rule='all input elements are one',
                aliasing='disjoint borrowed input/output buffers; setup outside timed region'))
    ids={c['id'] for c in additions}
    manifest['cases']=[c for c in manifest['cases'] if c['id'] not in ids]+additions
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{len(additions)} extended matrix workloads; {len(manifest["cases"])} total')


if __name__=='__main__':main()
