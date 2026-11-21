"""Focused small/medium/streaming arithmetic workloads for SIMD A/B evaluation."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    path=ROOT/'benchmarks/cases/manifest.json'
    manifest=json.loads(path.read_text())
    manifest['cases']=[c for c in manifest['cases'] if not c.get('simd_evaluation',False)]
    existing={c['id'] for c in manifest['cases']}
    additions=[]
    def add(case):
        if case['id'] not in existing:
            case['simd_evaluation']=True
            additions.append(case)
    for dtype in ('f32','f64'):
        for n in (3,33,4096,65537):
            ops={'normL1':n,'normL2':math.sqrt(n),'normL2Square':n,'min':1,'mean':1,
                 'maxAbs':1,'allFinite':1,'sub':0,'div':1,'scale':0.5,'fill':2,
                 'permute':1,'axpy':1.5,'dotBatch3':3}
            for op,checksum in ops.items():
                add(dict(id=f'bench.{op}.slice.{n}.{dtype}',operation=op,dtype=dtype,
                    representation='slice',shape=[n,3] if op=='dotBatch3' else [n],
                    warmup=3,samples=10,runs_per_sample=100 if n<100 else 5,
                    expected_checksum=checksum,input_family='deterministic',
                    input_rule='all ones; scale/axpy alpha=1/2, fill=2; permutation=(17*i)%N; batch dot uses AoS vectors of length 3',
                    aliasing='disjoint borrowed buffers; axpy output reset included in timing'))
        for n in (3,33,128,512):
            ops={'subMatrix':0,'mulMatrix':(n+1)**2,'divMatrix':1,
                 'scaleMatrix':(n+1)/2,'fillMatrix':2,
                 'matrixNorm':math.sqrt(n*(n+1)**2+n*(n-1)/64),
                 'matrixMaxAbs':n+1,'matrixAllFinite':1,'trace':n*(n+1),
                 'outer':1,'outerAdd':n+1+0.5,'scaleRow':(n+1)/2,
                 'addScaledRow':0.125+(n+1)/2}
            for op,checksum in ops.items():
                add(dict(id=f'bench.{op}.MatView.{n}.{dtype}',operation=op,dtype=dtype,
                    representation='MatView',shape=[n,n],warmup=3,samples=10,
                    runs_per_sample=5,expected_checksum=checksum,input_family='deterministic',
                    input_rule='matrix diagonal=N+1, off-diagonal=1/8; vectors ones; scale/row updates alpha=1/2; fill=2',
                    aliasing='disjoint borrowed buffers; update input resets included in timing',optional=n>=512))
    manifest['cases']+=additions
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{len(additions)} SIMD evaluation definitions; {len(manifest["cases"])} total workloads')


if __name__=='__main__':main()
