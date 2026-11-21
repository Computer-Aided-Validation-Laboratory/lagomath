"""Sequential native automatic/SIMD A/B runs with one shared timing schedule.

Results and generated schedules stay in ignored machine-local directories.
Each child rejects timings if CLOCK_BOOTTIME detects a system suspend.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'benchmarks/results'
RUNNER=ROOT/'benchmarks/python/run_benchmarks.py'
VARIANTS=[('zig','scalar','auto'),('zig','simd','manual'),('mojo','scalar','auto'),('mojo','simd','manual')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label-prefix',default='simd-native')
    parser.add_argument('--samples',type=int,default=10)
    parser.add_argument('--target-ms',type=float,default=10)
    parser.add_argument('--include-slow',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.label_prefix):parser.error('invalid label prefix')
    if args.samples<2 or not math.isfinite(args.target_ms) or args.target_ms<=0:parser.error('invalid timing settings')
    def run(backend,implementation,label,timing):
        print(f'{datetime.now(timezone.utc).isoformat()} {label}',flush=True)
        command=[sys.executable,RUNNER,'--label',label,'--backend',backend,
                 '--implementation',implementation,'--cpu','native']
        if implementation=='scalar':command+=['--simd-comparable']
        if args.include_slow:command+=['--include-slow']
        subprocess.run(command+list(map(str,timing)),cwd=ROOT,check=True)
        return RESULTS/f'{label}.summary.json'
    pilots=[]
    for backend,implementation,name in VARIANTS:
        # Stable pilot paths allow executable reuse; every timing is freshly run.
        pilots.append(run(backend,implementation,f'{backend}-{name}-pilot',['--samples',2]))
    calibrations=[]
    for backend,implementation,name in VARIANTS:
        calibrations.append(run(backend,implementation,f'{backend}-{name}-calibration',
            ['--samples',2,'--target-ms',2,'--calibration',*pilots]))
    final={}
    for backend,implementation,name in VARIANTS:
        label=f'{args.label_prefix}-{backend}-{name}'
        final[(backend,name)]=run(backend,implementation,label,
            ['--samples',args.samples,'--target-ms',args.target_ms,'--calibration',*calibrations])
    controls={}
    schedule=ROOT/'build/bench-override'/f'{args.label_prefix}-zig-auto'/'manifest.json'
    for backend in ('zig','mojo'):
        controls[backend]=run(backend,'locality',f'{args.label_prefix}-{backend}-locality',['--schedule-from',schedule])
    summaries={key:json.loads(path.read_text()) for key,path in final.items()}
    assert len({d['manifest_sha256'] for d in summaries.values()})==1,'Timing schedules differ'
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),samples=args.samples,
                target_ms=args.target_ms,target='native',fp_contraction='off',
                note='speedup=automatic median/manual median; values above 1 favour manual SIMD',
                inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(final.values())+list(controls.values())},
                backends={})
    for backend in ('zig','mojo'):
        auto=summaries[(backend,'auto')]['results'];manual=summaries[(backend,'manual')]['results']
        control=json.loads(controls[backend].read_text())['results']
        assert auto.keys()==manual.keys()
        rows={}
        for key,a in auto.items():
            m=manual[key]
            assert a['runs_per_sample']==m['runs_per_sample']
            row=dict(automatic=a,manual=m,speedup=a['median_ns']/m['median_ns'])
            if key in control:
                c=control[key]
                assert c['runs_per_sample']==m['runs_per_sample']
                row.update(locality_control=c,manual_vs_locality_speedup=c['median_ns']/m['median_ns'])
            rows[key]=row
        report['backends'][backend]=rows
    path=RESULTS/f'{args.label_prefix}.comparison.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Validated {len(rows)} shared workloads in all four variants; report: {path}',flush=True)


if __name__=='__main__':main()
