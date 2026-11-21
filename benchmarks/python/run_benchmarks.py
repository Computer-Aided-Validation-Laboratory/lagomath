"""Run a native benchmark, validate workload coverage, and retain host metadata."""
import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools"))
from quiet import run
from simd_harness import transform, supported_benchmark


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault(record['id'], []).append(record['ns'] / record['runs'])
    return {key: dict(min_ns=min(v), median_ns=statistics.median(v), mean_ns=statistics.mean(v), max_ns=max(v), stddev_ns=statistics.pstdev(v), samples=len(v)) for key, v in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--existing', type=Path)
    parser.add_argument('--manifest',type=Path,default=ROOT/'benchmarks/cases/manifest.json')
    parser.add_argument('--backend', choices=['zig', 'mojo'], default='zig')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--runs', type=int)
    parser.add_argument('--implementation',choices=['scalar','simd','locality'],default='scalar')
    parser.add_argument('--simd-comparable',action='store_true',help='restrict scalar run to workloads with explicit SIMD kernels')
    parser.add_argument('--cpu',choices=['baseline','native'],default='baseline')
    parser.add_argument('--include-slow', action='store_true', help='include optional 512/1024 matrix workloads')
    parser.add_argument('--schedule-from',type=Path,help='reuse exact sample/warmup/repetition counts from an ignored generated manifest')
    parser.add_argument('--target-ms', type=float, help='target batch duration using prior calibration summaries')
    parser.add_argument('--calibration', type=Path, nargs='+', help='matching pilot summaries; use fastest backend estimate for shared repetitions')
    args = parser.parse_args()
    if bool(args.calibration) != (args.target_ms is not None):
        parser.error('--target-ms and --calibration must be supplied together')
    if args.target_ms is not None and (not math.isfinite(args.target_ms) or args.target_ms<=0 or args.runs is not None):
        parser.error('--target-ms must be positive and cannot be combined with --runs')
    if args.schedule_from and any(v is not None for v in (args.samples,args.runs,args.target_ms,args.calibration)):
        parser.error('--schedule-from cannot be combined with timing overrides')
    manifest = json.loads(args.manifest.read_text())
    if not args.include_slow:
        manifest['cases']=[c for c in manifest['cases'] if not c.get('optional',False)]
    if args.implementation!='scalar' or args.simd_comparable:
        manifest['cases']=[c for c in manifest['cases'] if supported_benchmark(c)]
    if args.implementation=='locality':
        manifest['cases']=[c for c in manifest['cases'] if c['representation']=='MatView' and c['operation'] in ('matmul','matmulRect')]
    workload={**manifest,'cases':[{k:v for k,v in c.items() if k not in ('warmup','samples','runs_per_sample')} for c in manifest['cases']]}
    workload_hash=hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest()
    calibration=[json.loads(p.read_text()) for p in args.calibration or []]
    case_ids={c['id'] for c in manifest['cases']}
    for pilot in calibration:
        if pilot['target']!='x86-64 '+args.cpu:parser.error('Calibration CPU target does not match')
        if pilot.get('workload_sha256',workload_hash)!=workload_hash:parser.error('Calibration definitions do not match')
        if set(pilot['results'])!=case_ids:parser.error('Calibration workload IDs do not match')
    schedule={c['id']:c for c in json.loads(args.schedule_from.read_text())['cases']} if args.schedule_from else {}
    for case in manifest['cases']:
        if args.schedule_from:
            if case['id'] not in schedule:parser.error('Schedule is missing '+case['id'])
            planned=schedule[case['id']]
            definition=lambda c:{k:v for k,v in c.items() if k not in ('warmup','samples','runs_per_sample')}
            if definition(case)!=definition(planned):parser.error('Scheduled definition differs: '+case['id'])
            for field in ('warmup','samples','runs_per_sample'):case[field]=planned[field]
        if calibration:
            estimate=min(p['results'][case['id']]['median_ns'] for p in calibration)
            if not math.isfinite(estimate) or estimate<=0:parser.error('Invalid calibration timing: '+case['id'])
            case['runs_per_sample']=max(1,math.ceil(args.target_ms*1e6/estimate))
            case['warmup']=max(case['warmup'],min(case['runs_per_sample'],10000))
        if args.samples is not None:
            if args.samples < 2: parser.error('--samples must be at least 2')
            case['samples'] = args.samples
        if args.runs is not None:
            if args.runs < 1: parser.error('--runs must be positive')
            case['runs_per_sample'] = args.runs
    cases = manifest['cases']
    dest = ROOT / 'benchmarks/results'
    dest.mkdir(parents=True, exist_ok=True)
    raw = args.existing
    source = None
    started_utc = None
    finished_utc = None
    suspended_seconds = None
    if raw is None:
        override = ROOT / 'build/bench-override' / args.label
        override.mkdir(parents=True, exist_ok=True)
        (override / 'manifest.json').write_text(json.dumps(manifest))
        (override / 'data.zig').write_text('pub const data = @embedFile("manifest.json");\n')
        env = dict(os.environ, LM_BENCH_MANIFEST=str(override / 'manifest.json'))
        source=ROOT/('benchmarks/mojo/bench.mojo' if args.backend=='mojo' else 'src/benchmark.zig')
        if args.implementation!='scalar':
            adapted=override/('bench.mojo' if args.backend=='mojo' else 'benchmark.zig')
            adapted.write_text(transform(source.read_text(),'mojo' if args.backend=='mojo' else 'zig',benchmark=True,locality=args.implementation=='locality'))
            source=adapted
        started_utc=datetime.now(timezone.utc).isoformat()
        start_active=time.monotonic()
        start_boot=time.clock_gettime(getattr(time,'CLOCK_BOOTTIME',time.CLOCK_MONOTONIC))
        if args.backend == 'mojo':
            target_flags=['--target-cpu','x86-64'] if args.cpu=='baseline' else []
            run([ROOT / '.venv/bin/mojo', 'build', '-O3', '-D', 'ASSERT=none', '--fp-mode','contract=off', *target_flags, '-I', ROOT / 'src/mojo', source, '-o', override / 'mojo-bench'], label='mojo-benchmark-build')
            command = [override / 'mojo-bench','--jsonl']
        else:
            command = ['zig', 'build', 'bench', '-Doptimize=ReleaseFast', '-Dcpu='+args.cpu, '-Dbench-source='+str(source.relative_to(ROOT)), '-Dbench-cases='+str((override/'data.zig').relative_to(ROOT)),'--','--jsonl']
        proc = run(command, cwd=ROOT, env=env, label=args.backend+'-benchmark-'+args.label)
        suspended_seconds=max(0.0,(time.clock_gettime(getattr(time,'CLOCK_BOOTTIME',time.CLOCK_MONOTONIC))-start_boot)-(time.monotonic()-start_active))
        finished_utc=datetime.now(timezone.utc).isoformat()
        if suspended_seconds>0.5:raise RuntimeError(f'System suspended for {suspended_seconds:.1f}s during this run; discard timings and rerun')
        raw = dest / f'{args.label}.jsonl'
        raw.write_text(proc.stderr if args.backend == 'zig' else proc.stdout)
    records = [json.loads(line) for line in raw.read_text().splitlines() if line.startswith('{')]
    expected = {c['id']: c['samples'] for c in cases}
    actual = summarize(records)
    if set(actual) != set(expected): raise RuntimeError('Missing or unexpected benchmark IDs')
    for case in cases:
        matches = [r for r in records if r['id'] == case['id']]
        assert len(matches) == case['samples'], case['id']
        assert {r['sample'] for r in matches} == set(range(case['samples'])), case['id']
        for record in matches:
            assert record['backend'] == args.backend
            assert record['runs'] == case['runs_per_sample'] and record['ns'] >= 0
            assert math.isfinite(record['checksum'])
            assert abs(record['checksum'] - case['expected_checksum']) <= 1e-4 * max(1, abs(case['expected_checksum'])), case['id']
    for case in cases:
        actual[case['id']]['runs_per_sample']=case['runs_per_sample']
        matches=[r['ns']/1e6 for r in records if r['id']==case['id']]
        actual[case['id']]['min_sample_ms']=min(matches)
        actual[case['id']]['median_sample_ms']=statistics.median(matches)
        elements=None
        if case['representation'] in ['Vec','VecView','slice']:elements=case['shape'][0]
        elif case['representation']=='NDArray' and case['operation']=='stream':elements=math.prod(case['shape'])
        if elements:actual[case['id']]['median_ns_per_element']=actual[case['id']]['median_ns']/elements
    cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), platform.machine())
    metadata = dict(label=args.label, backend=args.backend, git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), zig=subprocess.check_output(['zig', 'version'], text=True).strip(), mode='ReleaseFast' if args.backend == 'zig' else '-O3', target='x86-64 '+args.cpu, cpu=cpu, host=platform.platform(), manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(), explicit_simd=args.implementation=='simd', auto_vectorisation='compiler default; instruction selection not audited', timing='native monotonic clock; dispatch and barriers included; allocation and JSON excluded', raw=str(raw), results=actual)
    metadata['started_utc']=started_utc
    metadata['finished_utc']=finished_utc
    metadata['suspended_seconds']=suspended_seconds
    metadata['workload_sha256']=workload_hash
    if source is not None:
        metadata['adapter_source']=str(source.relative_to(ROOT))
        metadata['adapter_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    metadata['implementation']=args.implementation
    metadata['fp_contraction']='off (strict arithmetic)'
    metadata['schedule_from']=str(args.schedule_from) if args.schedule_from else None
    metadata['target_sample_ms']=args.target_ms
    metadata['calibration_files']=[str(p) for p in args.calibration or []]
    metadata['calibration_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.calibration or []}
    metadata['schedule_sha256']=hashlib.sha256(args.schedule_from.read_bytes()).hexdigest() if args.schedule_from else None
    metadata['assertions']='disabled: ReleaseFast for Zig; ASSERT=none for Mojo'
    metadata['working_tree_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True))
    metadata['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*')) if p.suffix in ['.zig','.mojo']}
    metadata['harness_source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/benchmark.zig',ROOT/'benchmarks/mojo/bench.mojo',ROOT/'benchmarks/python/run_benchmarks.py',ROOT/'tools/run_original_baseline.py',ROOT/'tools/simd_harness.py']}
    if args.backend == 'mojo': metadata['mojo'] = subprocess.check_output([ROOT / '.venv/bin/mojo', '--version'], text=True).strip()
    previous=dest/f'{args.label}.summary.json'
    if args.existing is not None and previous.exists() and previous.stat().st_mtime>=raw.stat().st_mtime:
        provenance=json.loads(previous.read_text())
        if provenance['manifest_sha256']!=metadata['manifest_sha256']:raise RuntimeError('Existing summary belongs to another workload')
        provenance['results']=actual
        metadata=provenance
    (dest / f'{args.label}.summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{len(actual)} cases validated; summary: {dest / (args.label + ".summary.json")}')


if __name__ == '__main__': main()
