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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools"))
from quiet import run


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault(record['id'], []).append(record['ns'] / record['runs'])
    return {key: dict(min_ns=min(v), median_ns=statistics.median(v), mean_ns=statistics.mean(v), max_ns=max(v), stddev_ns=statistics.pstdev(v), samples=len(v)) for key, v in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--existing', type=Path)
    parser.add_argument('--backend', choices=['zig', 'mojo'], default='zig')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--runs', type=int)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'benchmarks/cases/manifest.json').read_text())
    for case in manifest['cases']:
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
    if raw is None:
        override = ROOT / 'build/bench-override'
        override.mkdir(parents=True, exist_ok=True)
        (override / 'manifest.json').write_text(json.dumps(manifest))
        (override / 'data.zig').write_text('pub const data = @embedFile("manifest.json");\n')
        env = dict(os.environ, LM_BENCH_MANIFEST=str(override / 'manifest.json'))
        if args.backend == 'mojo':
            run([ROOT / '.venv/bin/mojo', 'build', '-O3', '-D', 'ASSERT=none', '--target-cpu', 'x86-64', '-I', ROOT / 'src/mojo', ROOT / 'benchmarks/mojo/bench.mojo', '-o', ROOT / 'build/mojo-bench'], label='mojo-benchmark-build')
            command = [ROOT / 'build/mojo-bench','--jsonl']
        else:
            command = ['zig', 'build', 'bench', '-Doptimize=ReleaseFast', '-Dcpu=baseline', '-Dbench-cases=build/bench-override/data.zig','--','--jsonl']
        proc = run(command, cwd=ROOT, env=env, label=args.backend+'-benchmark-'+args.label)
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
        elements=None
        if case['representation'] in ['Vec','VecView','slice']:elements=case['shape'][0]
        elif case['representation']=='NDArray' and case['operation']=='stream':elements=math.prod(case['shape'])
        if elements:actual[case['id']]['median_ns_per_element']=actual[case['id']]['median_ns']/elements
    cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), platform.machine())
    metadata = dict(label=args.label, backend=args.backend, git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), zig=subprocess.check_output(['zig', 'version'], text=True).strip(), mode='ReleaseFast' if args.backend == 'zig' else '-O3', target='x86-64 baseline', cpu=cpu, host=platform.platform(), manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(), explicit_simd=False, auto_vectorisation='compiler default; instruction selection not audited', timing='native monotonic clock; dispatch and barriers included; allocation and JSON excluded', raw=str(raw), results=actual)
    metadata['assertions']='disabled: ReleaseFast for Zig; ASSERT=none for Mojo'
    metadata['working_tree_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True))
    metadata['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*')) if p.suffix in ['.zig','.mojo']}
    metadata['harness_source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/benchmark.zig',ROOT/'benchmarks/mojo/bench.mojo',ROOT/'benchmarks/python/run_benchmarks.py',ROOT/'tools/run_original_baseline.py']}
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
