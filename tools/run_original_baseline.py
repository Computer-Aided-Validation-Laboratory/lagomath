"""Run the current workload/harness against an immutable original Zig revision."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from quiet import run

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL='d4ebf9010496d4196dc8e8b87b8b4583a008441e'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label')
    parser.add_argument('--samples',type=int)
    parser.add_argument('--runs',type=int)
    args=parser.parse_args()
    label=args.label or ('zig-original-current-harness-smoke' if args.samples is not None or args.runs is not None else 'zig-original-current-harness')
    dest=ROOT/'benchmarks/results';dest.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'benchmarks/cases/manifest.json').read_text())
    # New SIMD evaluation operations did not exist in the archived API.
    manifest['cases']=[c for c in manifest['cases'] if not c.get('simd_evaluation',False)]
    for case in manifest['cases']:
        if args.samples is not None:case['samples']=args.samples
        if args.runs is not None:case['runs_per_sample']=args.runs
    if any(c['samples']<2 or c['runs_per_sample']<1 for c in manifest['cases']):parser.error('invalid sample/repetition counts')
    manifest_path=dest/(label+'.manifest.json')
    manifest_path.write_text(json.dumps(manifest))
    with tempfile.TemporaryDirectory(prefix='lagomath-baseline-') as directory:
        base=Path(directory)
        archive=subprocess.check_output(['git','archive',ORIGINAL],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(base,filter='data')
        shutil.copytree(ROOT/'benchmarks/cases',base/'benchmarks/cases')
        (base/'benchmarks/cases/manifest.json').write_text(json.dumps(manifest))
        source=(ROOT/'src/benchmark.zig').read_text()
        source=source.replace('checksum = lm.vecops.normL1VecView(T, v)', 'return error.OperationAbsentFromOriginal')
        replacements={'MatViewOps':'MatSliceOps','Vec':'VecStack','Mat':'MatStack','VecView':'VecSlice','MatView':'MatSlice','dotVec':'dotStack','dotVecView':'dotSlice','normL2Vec':'normStack','axpyVec':'axpyStack','mulVecMat':'mulVecStack','mulMatMat':'mulMatStack','outerMat':'outerStack','outerAddMat':'outerAddStack','swapRowsMat':'swapRowsStack'}
        for name in ['solveGauss','factorLU','solveLU','solveLUFactored','factorCholesky','solveCholesky','solveCholeskyFactored']:
            replacements[name+'Mat']=name+'Stack';replacements[name+'MatView']=name+'Slice'
        for name in ['mulVec','mulMat','outer','outerAdd','swapRows','scaleRow','addScaledRow']:replacements[name+'MatView']=name+'Slice'
        replacements.update({'axpyVecView':'axpySlice','normL2VecView':'normSlice','normL2SquareVecView':'squaredNormSlice','applyPermutationVec':'applyPermutationStack','applyPermutationVecView':'applyPermutationSlice'})
        for prefix in ['maxAbs','allFinite']:
            replacements[prefix+'VecView']=prefix+'Slice'
        for prefix in ['frobeniusNorm','maxAbs','allFinite']:replacements[prefix+'MatView']=prefix+'Slice'
        for old,new in replacements.items():source=re.sub(r'\b'+old+r'\b',new,source)
        for old,new in [('VecStack','Vec'),('MatStack','Mat'),('VecSlice','VecView'),('MatSlice','MatView')]:source=source.replace(chr(34)+old+chr(34),chr(34)+new+chr(34))
        for name in ['Mat22Ops','Mat33Ops','Mat44Ops']:source=source.replace('lm.'+name,'lm.matstack.'+name)
        (base/'src/benchmark.zig').write_text(source)
        (base/'build.zig').write_text('''const std=@import("std");
pub fn build(b:*std.Build) void {
    const target=b.standardTargetOptions(.{});const optimize=b.standardOptimizeOption(.{});
    const cases=b.createModule(.{.root_source_file=b.path("benchmarks/cases/data.zig")});
    const exe=b.addExecutable(.{.name="baseline",.root_module=b.createModule(.{
        .root_source_file=b.path("src/benchmark.zig"),.target=target,.optimize=optimize,.link_libc=true,
        .imports=&.{.{.name="bench_cases",.module=cases}},
    })});
    const run=b.addRunArtifact(exe);if(b.args)|args|run.addArgs(args);
    b.step("bench","Run original kernels").dependOn(&run.step);
}
''')
        proc=run(['zig','build','bench','-Doptimize=ReleaseFast','-Dcpu=baseline','--','--jsonl'],cwd=base,label='original-kernel-benchmark')
    dest=ROOT/'benchmarks/results';dest.mkdir(parents=True,exist_ok=True)
    raw=dest/(label+'.jsonl');raw.write_text(proc.stderr)
    subprocess.run([sys.executable,ROOT/'benchmarks/python/run_benchmarks.py','--label',label,'--include-slow','--manifest',manifest_path,'--existing',raw],cwd=ROOT,check=True)
    path=dest/(label+'.summary.json');metadata=json.loads(path.read_text());metadata['kernel_git_sha']=ORIGINAL;metadata['harness_tree_source_hashes']=metadata.pop('source_hashes',{});metadata['harness_sha256']=hashlib.sha256((ROOT/'src/benchmark.zig').read_bytes()).hexdigest();path.write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
