"""Run the current workload/harness against an immutable original Zig revision."""
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
    with tempfile.TemporaryDirectory(prefix='lagomath-baseline-') as directory:
        base=Path(directory)
        archive=subprocess.check_output(['git','archive',ORIGINAL],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(base,filter='data')
        shutil.copytree(ROOT/'benchmarks/cases',base/'benchmarks/cases')
        source=(ROOT/'src/benchmark.zig').read_text()
        replacements={'Vec':'VecStack','Mat':'MatStack','VecView':'VecSlice','MatView':'MatSlice','dotVec':'dotStack','dotVecView':'dotSlice','normL2Vec':'normStack','axpyVec':'axpyStack','mulVecMat':'mulVecStack','mulMatMat':'mulMatStack','outerMat':'outerStack','outerAddMat':'outerAddStack','swapRowsMat':'swapRowsStack'}
        for name in ['solveGauss','factorLU','solveLU','solveLUFactored','factorCholesky','solveCholesky','solveCholeskyFactored']:
            replacements[name+'Mat']=name+'Stack';replacements[name+'MatView']=name+'Slice'
        for name in ['mulVec','mulMat','outer','outerAdd','swapRows']:replacements[name+'MatView']=name+'Slice'
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
    raw=dest/'zig-original-current-harness.jsonl';raw.write_text(proc.stderr)
    subprocess.run([sys.executable,ROOT/'benchmarks/python/run_benchmarks.py','--label','zig-original-current-harness','--existing',raw],cwd=ROOT,check=True)
    path=dest/'zig-original-current-harness.summary.json';metadata=json.loads(path.read_text());metadata['kernel_git_sha']=ORIGINAL;metadata['harness_tree_source_hashes']=metadata.pop('source_hashes',{});metadata['harness_sha256']=hashlib.sha256((ROOT/'src/benchmark.zig').read_bytes()).hexdigest();path.write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
