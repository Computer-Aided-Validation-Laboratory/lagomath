"""Verify explicit SIMD against the existing independent numerical fixtures."""
import json
from pathlib import Path
import sys
from quiet import run
from simd_harness import ROOT, transform, supported_case


def prepare():
    dest=ROOT/'build/simd-cases'
    dest.mkdir(parents=True,exist_ok=True)
    cases=json.loads((ROOT/'tests/cases/all.json').read_text())
    cases['cases']=[c for c in cases['cases'] if supported_case(c)]
    (dest/'cases.json').write_text(json.dumps(cases))
    (dest/'data.zig').write_text('pub const data = @embedFile("cases.json");\n')
    zig=transform((ROOT/'src/case_runner.zig').read_text(),'zig')
    (dest/'cases.zig').write_text(zig)
    mojo=transform((ROOT/'tests/mojo/case_runner.mojo').read_text(),'mojo')
    mojo=mojo.replace('"tests/cases/all.json"','"build/simd-cases/cases.json"')
    (dest/'cases.mojo').write_text(mojo)
    return dest,cases['cases']


def main():
    dest,cases=prepare()
    for mode in ('Debug','ReleaseFast'):
        run(['zig','build','cases',f'-Doptimize={mode}','-Dcases-source=build/simd-cases/cases.zig','-Dcases-data=build/simd-cases/data.zig','--','--quiet'],label='simd-cases-zig-'+mode)
        print(f'{len(cases)} SIMD Zig cases passed ({mode})',flush=True)
    for mode in ('-O0','-O3'):
        binary=dest/('mojo'+mode)
        run([ROOT/'.venv/bin/mojo','build',mode,'-D','ASSERT=all','-I',ROOT/'src/mojo','-I',ROOT/'tests/mojo',dest/'cases.mojo','-o',binary],label='simd-cases-mojo-build'+mode)
        run([binary,'--quiet'],label='simd-cases-mojo'+mode)
        edges=dest/('mojo-edges'+mode)
        run([ROOT/'.venv/bin/mojo','build',mode,'-D','ASSERT=all','-I',ROOT/'src/mojo',ROOT/'tests/mojo/simd_edges.mojo','-o',edges],label='simd-edges-mojo-build'+mode)
        run([edges],label='simd-edges-mojo'+mode)
        print(f'{len(cases)} SIMD Mojo cases and edge checks passed ({mode})',flush=True)


if __name__=='__main__':main()
