"""Build native runner and C ABI with pinned Mojo; vendor local runtime dependencies."""
from importlib.metadata import distribution
from pathlib import Path
import shutil
import subprocess
from quiet import run

ROOT=Path(__file__).resolve().parents[1]
COMPILER=ROOT/'.venv/bin/mojo'

def build():
    version=subprocess.check_output([COMPILER,'--version'],text=True).strip()
    if not version.startswith('Mojo 1.2.0 '):raise RuntimeError(version)
    output=ROOT/'build'
    output.mkdir(exist_ok=True)
    for name,opt in [('mojo-cases','-O3'),('mojo-cases-debug','-O0')]:
        run([COMPILER,'build',opt,'-D','ASSERT=all','-I',ROOT/'src/mojo',ROOT/'tests/mojo/case_runner.mojo','-o',output/name],label=name,cwd=ROOT)
    for name,opt,asserts in [('liblagomath_mojo.so','-O3','ASSERT=none'),('liblagomath_mojo_debug.so','-O0','ASSERT=all')]:
        run([COMPILER,'build','--emit','shared-lib',opt,'-D',asserts,'--target-cpu','x86-64','-I',ROOT/'src/mojo',ROOT/'src/mojo/cabi.mojo','-o',output/name],label=name,cwd=ROOT)
    runtime=Path(distribution('mojo-compiler').locate_file('modular/lib'))
    libs=output/'.libs';libs.mkdir(exist_ok=True)
    for name in ['libKGENCompilerRTShared.so','libMSupportGlobals.so','libAsyncRTRuntimeGlobals.so']:
        shutil.copy2(runtime/name,libs/name)
    # Existing compiler emits an environment-specific RPATH; make the artifact relocatable.
    patch=ROOT/'.venv/bin/patchelf'
    if patch.exists():
        for lib in libs.glob('*.so'):subprocess.run([patch,'--set-rpath','$ORIGIN',lib],check=True)
        for name in ['liblagomath_mojo.so','liblagomath_mojo_debug.so']:subprocess.run([patch,'--set-rpath','$ORIGIN/.libs',output/name],check=True)
    print(version)

if __name__=='__main__':build()
