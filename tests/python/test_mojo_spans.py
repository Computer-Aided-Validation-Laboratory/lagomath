"""Mechanical ownership/type checks; numerical truth remains in shared JSON."""
from pathlib import Path
import subprocess
from validate_cases import ROOT


def compile_source(tmp_path, body):
    source=tmp_path/'consumer.mojo'
    source.write_text('import lagomath as lm\nfrom std.collections import Array, Span\n'+body)
    return subprocess.run([ROOT/'.venv/bin/mojo','build','-I',ROOT/'src/mojo',source,'-o',tmp_path/'consumer'],capture_output=True,text=True)


def test_readonly_span_cannot_mutate(tmp_path):
    result=compile_source(tmp_path,'def main():\n    var data=Array[Float64,3](fill=0)\n    var view=lm.VecView(Span(data).as_imm())\n    view.set(0,1)\n')
    assert result.returncode!=0
    assert 'constraint' in result.stderr or 'mut' in result.stderr


def test_span_cannot_outlive_storage(tmp_path):
    result=compile_source(tmp_path,'def dangling() -> lm.VecView[DType.float64, MutUntrackedOrigin]:\n    var data=Array[Float64,3](fill=0)\n    return lm.VecView(Span(data))\ndef main():\n    var view=dangling()\n    print(view.get(0))\n')
    assert result.returncode!=0
    assert 'origin' in result.stderr or 'lifetime' in result.stderr


def test_pointer_backing_is_confined_to_abi():
    for folder in ['src/mojo','tests/mojo','examples/mojo','benchmarks/mojo','integration/mojo-consumer']:
        for path in (ROOT/folder).glob('*.mojo'):
            if path.name=='cabi.mojo':continue
            text=path.read_text()
            for token in ['Pointer[','unsafe_ptr','unsafe_mut_cast','unsafe_alloc','unsafe_free','unsafe_offset','.as_ref(']:
                assert token not in text,(path,token)


def test_solver_overlap_rejected_by_borrow_checker(tmp_path):
    result=compile_source(tmp_path,'def main() raises:\n    var data=Array[Float64,4](fill=0)\n    var matrix=lm.MatView(Span(data),2,2)\n    var perm=Array[Int,2](fill=0)\n    lm.linsolve.factorLUMatView(matrix,matrix,Span(perm))\n')
    assert result.returncode!=0
    assert 'aliasing' in result.stderr


def test_readonly_consumers_and_print_methods_compile(tmp_path):
    result=compile_source(tmp_path,'def main():\n    var data=Array[Float64,4](fill=1)\n    var readonly=Span(data).as_imm()\n    var vector=lm.VecView(readonly)\n    var matrix=lm.MatView(readonly,2,2)\n    print(lm.vecops.normL1VecView(vector),lm.vecops.normL2VecView(vector),lm.vecops.normL2SquareVecView(vector))\n    print(lm.sliceops.dot(readonly,readonly),matrix.trace())\n    vector.vecPrint()\n    matrix.matPrint()\n    lm.sliceops.slicePrint(readonly)\n    var fixed=lm.Vec[4,DType.float64]()\n    fixed.vecPrint()\n    var fixedMatrix=lm.Mat[2,2,DType.float64]()\n    fixedMatrix.matPrint()\n')
    assert result.returncode==0,result.stderr[-4000:]


def test_ndarray_metadata_does_not_own_storage(tmp_path):
    result=compile_source(tmp_path,'from std.benchmark import keep\ndef main():\n    var storage=List[Float64](length=6,fill=0)\n    var dims:List[Int]=[2,3]\n    var parent=lm.NDArray(Span(storage),dims)\n    var prefix:List[Int]=[1]\n    var child=parent.fixedPrefixView(prefix)\n    parent.deinit()\n    child.fill(1)\n    keep(child.data)\n    child.deinit()\n    keep(storage)\n    var map=List[Int]()\n    var array=lm.NDArray(Span(storage),dims)\n    var mapped=lm.MappedNDArray(array^,map^)\n    mapped.deinit()\n    keep(storage)\n')
    assert result.returncode==0,result.stderr[-4000:]
    subprocess.run([tmp_path/'consumer'],check=True,capture_output=True)
