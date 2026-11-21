"""Shared precondition contracts exercised in isolated native processes."""
import json
import subprocess
from pathlib import Path
import pytest
from validate_cases import ROOT

CONTRACTS=json.loads((ROOT/'tests/cases/contracts.json').read_text())['cases'] if (ROOT/'tests/cases/contracts.json').exists() else []

@pytest.mark.parametrize('case',CONTRACTS,ids=lambda c:c['id'])
@pytest.mark.parametrize('backend',['zig','mojo'])
def test_ndarray_preconditions(case,backend,tmp_path):
    dims=case['inputs']['shape']
    assert case['expected_error']=='AssertionFailed'
    exe=tmp_path/'contract'
    operation=case['operation']
    count=len(case['inputs']['a'])
    indices=case['inputs'].get('index',[])
    if backend=='zig':
        source=tmp_path/'contract.zig'
        literal='&.{'+','.join(map(str,dims))+'}'
        index_literal='&.{'+','.join(map(str,indices))+'}'
        setup=f'var data=[_]{"f64"}{{0}} ** {count};'
        if operation=='matrixInit':
            action=f'const array=lm.MatView(f64).init(&data,{dims[0]},{dims[1]+1});std.mem.doNotOptimizeAway(array);'
        else:
            action='var array=try lm.NDArray(f64).init(std.heap.page_allocator,&data,'+literal+');defer array.deinit(std.heap.page_allocator);'
            if operation=='get':action+='std.mem.doNotOptimizeAway(array.get('+index_literal+'));'
            elif operation=='offset3':action+='std.mem.doNotOptimizeAway(array.offset3('+','.join(map(str,indices))+'));'
        source.write_text('const std=@import("std");const lm=@import("lagomath");pub fn main() !void {'+setup+action+'}')
        subprocess.run(['zig','build-exe','-ODebug','--dep','lagomath',f'-Mroot={source}',f'-Mlagomath={ROOT}/src/root.zig',f'-femit-bin={exe}'],check=True,capture_output=True)
    else:
        source=tmp_path/'contract.mojo'
        setup='import lagomath as lm\nfrom std.collections import Array, Span\ndef main():\n'+f'    var data=Array[Float64,{count}](fill=0)\n    var dims=List[Int]()\n'+''.join(f'    dims.append({n})\n' for n in dims)
        if operation=='matrixInit':action=f'    var array=lm.MatView(Span(data),{dims[0]},{dims[1]+1})\n    print(array.rows_num)\n'
        else:
            action='    var array=lm.NDArray(Span(data),dims)\n'
            if operation=='get':
                action+='    var idx=List[Int]()\n'+''.join(f'    idx.append({n})\n' for n in indices)+'    print(array.get(idx))\n'
            elif operation=='offset3':action+='    print(array.offset3('+','.join(map(str,indices))+'))\n'
            else:action+='    print(array.length)\n'
        source.write_text(setup+action)
        subprocess.run([ROOT/'.venv/bin/mojo','build','-O0','-D','ASSERT=all','--debug-level','full','-I',ROOT/'src/mojo',source,'-o',exe],check=True,capture_output=True)
    result=subprocess.run([exe],capture_output=True,text=True)
    assert result.returncode!=0,(case['id'],result.stdout)
    assert any(s in (result.stdout+result.stderr).lower() for s in ['assert','unreachable']),result.stderr
