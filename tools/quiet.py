"""Capture tool diagnostics; show only a bounded tail on failure."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def run(command, *, label, cwd=ROOT, env=None):
    logs=ROOT/'build/logs'
    logs.mkdir(parents=True,exist_ok=True)
    log=logs/(label+'.log')
    result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True)
    log.write_text(result.stdout+result.stderr)
    if result.returncode:
        lines=(result.stderr or result.stdout).splitlines()
        positions=[i for i,line in enumerate(lines) if 'error:' in line and 'following build command' not in line]
        selected=[]
        for i in positions[:6]:
            for line in lines[i:i+3]:
                if line not in selected:selected.append(line)
        tail='\n'.join(selected[:24] if selected else lines[-24:])
        raise RuntimeError(f'{label} failed ({result.returncode}); full log: {log}\n{tail}')
    return result
