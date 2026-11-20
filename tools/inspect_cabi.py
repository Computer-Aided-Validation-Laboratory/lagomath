"""Audit ABI symbols and record native runtime dependencies on Linux."""
import json
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[1]
required=set(re.findall(r'\b(lm_\w+)\(', (ROOT/'include/lagomath.h').read_text()))
report={}
for name,path in {'zig':ROOT/'zig-out/lib/liblagomath_zig.so','zig_release':ROOT/'build/cabi-release/lib/liblagomath_zig.so','mojo':ROOT/'build/liblagomath_mojo.so','mojo_debug':ROOT/'build/liblagomath_mojo_debug.so'}.items():
    symbols=subprocess.check_output(['nm','-D','--defined-only',path],text=True)
    names={line.split()[-1] for line in symbols.splitlines() if line.split()}
    if not required<=names:raise RuntimeError(f'{name}: missing exports {required-names}')
    dynamic=subprocess.check_output(['readelf','-d',path],text=True)
    report[name]={'path':str(path),'symbols':sorted(required),'dynamic_section':dynamic}
(ROOT/'build/abi-inspection.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'{len(required)} required ABI symbols verified in four libraries')
