"""Exercise the actual Zig package whitelist from a separate, tool-free consumer."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from quiet import run
ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=re.findall(r'"([^"]+)"',(ROOT/'build.zig.zon').read_text().split('.paths = .{',1)[1])
    with tempfile.TemporaryDirectory(prefix='lagomath-consumer-') as directory:
        staging=Path(directory);package=staging/'package';package.mkdir()
        for name in paths:
            source=ROOT/name;destination=package/name
            if source.is_dir():shutil.copytree(source,destination)
            else:shutil.copy2(source,destination)
        consumer=staging/'consumer';shutil.copytree(ROOT/'integration/consumer',consumer,ignore=shutil.ignore_patterns('.zig-cache','zig-out'))
        zon=consumer/'build.zig.zon';zon.write_text(zon.read_text().replace('.path = "../.."','.path = "../package"'))
        for mode in ['Debug','ReleaseFast']:
            run(['zig','build','run',f'-Doptimize={mode}'],cwd=consumer,label='source-consumer-'+mode)
    print('Whitelisted source-only consumer passed in Debug and ReleaseFast')

if __name__=='__main__':main()
