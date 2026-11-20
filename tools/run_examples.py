"""Run paired Mojo examples, keeping compiler/example output in build/logs."""
from pathlib import Path
from quiet import ROOT, run

def main():
    sources=sorted((ROOT/'examples/mojo').glob('*.mojo'))
    for source in sources:
        run([ROOT/'.venv/bin/mojo','-I',ROOT/'src/mojo',source],label=source.stem)
    print(f'{len(sources)} Mojo examples passed; logs: build/logs/')

if __name__=='__main__':main()
