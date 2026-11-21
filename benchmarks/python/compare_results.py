"""Compare matching workloads, writing details without flooding the console."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('before',type=Path)
    parser.add_argument('after',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verbose',action='store_true',help='print every case comparison')
    args=parser.parse_args()
    before=json.loads(args.before.read_text());after=json.loads(args.after.read_text())
    if before['manifest_sha256']!=after['manifest_sha256']:parser.error('Cannot compare different workload manifests')
    if before['results'].keys()!=after['results'].keys():parser.error('Mismatched benchmark IDs')
    rows=[]
    for key,old in before['results'].items():
        new=after['results'][key]
        if old['median_ns']<=0:parser.error('Non-positive reference timing: '+key)
        rows.append(dict(id=key,ratio=new['median_ns']/old['median_ns'],before_ns=old['median_ns'],after_ns=new['median_ns']))
    output=args.output or args.after.with_suffix('.comparison.json')
    output.write_text(json.dumps(dict(before=str(args.before),after=str(args.after),cases=rows),indent=2)+'\n')
    print(f'{len(rows)} workloads compared; median time ratio {statistics.median(r["ratio"] for r in rows):.3f}x; details: {output}')
    if args.verbose:
        for row in rows:print(f'{row["id"]}: {row["ratio"]:.3f}x ({row["before_ns"]:.1f} -> {row["after_ns"]:.1f} ns)')


if __name__=='__main__':main()
