"""Offline parser: run in an independent non-daemon process, after collection."""
import argparse
from pathlib import Path

from torch_npu.profiler.profiler import analyse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/ws/opt_20261011/profiles')
parser.add_argument('--modes', nargs='+', required=True)
args = parser.parse_args()
for mode in args.modes:
    paths = sorted((Path(args.root) / mode).glob('*_ascend_pt'))
    if not paths:
        raise RuntimeError(f'no raw worker profiles in {mode}')
    for path in paths:
        print('offline_parse', str(path), flush=True)
        analyse(str(path))
print('offline_parse_finished', flush=True)
