#!/usr/bin/env python3
"""Push this task's explicit files or run a safely quoted host command."""
import argparse
import io
from pathlib import Path
import shlex
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['push', 'run'])
    parser.add_argument('--command')
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           '-o', 'ServerAliveInterval=15', '-o', 'ControlPath=/tmp/tq12-%h-%p-%r.sock', 'root@112.29.145.3']
    if args.action == 'push':
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode='w:gz') as archive:
            for name in ['scripts', 'candidates', 'screen2', 'screen3', 'stage.json', 'issues.json']:
                archive.add(root / name, arcname=name)
        directory = '/data02/tq-opt-20261011/opt_20261012'
        command = 'mkdir -p ' + shlex.quote(directory) + ' && tar -xzf - -C ' + shlex.quote(directory)
        result = subprocess.run(ssh + [command], input=payload.getvalue(), capture_output=True)
    else:
        result = subprocess.run(ssh + [args.command], capture_output=True)
    output = result.stdout + result.stderr
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_bytes(output)
    print(output.decode(errors='replace'))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
