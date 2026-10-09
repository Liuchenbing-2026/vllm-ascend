#!/usr/bin/env python3
"""Preserve completed cohorts and continue this task with sufficient HTTP deadlines."""
import argparse
import json
import os
import pathlib
import signal
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-transition', action='store_true')
    args = parser.parse_args()
    root = pathlib.Path('/ws')
    result = root / 'results/matrix_20261009T182045Z/bf16_in131072_out1024_c8.json'
    while not result.exists():
        if (root / 'matrix-attempt3.exit').exists():
            raise RuntimeError('Original matrix ended before the 8-concurrency cohort')
        time.sleep(10)
    data = json.loads(result.read_text())
    if not data['complete_cohort']:
        raise RuntimeError('The 8-concurrency cohort did not complete; inspect its evidence')
    if args.after_transition:
        if not (root / 'matrix-attempt3.exit').exists():
            raise RuntimeError('Original controller has not stopped; refusing another server')
        if not (root / 'logs/deadline-continuation.json').exists():
            raise RuntimeError('No recorded transition; inspect before restarting')
    targets = []
    for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
        try:
            process_args = path.read_bytes().split(b'\0')
        except OSError:
            continue
        if process_args[:2] == [b'/ws/.venv/bin/python', b'/ws/scripts/run_matrix.py'] and len(process_args) == 3:
            targets.append(int(path.parent.name))
    if not args.after_transition and len(targets) != 1:
        raise RuntimeError(f'Expected exactly one original task controller, found {targets}')
    if not args.after_transition:
        (root / 'logs/deadline-continuation.json').write_text(json.dumps({
        'preserved_full_cohorts': [1, 2, 4, 8], 'original_timeout_s': 1800,
        'continuation_timeout_s': 7200, 'controller_pid': targets[0],
        'reason': 'Measured C4 cohort took 466s; high concurrency may exceed 1800s',
        'model_and_serving_configuration_changed': False,
    }, indent=2) + '\n')
        os.kill(targets[0], signal.SIGTERM)
    for _ in range(18):
        if (root / 'matrix-attempt3.exit').exists():
            break
        time.sleep(5)
    else:
        raise RuntimeError('Task controller did not stop; refusing to start another server')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', 18377))
    for label, modes, concurrencies in [
        ('store4_lower_concurrency', ['store4'], ['1', '2', '4', '8']),
        ('bf16_high_concurrency', ['bf16'], ['16', '32']),
        ('store4_high_concurrency', ['store4'], ['16', '32']),
    ]:
        command = ['/ws/.venv/bin/python', '/ws/scripts/run_matrix.py',
                   '--modes', *modes, '--concurrencies', *concurrencies,
                   '--request-timeout', '7200', '--precision-log-dir',
                   '/ws/logs/matrix_20261009T182045Z']
        (root / f'logs/{label}.command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (root / f'logs/{label}.controller.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        (root / f'{label}.exit').write_text(str(completed.returncode) + '\n')
        if completed.returncode:
            raise RuntimeError(f'{label} controller failed; inspect the preserved logs')


if __name__ == '__main__':
    main()
