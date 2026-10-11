#!/usr/bin/env python3
"""Download explicit task evidence over SSH without using the failing SFTP path."""
import argparse
import hashlib
import io
import json
import os
import pathlib
import shlex
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='+', required=True)
    parser.add_argument('--destination', required=True, type=pathlib.Path)
    args = parser.parse_args()
    for name in args.path:
        path = pathlib.PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Invalid evidence path: {name}')
    # Each file is read once before its TAR header is emitted. An active log is
    # therefore an explicit snapshot, without a TAR "file changed" truncation.
    source = '''import datetime, hashlib, io, json, pathlib, sys, tarfile
root = pathlib.Path('/data02/tq-opt-20261011/opt_20261012').resolve()
records = []
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for name in json.loads(sys.argv[1]):
        path = (root / name).resolve()
        if root not in path.parents and path != root:
            raise ValueError(name)
        paths = sorted(path.rglob('*')) if path.is_dir() else [path]
        for item in paths:
            if item.is_symlink() or not item.is_file():
                continue
            relative = str(item.relative_to(root))
            payload = item.read_bytes()
            member = tarfile.TarInfo(relative)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
            records.append({'path': relative, 'bytes': len(payload),
                            'sha256': hashlib.sha256(payload).hexdigest()})
    payload = (json.dumps({'snapshot_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'files': records}, indent=2) + '\\n').encode()
    member = tarfile.TarInfo('snapshot-manifest.json')
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))
'''
    command = 'python3 -I -c ' + shlex.quote(source) + ' ' + shlex.quote(json.dumps(args.path))
    env = dict(os.environ)
    completed = subprocess.run([
        'ssh', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12',
        '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
        '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
        '-o', 'ControlPath=/tmp/tq12-pull-%h-%p-%r.sock',
        'root@112.29.145.3', command], capture_output=True, env=env)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors='replace'))
    files = {}
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode='r:gz') as archive:
        for member in archive:
            path = pathlib.PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Invalid downloaded member: {member.name}')
            files[member.name] = archive.extractfile(member).read()
    manifest = json.loads(files['snapshot-manifest.json'])
    for item in manifest['files']:
        payload = files[item['path']]
        assert len(payload) == item['bytes'], item['path']
        assert hashlib.sha256(payload).hexdigest() == item['sha256'], item['path']
    for name, payload in files.items():
        path = args.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    print(json.dumps({'snapshot_utc': manifest['snapshot_utc'], 'verified_files': len(manifest['files'])}))


if __name__ == '__main__':
    main()
