"""Transfer bounded, hash-verified diagnostic evidence via private SSH setup."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['byte','byte2','micro'],required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    name={'byte':'byte_read_20261015','byte2':'byte_read_attempt2_20261015','micro':'byte_read_micro_20261015'}[args.kind]
    spec = importlib.util.spec_from_file_location('a', '/Users/liucb/.codex/memory/private/task18_remote_access.py')
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    export = '/data2/tq-profile-20261014/evidence-' + name + '-' + str(time.time_ns()) + '.tar.gz'
    code = '''from pathlib import Path
import tarfile,json,hashlib,os
root=Path('/data2/tq-profile-20261014/NAME')
target=Path(EXPORT)
with tarfile.open(target,'w:gz',compresslevel=1) as t:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or '/PROF_' in str(p) or 'op-build' in p.parts:continue
  if p.suffix in ['.db','.so','.bin','.o','.pyc','.a'] or p.stat().st_size>250000000:continue
  t.add(p,arcname=str(p.relative_to(root)),recursive=False)
os.chmod(str(target),0o644)
print(json.dumps({'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}))
'''.replace('NAME', name).replace('EXPORT', repr(export))
    r = a.run('18', code, 300)
    (root / (name + '-export.log')).write_text(r.stdout + r.stderr)
    r.check_returncode()
    metadata = json.loads(r.stdout)
    auth = json.loads(Path('/Users/liucb/.codex/memory/private/task18-access.json').read_text())
    env = dict(os.environ, SSH_ASKPASS='/Users/liucb/.codex/memory/private/task18-ssh-askpass.py', SSH_ASKPASS_REQUIRE='force', DISPLAY='task18:0')
    archive = root / (name + '-snapshot.tar.gz')
    r = subprocess.run(['scp', '-o', 'ConnectTimeout=15', '-o', 'NumberOfPasswordPrompts=1',
                        auth['username'] + '@' + auth['hosts']['18'] + ':' + export, str(archive)], env=env, capture_output=True)
    (root / (name + '-scp.log')).write_bytes(r.stdout + r.stderr)
    r.check_returncode()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata['sha256']
    destination = root / (name + '-snapshot')
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            p = Path(member.name)
            assert member.isfile() and not p.is_absolute() and '..' not in p.parts, member.name
            out = destination / p
            out.parent.mkdir(parents=True, exist_ok=True)
            if p.name == 'trace_view.json':
                with t.extractfile(member) as src, gzip.open(out.with_suffix('.json.gz'), 'wb') as dst:
                    for chunk in iter(lambda: src.read(1048576), b''):
                        dst.write(chunk)
            else:
                out.write_bytes(t.extractfile(member).read())
    print(json.dumps({'name': name, **metadata, 'members': len(list(destination.rglob('*')))}))


if __name__ == '__main__':
    main()
