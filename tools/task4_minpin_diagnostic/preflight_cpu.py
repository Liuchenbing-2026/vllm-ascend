"""Read-only source audit; allows only the two inspected runtime artifacts by hash."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def audit(repo, expected):
    repo=Path(repo)
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True)
    status=git('status','--porcelain')
    result=dict(head=git('rev-parse','HEAD').strip(),status=status,artifacts={})
    print(json.dumps(result),flush=True)
    for args in [('diff','--exit-code','HEAD','--'),('diff','--cached','--exit-code')]:
        git(*args)
    untracked=git('ls-files','--others','--exclude-standard','-z').split('\0')
    for name in filter(None,untracked):
        if name not in expected:
            raise ValueError('Unexpected untracked path: '+name)
        path=repo/name
        if path.is_symlink() or path.stat().st_size>1024*1024:
            raise ValueError('Unexpected artifact kind/size: '+name)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=expected[name]:
            raise ValueError('Artifact content changed: '+name)
        result['artifacts'][name]=digest
    print(json.dumps(dict(result,passed=True)),flush=True)
    return result

if __name__=='__main__':
    audit(sys.argv[1],{
        'runtime/config/runtime_config.json.lock':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
        'runtime/config/runtime_config.json':'aedaf99be47cb6631c3a333cf1a13b71f619b795f1be0f5720a0b30e8d040f48',
    })
