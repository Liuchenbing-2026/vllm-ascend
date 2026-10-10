#!/usr/bin/env python3
"""Start only the preserved task runtime after checking its physical cards."""
import json
import subprocess


def main():
    name = 'tq-128k-ab-20261010-runtime'
    container = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert container['Id'].startswith('0354eaaf')
    assert not container['State']['Running']
    assert container['Image'] == 'sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14'
    command = ' '.join(container['Config']['Cmd'] or [])
    assert 'sleep' in command, command
    info = subprocess.check_output(['npu-smi', 'info'], text=True)
    assert 'No running processes found in NPU 0' in info
    assert 'No running processes found in NPU 1' in info
    print(info, flush=True)
    subprocess.run(['docker', 'start', name], check=True)
    probe = "import pathlib,sys; p=pathlib.Path('/ws/.venv/bin/python'); print({'venv_python_exists':p.exists(),'python':sys.version}); assert p.exists()"
    subprocess.run(['docker', 'exec', name, 'python3', '-c', probe], check=True)
    subprocess.run(['docker', 'exec', name, 'fuser', '/dev/davinci0', '/dev/davinci1'], check=False)


if __name__ == '__main__':
    main()
