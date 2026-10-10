#!/usr/bin/env python3
"""Launch the follow-up inside the identified runtime without tying it to SSH."""
import json
import subprocess


def main():
    name = 'tq-128k-ab-20261010-runtime'
    container = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert container['Id'].startswith('0354eaaf')
    assert container['State']['Running']
    processes = subprocess.check_output(['docker', 'top', name, '-eo', 'pid,args'], text=True)
    assert '/run_short.py' not in processes
    assert 'VLLM::' not in processes
    root = '/ws/short_ab_20261010'
    command = (
        'source /usr/local/Ascend/cann-9.1.0/set_env.sh\n'
        f'/ws/.venv/bin/python {root}/scripts/run_short.py > {root}/controller.log 2>&1\n'
        'task_rc=$?\n'
        f'printf "%s\\n" "$task_rc" > {root}/controller.exit\n'
        'exit "$task_rc"\n'
    )
    subprocess.run(['docker', 'exec', '-d', name, 'bash', '-c', command], check=True)
    print(json.dumps({'container': name, 'id': container['Id'],
                      'launch_command': command, 'root': root}), flush=True)


if __name__ == '__main__':
    main()
