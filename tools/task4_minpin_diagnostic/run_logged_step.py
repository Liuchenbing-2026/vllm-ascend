"""Run one explicitly supplied step and preserve argv, output, exit and first failure."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path


def run_step(directory,step,argv):
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    if not step or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in step):
        raise ValueError('Invalid step name')
    command_file=path/(step+'.command.json')
    log=path/(step+'.log')
    record=dict(step=step,argv=argv,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(log))
    with command_file.open('x') as handle:json.dump(record,handle,indent=2)
    with log.open('x') as handle:
        try:
            proc=subprocess.run(argv,stdout=handle,stderr=subprocess.STDOUT,check=False)
            rc=proc.returncode
        except OSError as error:
            handle.write(repr(error)+'\n');rc=127
    record.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (path/(step+'.exit.json')).open('x') as handle:json.dump(record,handle,indent=2)
    if rc:
        try:
            with (path/'FIRST_FAILURE.json').open('x') as handle:json.dump(record,handle,indent=2)
        except FileExistsError:pass
    return rc

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--step',required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args();argv=args.command[1:] if args.command[:1]==['--'] else args.command
    if not argv:p.error('Command required')
    raise SystemExit(run_step(args.directory,args.step,argv))
