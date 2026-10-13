import pathlib,subprocess,time,json
root=pathlib.Path('/data02/tq-opt-20261011/read_tiles_20261014')
previous=pathlib.Path('/data02/tq-opt-20261011/decode_profiles_20261014')
while not (previous/'host.exit').exists():
 (root/'build-status.json').write_text(json.dumps({'stage':'waiting_for_profiles'})+'\n')
 time.sleep(20)
assert (previous/'host.exit').read_text().strip()=='0'
(root/'build-status.json').write_text(json.dumps({'stage':'native_build'})+'\n')
command=['docker','exec','tq-opt-20261011-model','bash','/ws/read_tiles_20261014/scripts/build_read_tiles.sh']
(root/'build.command.json').write_text(json.dumps(command)+'\n')
with (root/'logs/host-build.log').open('w') as output:
 result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
(root/'build.exit').write_text(str(result.returncode)+'\n')
(root/'build-status.json').write_text(json.dumps({'stage':'build_finished','exit':result.returncode})+'\n')
result.check_returncode()
