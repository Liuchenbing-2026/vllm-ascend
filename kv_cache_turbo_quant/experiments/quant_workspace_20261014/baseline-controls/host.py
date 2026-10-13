from pathlib import Path
import subprocess,json,traceback
root=Path('/data02/tq-opt-20261011/quant_memory_controls_20261014')
code=1
try:
 devices=subprocess.check_output(['npu-smi','info'],text=True)
 assert 'No running processes found in NPU 2' in devices and 'No running processes found in NPU 5' in devices
 (root/'resources.log').write_text(devices)
 for mode in ['original','queued']:
  directory=root/mode;directory.mkdir()
  command=['docker','exec','-e','ASCEND_RT_VISIBLE_DEVICES=2,5','-e','VLLM_ASCEND_KVTQ_BITS=4','tq-opt-20261011-model','bash','-lc',
   'source /usr/local/Ascend/cann-9.1.0/set_env.sh && exec mssanitizer --tool=memcheck --kernel-name=KvCacheTurboQuant_0bce30e4690f2b43628265dfd3b6214f_0 --leak-check=yes --log-file=/ws/quant_memory_controls_20261014/'+mode+'/memcheck.log -- /ws/.venv/bin/python -u /ws/quant_memory_controls_20261014/quant_memory_control.py --variant '+mode]
  (directory/'command.json').write_text(json.dumps(command)+'\n')
  with (directory/'stdout.log').open('w') as out:result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,timeout=600)
  (directory/'execution.exit').write_text(str(result.returncode)+'\n');result.check_returncode()
 code=0
except Exception:
 (root/'failure.log').write_text(traceback.format_exc());raise
finally:(root/'host.exit').write_text(str(code)+'\n')
