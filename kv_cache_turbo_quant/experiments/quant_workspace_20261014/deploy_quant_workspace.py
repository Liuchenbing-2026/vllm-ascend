"""Derive serial gates for an isolated system-workspace candidate."""

from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
host = (root / "host_queued_codec.py").read_text()
host = host.replace("queued_codec_model_20261014", "quant_ws_model_20261014").replace("queued_codec_20261014", "quant_ws_20261014")
host = host.replace("tq_queued_quant", "tq_quant_ws").replace("build_queued_quant.py", "build_quant_ws.py")
host = host.replace('QUANT_KERNEL = "KvCacheTurboQuant_0bce30e4690f2b43628265dfd3b6214f_0"\n', '')
start = host.index('    command("build",')
stop = host.index('    common = ', start)
host = host[:start] + '''    tile_gate = ROOT.parent / "read_tiles_20261014/memory-attempt1/execution.exit"
    while not tile_gate.exists():
        status("waiting_tiled_reader_memory")
        time.sleep(10)
    assert tile_gate.read_text().strip() == "0"
    devices = subprocess.check_output(["npu-smi", "info"], text=True)
    assert "No running processes found in NPU 2" in devices and "No running processes found in NPU 5" in devices
    command("build", "exec bash /ws/quant_ws_20261014/scripts/build_quant_ws.sh")
    capture = "from pathlib import Path;import hashlib,json,re;root=Path('/ws/quant_ws_20261014');files=list((root/'code').rglob('*'))+list((root/'opp').rglob('*'));files.append(root/'torch-extensions/tq_quant_ws/tq_quant_ws.so');names=set();\\nfor p in (root/'opp').rglob('*.o'):\\n names.update(x.decode() for x in re.findall(rb'TqQuantWs_[0-9a-f]+_[0-9]+',p.read_bytes()))\\nprint(json.dumps({'files':{str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files if p.is_file() and '__pycache__' not in p.parts},'kernel_names':sorted(names)}))"
    build = json.loads(subprocess.check_output(["docker", "exec", CONTAINER, "/ws/.venv/bin/python", "-I", "-c", capture]))
    (ROOT / "build-manifest.json").write_text(json.dumps(build["files"], indent=2) + "\\n")
    assert len(build["kernel_names"]) == 1, build["kernel_names"]
    kernel = build["kernel_names"][0]
    (ROOT / "kernel-name.json").write_text(json.dumps({"kernel": kernel}) + "\\n")
''' + host[stop:]
host = host.replace('common = "export PYTHONPATH=', 'common = "source /ws/quant_ws_20261014/opp/vendors/tqquantws/bin/set_env.bash && export PYTHONPATH=')
host = host.replace('command("paired-gate", "export PYTHONPATH=', 'command("paired-gate", "source /ws/quant_ws_20261014/opp/vendors/tqquantws/bin/set_env.bash && export PYTHONPATH=')
host = host.replace("QUANT_KERNEL", "kernel").replace("verify_queued_quant.py", "verify_quant_ws.py")
host = host.replace('kernel, 54)', 'kernel, 35)').replace('checked_quant_invocations=54', 'checked_quant_invocations=35')
start = host.index('    tiles = ROOT.parent')
stop = host.index('    (ROOT / "gates.exit").write_text', start)
host = host[:start] + host[stop:]
(root / "host_quant_ws.py").write_text(host)
verify = (root / "verify_queued_quant.py").read_text().replace("queued_codec_20261014", "quant_ws_20261014")
(root / "verify_quant_ws.py").write_text(verify)
deploy = (root / "deploy_queued_codec.py").read_text()
for before, after in [("queued-codec-model-runtime", "quant-ws-model-runtime"),
                      ("queued_codec_model_20261014", "quant_ws_model_20261014"),
                      ("queued_codec_20261014", "quant_ws_20261014"),
                      ("tq-queued-codec-20261014", "tq-quant-workspace-20261014"),
                      ("queued-codec-gate-input", "quant-ws-gate-input"),
                      ("verify_queued_quant.py", "verify_quant_ws.py"),
                      ("host_queued_codec.py", "host_quant_ws.py"),
                      ("queued-codec-gate-deploy.log", "quant-ws-gate-deploy.log"),
                      ("queued-codec-model-deploy.log", "quant-ws-model-deploy.log")]:
    deploy = deploy.replace(before, after)
deploy = deploy.replace("== 54", "== 35")
deploy = deploy.replace('(codec / "source-manifest.json").write_text', '''shutil.copytree(source.parent / "op", codec / "code/op", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy2(root / "build_quant_ws.sh", codec / "scripts/build_quant_ws.sh")
(codec / "source-manifest.json").write_text''')
deploy = deploy.replace('serve.write_text(text)', '''text = text.replace("source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash",
                    "source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash\\nsource /ws/quant_ws_20261014/opp/vendors/tqquantws/bin/set_env.bash")
serve.write_text(text)''')
(root / "deploy_quant_ws_generated.py").write_text(deploy)
subprocess.run(["python3", "-m", "py_compile", str(root / "host_quant_ws.py"), str(root / "verify_quant_ws.py"), str(root / "deploy_quant_ws_generated.py")], check=True)
subprocess.run(["bash", "-n", str(root / "build_quant_ws.sh")], check=True)
subprocess.run(["python3", str(root / "deploy_quant_ws_generated.py")], check=True)
