"""Use the installed exported PTA execution macro, preserving the device kernel."""

from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
source = root.parent.parent / "source/tq-standard-pta-20261014/kv_cache_turbo_quant/code/integration"
parent = root.parent.parent / "source/tq-queued-codec-20261014/kv_cache_turbo_quant/code/integration"
for name in ["kvtq_store.py", "kvtq_queued_quant.py"]:
    shutil.copy2(parent / name, source / name)
wrapper = source / "kvtq_queued_quant.py"
wrapper.write_text(wrapper.read_text().replace("queued_codec_20261014", "standard_pta_20261014").replace("tq_queued_quant", "tq_standard_quant").replace("kv_cache_turbo_quant_queued", "kv_cache_turbo_quant_standard"))
text = (parent / "torch_ext/tq_queued_quant.cpp").read_text()
text = text.replace('#include "torch_npu/csrc/framework/OpCommand.h"', '#include "torch_npu/csrc/framework/OpCommand.h"\n#include "op_plugin/utils/op_api_common_base.h"')
text = text.replace('    const OpApi &api = GetOpApi();\n', '')
start, stop = text.index('    aclTensor *kvTensor ='), text.index('    return std::make_tuple')
text = text[:start] + '''    EXEC_NPU_CMD_V1_EXT(aclnnKvCacheTurboQuant, kv, rotation, qjl, mseBits,
        quantIdx, quantQjl, quantNorm, quantGamma);
''' + text[stop:]
text = text.replace("kv_cache_turbo_quant_queued", "kv_cache_turbo_quant_standard")
(source / "torch_ext/tq_standard_quant.cpp").write_text(text)
builder = (parent / "torch_ext/build_queued_quant.py").read_text().replace("tq_queued_quant", "tq_standard_quant")
builder = builder.replace('str(package / "include"),', 'str(package / "include"), str(package / "include/third_party/op-plugin"),')
(source / "torch_ext/build_standard_quant.py").write_text(builder)
host = (root / "host_queued_codec.py").read_text().replace("queued_codec_model_20261014", "standard_pta_model_20261014").replace("queued_codec_20261014", "standard_pta_20261014")
host = host.replace("tq_queued_quant", "tq_standard_quant").replace("build_queued_quant.py", "build_standard_quant.py").replace("verify_queued_quant.py", "verify_standard_quant.py")
start, stop = host.index('    tiles = ROOT.parent'), host.index('    (ROOT / "gates.exit").write_text')
host = host[:start] + host[stop:]
host = host.replace('    common = "export PYTHONPATH=', '    common = "source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash && export PYTHONPATH=')
host = host.replace('command("paired-gate", "export PYTHONPATH=', 'command("paired-gate", "source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash && export PYTHONPATH=')
(root / "host_standard_pta.py").write_text(host)
(root / "verify_standard_quant.py").write_text((root / "verify_queued_quant.py").read_text().replace("queued_codec_20261014", "standard_pta_20261014"))
deploy = (root / "deploy_queued_codec.py").read_text()
for before, after in [("queued-codec-model-runtime", "standard-pta-model-runtime"), ("queued_codec_model_20261014", "standard_pta_model_20261014"),
                      ("queued_codec_20261014", "standard_pta_20261014"), ("tq-queued-codec-20261014", "tq-standard-pta-20261014"),
                      ("queued-codec-gate-input", "standard-pta-gate-input"), ("verify_queued_quant.py", "verify_standard_quant.py"),
                      ("host_queued_codec.py", "host_standard_pta.py"), ("queued-codec-gate-deploy.log", "standard-pta-gate-deploy.log"),
                      ("queued-codec-model-deploy.log", "standard-pta-model-deploy.log")]:
    deploy = deploy.replace(before, after)
deploy = deploy.replace('serve.write_text(text)', '''text = text.replace("source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash",
                    "source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash\\nsource /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash")
serve.write_text(text)''')
(root / "deploy_standard_pta_generated.py").write_text(deploy)
subprocess.run(["python3", "-m", "py_compile", str(root / "host_standard_pta.py"), str(root / "verify_standard_quant.py"), str(root / "deploy_standard_pta_generated.py")], check=True)
subprocess.run(["python3", str(root / "deploy_standard_pta_generated.py")], check=True)
