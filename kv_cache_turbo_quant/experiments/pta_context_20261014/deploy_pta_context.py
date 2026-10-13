"""Restore PTA stream-resource/workspace setup around the unchanged native API."""

from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
source = root.parent.parent / "source/tq-pta-context-20261014/kv_cache_turbo_quant/code/integration"
parent = root.parent.parent / "source/tq-queued-codec-20261014/kv_cache_turbo_quant/code/integration"
for name in ["kvtq_store.py", "kvtq_queued_quant.py"]:
    shutil.copy2(parent / name, source / name)
wrapper = source / "kvtq_queued_quant.py"
wrapper.write_text(wrapper.read_text().replace("queued_codec_20261014", "pta_context_20261014").replace("tq_queued_quant", "tq_pta_quant").replace("kv_cache_turbo_quant_queued", "kv_cache_turbo_quant_pta"))
text = (parent / "torch_ext/tq_queued_quant.cpp").read_text()
text = text.replace('#include "torch_npu/csrc/framework/OpCommand.h"', '#include "torch_npu/csrc/framework/OpCommand.h"\n#include "torch_npu/csrc/framework/utils/OpPreparation.h"\n#include "torch_npu/csrc/core/npu/NPUFunctions.h"')
text = text.replace('    uint64_t workspaceSize = 0;', '''    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    if (c10_npu::check_enqueue_need_use(stream)) {
        auto resourceStatus = c10_npu::UseStreamResInCurrentThread(stream);
        TORCH_CHECK(resourceStatus == ACL_SUCCESS, "PTA enqueue stream-resource setup failed");
    }
    uint64_t workspaceSize = 0;''')
text = text.replace('workspace = at::empty({static_cast<int64_t>(workspaceSize)}, u8Options);', 'workspace = at_npu::native::OpPreparation::unsafe_empty_workspace(workspaceSize);')
needle = '    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);'
position = text.rfind(needle)
assert text.count(needle) == 2
text = text[:position] + text[position:].replace(needle + '\n', '', 1)
text = text.replace('        auto status = run(', '''        if (c10_npu::check_dequeue_need_use(stream)) {
            auto resourceStatus = c10_npu::UseStreamResInCurrentThread(stream);
            TORCH_CHECK(resourceStatus == ACL_SUCCESS, "PTA dequeue stream-resource setup failed");
        }
        auto status = run(''')
text = text.replace("kv_cache_turbo_quant_queued", "kv_cache_turbo_quant_pta")
(source / "torch_ext/tq_pta_quant.cpp").write_text(text)
(source / "torch_ext/build_pta_quant.py").write_text((parent / "torch_ext/build_queued_quant.py").read_text().replace("tq_queued_quant", "tq_pta_quant"))
host = (root / "host_queued_codec.py").read_text()
host = host.replace("queued_codec_model_20261014", "pta_context_model_20261014").replace("queued_codec_20261014", "pta_context_20261014")
host = host.replace("tq_queued_quant", "tq_pta_quant").replace("build_queued_quant.py", "build_pta_quant.py").replace("verify_queued_quant.py", "verify_pta_quant.py")
start, stop = host.index('    tiles = ROOT.parent'), host.index('    (ROOT / "gates.exit").write_text')
host = host[:start] + host[stop:]
(root / "host_pta_context.py").write_text(host)
(root / "verify_pta_quant.py").write_text((root / "verify_queued_quant.py").read_text().replace("queued_codec_20261014", "pta_context_20261014"))
deploy = (root / "deploy_queued_codec.py").read_text()
for before, after in [("queued-codec-model-runtime", "pta-context-model-runtime"), ("queued_codec_model_20261014", "pta_context_model_20261014"),
                      ("queued_codec_20261014", "pta_context_20261014"), ("tq-queued-codec-20261014", "tq-pta-context-20261014"),
                      ("queued-codec-gate-input", "pta-context-gate-input"), ("verify_queued_quant.py", "verify_pta_quant.py"),
                      ("host_queued_codec.py", "host_pta_context.py"), ("queued-codec-gate-deploy.log", "pta-context-gate-deploy.log"),
                      ("queued-codec-model-deploy.log", "pta-context-model-deploy.log")]:
    deploy = deploy.replace(before, after)
(root / "deploy_pta_context_generated.py").write_text(deploy)
subprocess.run(["python3", "-m", "py_compile", str(root / "host_pta_context.py"), str(root / "verify_pta_quant.py"), str(root / "deploy_pta_context_generated.py")], check=True)
subprocess.run(["python3", str(root / "deploy_pta_context_generated.py")], check=True)
