"""Isolate the Matmul workspace hypothesis; preserve every quantization calculation."""

from pathlib import Path
import json
import shutil

root = Path(__file__).resolve().parent
source = root.parent.parent / "source/tq-quant-workspace-20261014/kv_cache_turbo_quant"
baseline = root.parent.parent / "source/tq-queued-read-20261014/kv_cache_turbo_quant"
queued = root.parent.parent / "source/tq-queued-codec-20261014/kv_cache_turbo_quant"
op = source / "code/op"
for section, name in [("op_host", "kv_cache_turbo_quant.cpp"), ("op_kernel", "kv_cache_turbo_quant.cpp"), ("op_kernel", "kv_cache_turbo_quant_tiling.h")]:
    text = (baseline / "code/op" / section / name).read_text()
    text = text.replace("KvCacheTurboQuant", "TqQuantWs").replace("kv_cache_turbo_quant", "tq_quant_ws")
    if section == "op_kernel" and name.endswith(".cpp"):
        needle = "REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(),"
        assert text.count(needle) == 1
        text = text.replace(needle, "REGIST_MATMUL_OBJ(&pipe, workspace,")
    (op / section / name.replace("kv_cache_turbo_quant", "tq_quant_ws")).write_text(text)
schema = json.loads((baseline / "code/op/KvCacheTurboQuant.json").read_text())
schema[0]["op"] = "TqQuantWs"
(op / "TqQuantWs.json").write_text(json.dumps(schema, indent=2) + "\n")
integration = source / "code/integration"
shutil.copy2(queued / "code/integration/kvtq_store.py", integration / "kvtq_store.py")
wrapper = (queued / "code/integration/kvtq_queued_quant.py").read_text()
wrapper = wrapper.replace("queued_codec_20261014", "quant_ws_20261014").replace("tq_queued_quant", "tq_quant_ws").replace("kv_cache_turbo_quant_queued", "kv_cache_turbo_quant_workspace")
(integration / "kvtq_queued_quant.py").write_text(wrapper)
extension = integration / "torch_ext"
glue = (queued / "code/integration/torch_ext/tq_queued_quant.cpp").read_text()
glue = glue.replace("aclnnKvCacheTurboQuant", "aclnnTqQuantWs")
glue = glue.replace("kv_cache_turbo_quant_queued", "kv_cache_turbo_quant_workspace")
glue = glue.replace("/usr/local/Ascend/cann-9.1.0/opp/vendors/customize", "/ws/quant_ws_20261014/opp/vendors/tqquantws")
(extension / "tq_quant_ws.cpp").write_text(glue)
builder = (queued / "code/integration/torch_ext/build_queued_quant.py").read_text().replace("tq_queued_quant", "tq_quant_ws")
(extension / "build_quant_ws.py").write_text(builder)
build = (root / "build_read_tiles.sh").read_text()
for old, new in [("read_tiles_20261014", "quant_ws_20261014"), ("TqPagedReadV2", "TqQuantWs"), ("tq_paged_read_v2", "tq_quant_ws"), ("tqreadv2", "tqquantws"), ("build_read_tiles.py", "build_quant_ws.py")]:
    build = build.replace(old, new)
(root / "build_quant_ws.sh").write_text(build)
print("Prepared isolated one-line system-workspace candidate with identical codec math")
