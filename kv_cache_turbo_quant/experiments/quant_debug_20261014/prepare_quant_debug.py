"""Create an isolated symbol-enabled clone of the unchanged quantizer."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    target = root.parent.parent / "source/tq-quant-debug-20261014/kv_cache_turbo_quant"
    baseline = root.parent.parent / "source/tq-queued-read-20261014/kv_cache_turbo_quant"
    queued = root.parent.parent / "source/tq-queued-codec-20261014/kv_cache_turbo_quant"
    for section, name in [("op_host", "kv_cache_turbo_quant.cpp"), ("op_kernel", "kv_cache_turbo_quant.cpp"),
                          ("op_kernel", "kv_cache_turbo_quant_tiling.h")]:
        text = (baseline / "code/op" / section / name).read_text()
        text = text.replace("KvCacheTurboQuant", "TqQuantDebug").replace("kv_cache_turbo_quant", "tq_quant_debug")
        (target / "code/op" / section / name.replace("kv_cache_turbo_quant", "tq_quant_debug")).write_text(text)
    schema = json.loads((baseline / "code/op/KvCacheTurboQuant.json").read_text())
    schema[0]["op"] = "TqQuantDebug"
    (target / "code/op/TqQuantDebug.json").write_text(json.dumps(schema, indent=2) + "\n")
    text = (queued / "code/integration/torch_ext/tq_queued_quant.cpp").read_text()
    text = text.replace("aclnnKvCacheTurboQuant", "aclnnTqQuantDebug").replace("kv_cache_turbo_quant_queued", "kv_cache_turbo_quant_debug")
    text = text.replace("/usr/local/Ascend/cann-9.1.0/opp/vendors/customize", "/ws/quant_debug_20261014/opp/vendors/tqquantdebug")
    (target / "code/integration/torch_ext/tq_quant_debug.cpp").write_text(text)
    text = (queued / "code/integration/torch_ext/build_queued_quant.py").read_text().replace("tq_queued_quant", "tq_quant_debug")
    (target / "code/integration/torch_ext/build_quant_debug.py").write_text(text)
    text = (root / "build_quant_ws.sh").read_text()
    for old, new in [("quant_ws_20261014", "quant_debug_20261014"), ("TqQuantWs", "TqQuantDebug"),
                     ("tq_quant_ws", "tq_quant_debug"), ("tqquantws", "tqquantdebug"), ("build_quant_ws.py", "build_quant_debug.py")]:
        text = text.replace(old, new)
    needle = "cd gen\nbash build.sh"
    assert needle in text
    text = text.replace(needle, """cd gen
/ws/.venv/bin/python - <<'PY'
from pathlib import Path
p = Path('op_kernel/CMakeLists.txt')
s = p.read_text()
flag = 'npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g)'
if flag not in s:
    p.write_text(flag + '\\n' + s)
PY
bash build.sh""")
    (root / "build_quant_debug.sh").write_text(text)


if __name__ == "__main__":
    main()
