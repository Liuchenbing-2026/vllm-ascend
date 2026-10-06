"""vLLM general plugin: installs the KvCacheTurboQuant hooks in every
vLLM process (main / EngineCore / worker) via the vllm.general_plugins
entry point.

Modes (mutually exclusive):
  VLLM_ASCEND_KVTQ=1        shadow mode (quantize + discard, cache BF16)
  VLLM_ASCEND_KVTQ_STORE=1  store mode (compressed real-storage KV cache)
"""
import os
import sys

_KVTQ_DIR = "/root/kvtq_integration"
for _p in (_KVTQ_DIR, os.path.join(_KVTQ_DIR, "torch_ext")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def register():
    print(f"[KVTQ] plugin register() called pid={os.getpid()} "
          f"shadow={os.environ.get('VLLM_ASCEND_KVTQ')} "
          f"store={os.environ.get('VLLM_ASCEND_KVTQ_STORE')}", flush=True)
    if os.environ.get("VLLM_ASCEND_KVTQ_STORE", "0") == "1":
        import build_ext
        build_ext.build()
        import kvtq_store
        kvtq_store.install()
        return
    if os.environ.get("VLLM_ASCEND_KVTQ", "0") != "1":
        return
    import build_ext
    build_ext.build()
    import kvtq_shadow
    kvtq_shadow.install()