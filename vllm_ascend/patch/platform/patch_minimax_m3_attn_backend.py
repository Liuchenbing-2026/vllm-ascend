import sys
import vllm_ascend.platform as _plat

_ORIG_GABC = _plat.NPUPlatform.get_attn_backend_cls.__func__
_installed = [False]


def _is_m3() -> bool:
    try:
        from vllm.config import get_current_vllm_config
        hf = get_current_vllm_config().model_config.hf_config
        mt = getattr(hf, "model_type", "") or ""
        archs = getattr(hf, "architectures", []) or []
        return mt in ("minimax_m3_vl", "minimax_m3") or any("MiniMaxM3" in a for a in archs)
    except Exception as e:
        return False


def _install_swap():
    if _installed[0]:
        return
    try:
        import vllm_ascend.attention.attention_v1 as _av1
        from vllm_ascend.attention.msa_v1 import AscendMSAImpl
        _orig_impl = _av1.AscendAttentionBackend.get_impl_cls

        def _gic():
            m3 = _is_m3()
            sys.stderr.write(f"[MSA-PATCH] get_impl_cls called, is_m3={m3}\n"); sys.stderr.flush()
            return AscendMSAImpl if m3 else _orig_impl()

        _av1.AscendAttentionBackend.get_impl_cls = staticmethod(_gic)
        _installed[0] = True
        sys.stderr.write("[MSA-PATCH] get_impl_cls swap INSTALLED\n"); sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[MSA-PATCH] install_swap FAILED: {type(e).__name__}: {e}\n"); sys.stderr.flush()


def _patched_gabc(cls, selected_backend, attn_selector_config, num_heads=None):
    sys.stderr.write(f"[MSA-PATCH] get_attn_backend_cls called, is_m3={_is_m3()}, use_mla={getattr(attn_selector_config,'use_mla',None)}\n"); sys.stderr.flush()
    _install_swap()
    return _ORIG_GABC(cls, selected_backend, attn_selector_config, num_heads)


_plat.NPUPlatform.get_attn_backend_cls = classmethod(_patched_gabc)
