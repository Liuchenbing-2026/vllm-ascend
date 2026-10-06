"""Shadow-mode integration of KvCacheTurboQuant into vllm-ascend.

Import and call install() BEFORE creating the vLLM engine. vLLM V1 forks the
EngineCore worker process, so the monkeypatch installed in the parent process
propagates to the worker.

Shadow mode: every K/V written to the paged KV cache is additionally quantized
by the custom aclnn KvCacheTurboQuant op. The quantized outputs are discarded
(or, with VLLM_ASCEND_KVTQ_DEBUG=1, used to verify the reconstruction error).
The paged KV cache itself stays BF16, so model outputs are bit-identical to
the baseline; this validates the op end-to-end inside a real serving engine
with zero correctness risk.

Env:
  VLLM_ASCEND_KVTQ=1        enable shadow quantization (default 1)
  VLLM_ASCEND_KVTQ_BITS=3   mse_bits 2/3/4 (default 3)
  VLLM_ASCEND_KVTQ_DEBUG=0  1 -> compute dequant reconstruction error per call
"""
import math
import os

import torch

_ENABLED = os.environ.get("VLLM_ASCEND_KVTQ", "0") == "1"
_DEBUG = os.environ.get("VLLM_ASCEND_KVTQ_DEBUG", "0") == "1"
_MSE_BITS = int(os.environ.get("VLLM_ASCEND_KVTQ_BITS", "3"))
_HEAD_DIM = 128

CENTROIDS = {
    2: [-0.1335033178, -0.04002048075, 0.04002048075, 0.1335033178],
    3: [-0.19020693, -0.1187859178, -0.06682205945, -0.02166347019,
        0.02166347019, 0.06682205945, 0.1187859178, 0.19020693],
    4: [-0.2414890379, -0.1828317791, -0.1429702938, -0.1109927073,
        -0.08325428516, -0.05802082643, -0.03428063914, -0.01134236995,
        0.01134236995, 0.03428063914, 0.05802082643, 0.08325428516,
        0.1109927073, 0.1429702938, 0.1828317791, 0.2414890379],
}

_matrices = {}
stats = {"calls": 0, "tokens": 0, "max_rel_err": 0.0}


def _get_matrices(device):
    key = str(device)
    if key not in _matrices:
        gen = torch.Generator().manual_seed(20260929)
        q, _ = torch.linalg.qr(
            torch.randn(_HEAD_DIM, _HEAD_DIM, generator=gen, dtype=torch.float64))
        rot = q.to(torch.float32).to(device).contiguous()
        qjl = (torch.randn(_HEAD_DIM, _HEAD_DIM, generator=gen, dtype=torch.float64)
               .to(torch.float32) / math.sqrt(_HEAD_DIM)).to(device).contiguous()
        _matrices[key] = (rot, qjl)
    return _matrices[key]


def _unpack_bits(packed, bit_width, length):
    values = packed.to(torch.int64)
    groups = values.shape[-1] // bit_width
    words = values.reshape(*values.shape[:-1], groups, bit_width)
    word = torch.zeros_like(words[..., 0], dtype=torch.int64)
    for byte in range(bit_width):
        word |= words[..., byte].to(torch.int64) << (8 * byte)
    lanes = [(word >> (lane * bit_width)) & ((1 << bit_width) - 1) for lane in range(8)]
    return torch.stack(lanes, dim=-1).reshape(*values.shape[:-1], groups * 8)[..., :length]


def _dequant(idx, qjlb, norm, gamma, rot, qjl, mse_bits):
    centroids = torch.tensor(CENTROIDS[mse_bits], dtype=torch.float32, device=idx.device)
    indices = _unpack_bits(idx, mse_bits, _HEAD_DIM).long()
    primary = centroids[indices]
    signs = _unpack_bits(qjlb, 1, _HEAD_DIM).to(torch.float32) * 2.0 - 1.0
    r_est = signs @ qjl * (math.sqrt(math.pi / 2.0) / _HEAD_DIM)
    norm_f = norm.to(torch.float32).unsqueeze(-1)
    gamma_f = gamma.to(torch.float32).unsqueeze(-1)
    rotated_hat = primary + (gamma_f / norm_f.clamp_min(1e-30)) * r_est
    return norm_f * (rotated_hat @ rot)


def _shadow_quantize(name, tensor):
    rot, qjl = _get_matrices(tensor.device)
    idx, qjlb, norm, gamma = torch.ops.turboquant.kv_cache_turbo_quant(
        tensor.contiguous(), rot, qjl, _MSE_BITS)
    stats["calls"] += 1
    stats["tokens"] += tensor.shape[0]
    if stats["calls"] == 1 or stats["calls"] % 500 == 0:
        print(f"[KVTQ] pid={os.getpid()} calls={stats['calls']} {name} total_tokens={stats['tokens']}", flush=True)
    if _DEBUG:
        recon = _dequant(idx, qjlb, norm, gamma, rot, qjl, _MSE_BITS)
        ref = tensor.to(torch.float32)
        denom = ref.norm(dim=-1).clamp_min(1e-6)
        rel = ((recon - ref).norm(dim=-1) / denom).max().item()
        stats["max_rel_err"] = max(stats["max_rel_err"], rel)
        if stats["calls"] <= 4 or stats["calls"] % 200 == 0:
            print(f"[KVTQ] call={stats['calls']} {name} tokens={tensor.shape[0]} "
                  f"max_rel_err={stats['max_rel_err']:.4f}", flush=True)


_installed = False


def install():
    global _installed
    if _installed:
        return
    _installed = True
    if not _ENABLED:
        print("[KVTQ] shadow disabled (VLLM_ASCEND_KVTQ=0)", flush=True)
        return
    import vllm_ascend.ops  # noqa: F401  (pre-load to avoid circular import)
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    orig = AscendAttentionBackendImpl.reshape_and_cache

    def patched(self, query, key, value, kv_cache, attn_metadata, output):
        if stats["calls"] == 0 and not stats.get("seen"):
            stats["seen"] = True
            print(f"[KVTQ] reshape_and_cache entered pid={os.getpid()} "
                  f"key={None if key is None else tuple(key.shape)} len_kv_cache={len(kv_cache)}", flush=True)
        try:
            if (key is not None and value is not None and len(kv_cache) > 1
                    and key.dim() == 3 and key.size(-1) == _HEAD_DIM):
                num = getattr(attn_metadata, "num_actual_tokens", key.shape[0])
                num = min(num, key.shape[0])
                if num > 0:
                    _shadow_quantize("key", key[:num])
                    _shadow_quantize("value", value[:num])
        except Exception as exc:  # shadow mode must never break serving
            print(f"[KVTQ] shadow quantize failed: {exc}", flush=True)
        return orig(self, query, key, value, kv_cache, attn_metadata, output)

    AscendAttentionBackendImpl.reshape_and_cache = patched
    print(f"[KVTQ] shadow installed on AscendAttentionBackendImpl.reshape_and_cache "
          f"(mse_bits={_MSE_BITS}, debug={_DEBUG})", flush=True)


def report():
    print(f"[KVTQ] stats: calls={stats['calls']} tokens={stats['tokens']} "
          f"max_rel_err={stats['max_rel_err']:.4f}", flush=True)