"""MiniMax-M3 MSA backend (M3b): prefill + decode block-sparse + IndexCache.

- prefill: verified MSA reference over in-hand q/k/v (overwrites dense output).
- decode: per-sequence indexer scores the new query against the cached index_k
  (an extra per-layer side cache, scattered by slot_mapping, mirroring the main
  paged KV layout), selects top-k blocks, and runs exact softmax over the
  gathered selected blocks' K/V.
- IndexCache: env MM3_INDEX_TOPK_FREQ (default 1). When >1, the expensive indexer
  top-k is recomputed only every N decode steps; in between the cached block
  selection is reused (plus the always-included local block). The decode-time
  saving of skipping the indexer is the IndexCache benefit.

Decode is enabled for batch sizes up to MM3_MSA_MAX_DECODE (default 8); larger
falls back to dense to stay safe. Any error falls back to the dense path.
"""
import os
import sys
import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
)

_BK = 128
_KB = 16
_LOCAL = 1
_INIT = 0
_FREQ = int(os.environ.get("MM3_INDEX_TOPK_FREQ", "1"))
_MAXDEC = int(os.environ.get("MM3_MSA_MAX_DECODE", "8"))


class AscendMSAImpl(AscendAttentionBackendImpl):
    _ann = False
    _pf_logged = False
    _dec_logged = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._idxk_cache = None          # [num_blocks, block_size, idx_dim]
        self._ic_step = 0                # decode step counter (IndexCache)
        self._ic_sel = None              # cached per-seq selected block ids: list[list[set]]
        if not AscendMSAImpl._ann:
            sys.stderr.write(f"[MSA] AscendMSAImpl ACTIVE (M3b: prefill+decode sparse, FREQ={_FREQ})\n")
            sys.stderr.flush()
            AscendMSAImpl._ann = True

    def _ensure_idxk_cache(self, kv_cache, idx_dim, device, dtype):
        if self._idxk_cache is None:
            kc = kv_cache[0]
            num_blocks, block_size = kc.shape[0], kc.shape[1]
            self._idxk_cache = torch.zeros(num_blocks, block_size, idx_dim, device=device, dtype=dtype)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        iq = getattr(self, "_msa_iq", None)
        ik = getattr(self, "_msa_ik", None)
        self._msa_iq = None
        self._msa_ik = None
        if iq is not None and ik is not None:
            try:
                _nih = iq.shape[1]
                if _nih > self.num_kv_heads:
                    _tps = get_tensor_model_parallel_world_size()
                    _tpr = get_tensor_model_parallel_rank()
                    if _tps >= _nih and _tps % _nih == 0:
                        _per = _tps // _nih
                        _g = min(_tpr // _per, _nih - 1)
                        iq = iq[:, _g:_g + self.num_kv_heads, :].contiguous()
            except Exception:
                pass
        out = super().forward(layer, query, key, value, kv_cache, attn_metadata,
                              output, output_scale, output_block_scale)
        if iq is None or ik is None or attn_metadata is None or kv_cache is None:
            return out
        try:
            idx_dim = ik.shape[-1]
            self._ensure_idxk_cache(kv_cache, idx_dim, ik.device, ik.dtype)
            slot = attn_metadata.slot_mapping
            nt = slot.shape[0]
            # write index_k of current tokens into the side cache (prefill+decode)
            self._idxk_cache.view(-1, idx_dim)[slot[:nt]] = ik[:nt, 0, :].to(self._idxk_cache.dtype)
        except Exception as e:
            if not AscendMSAImpl._dec_logged:
                sys.stderr.write(f"[MSA] idxk cache write ERROR: {type(e).__name__}: {e}\n"); sys.stderr.flush()
            return out

        ndt = getattr(attn_metadata, "num_decode_tokens", 0) or 0
        num_decodes = getattr(attn_metadata, "num_decodes", 0) or 0
        S = query.shape[0]
        nH, nKV, hd = self.num_heads, self.num_kv_heads, self.head_size

        # ---------- PREFILL (no decode tokens): verified reference over in-hand q/k/v ----------
        if ndt == 0 and S > 1 and iq.shape[0] == S:
            try:
                from vllm_ascend.models.minimax_m3_msa import msa_attention
                q = query.reshape(S, nH, hd); k = key.reshape(S, nKV, hd); v = value.reshape(S, nKV, hd)
                mo = msa_attention(q.float(), k.float(), v.float(), iq.float(), ik[:, 0, :].float(),
                                   block_size=_BK, topk_blocks=_KB, local_blocks=_LOCAL, init_blocks=_INIT,
                                   scale=self.scale).to(out.dtype)
                out[:S] = mo.reshape(out[:S].shape)
                if not AscendMSAImpl._pf_logged:
                    sys.stderr.write(f"[MSA] prefill SPARSE applied S={S}\n"); sys.stderr.flush()
                    AscendMSAImpl._pf_logged = True
            except Exception as e:
                import traceback
                if not AscendMSAImpl._pf_logged:
                    sys.stderr.write(f"[MSA] prefill ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}\n"); sys.stderr.flush()
                    AscendMSAImpl._pf_logged = True
            return out

        # ---------- DECODE (1 new token per sequence) ----------
        if num_decodes == 0 or ndt != num_decodes or num_decodes > _MAXDEC:
            return out  # mixed / spec / large batch -> dense for now
        try:
            from vllm_ascend.models.minimax_m3_msa import msa_decode_attn
            bt = attn_metadata.block_tables
            seqlens = attn_metadata.seq_lens
            if hasattr(seqlens, "tolist"):
                seqlens_l = seqlens.tolist()
            else:
                seqlens_l = list(seqlens)
            kc = kv_cache[0]; vc = kv_cache[1]
            recompute = (self._ic_step % _FREQ == 0)
            self._ic_step += 1
            if self._ic_sel is None or len(self._ic_sel) != num_decodes:
                self._ic_sel = [None] * num_decodes
            qd = query.reshape(S, nH, hd)[:num_decodes]        # [nd,nH,hd]
            iqd = iq.reshape(iq.shape[0], -1, idx_dim)[:num_decodes]  # [nd,G,d]
            for sidx in range(num_decodes):
                L = int(seqlens_l[sidx])
                nb = (L + _BK - 1) // _BK
                blocks = bt[sidx, :nb].to(torch.long)
                kf = kc[blocks].reshape(nb * _BK, nKV, hd)[:L]   # [L,nKV,hd]
                vf = vc[blocks].reshape(nb * _BK, nKV, hd)[:L]
                ikf = self._idxk_cache[blocks].reshape(nb * _BK, idx_dim)[:L]  # [L,d]
                q1 = qd[sidx:sidx + 1]                            # [1,nH,hd]
                iq1 = iqd[sidx:sidx + 1]                          # [1,G,d]
                if recompute or self._ic_sel[sidx] is None:
                    o1 = msa_decode_attn(q1.float(), kf.float(), vf.float(), iq1.float(), ikf.float(),
                                         block_size=_BK, topk_blocks=_KB, local_blocks=_LOCAL, init_blocks=_INIT,
                                         scale=self.scale, positions=torch.tensor([L - 1], device=q1.device),
                                         return_sel=True)
                    o1, sel = o1
                    self._ic_sel[sidx] = sel
                else:
                    # IndexCache: reuse cached block selection (+ current local block), skip indexer
                    o1 = msa_decode_attn(q1.float(), kf.float(), vf.float(), iq1.float(), ikf.float(),
                                         block_size=_BK, topk_blocks=_KB, local_blocks=_LOCAL, init_blocks=_INIT,
                                         scale=self.scale, positions=torch.tensor([L - 1], device=q1.device),
                                         forced_sel=self._ic_sel[sidx])
                out[sidx] = o1[0].reshape(out[sidx].shape).to(out.dtype)
            if not AscendMSAImpl._dec_logged:
                sys.stderr.write(f"[MSA] decode SPARSE applied nd={num_decodes} L0={int(seqlens_l[0])} recompute={recompute} FREQ={_FREQ}\n"); sys.stderr.flush()
                AscendMSAImpl._dec_logged = True
        except Exception as e:
            import traceback
            if not AscendMSAImpl._dec_logged:
                sys.stderr.write(f"[MSA] decode ERROR (kept dense): {type(e).__name__}: {e}\n{traceback.format_exc()}\n"); sys.stderr.flush()
                AscendMSAImpl._dec_logged = True
        return out


class AscendMSABackend(AscendAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> type["AscendMSAImpl"]:
        return AscendMSAImpl
