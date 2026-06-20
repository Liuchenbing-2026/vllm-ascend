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
        self._ic_step = 0                # decode step counter (IndexCache, legacy)
        self._ic_sel = None              # legacy per-seq selection (unused by vec path)
        self._ic_cache = {}              # IndexCache(vec): key(block0)->(topk_idx[G,K], sel_seq_len)
        self._ic_logged = False
        if not AscendMSAImpl._ann:
            sys.stderr.write(f"[MSA] AscendMSAImpl ACTIVE (M3b: prefill+decode sparse, FREQ={_FREQ})\n")
            sys.stderr.flush()
            AscendMSAImpl._ann = True

    def _ensure_idxk_cache(self, kv_cache, idx_dim, device, dtype):
        kc = kv_cache[0]
        block_size = kc.shape[1]
        # idxk side cache must cover the REAL num_gpu_blocks. Under FULL cudagraph the
        # kv_cache passed at capture is a 1-block dummy (vllm swaps the real paged cache
        # in only at replay and does NOT manage our side tensor); sizing from kc.shape[0]
        # gives 1 block -> slot=block_id*block_size overruns -> MTE OOB. Allocate a fixed
        # generous block count (>= real num_gpu_blocks) once.
        # Size to the REAL num_gpu_blocks (cache_config mutated in-place after
        # profiling). Profiling forward: num_gpu_blocks is None -> 1-block dummy
        # (no profiling-peak inflation -> no startup OOM). Warmup forward (eager,
        # pre-capture): real count -> allocate the persistent buffer ONCE so the
        # FULL graph captures/replays a stable, correctly-sized side cache.
        _ng = None
        _cc = getattr(self, "_cache_config", None)
        if _cc is not None:
            _ng = getattr(_cc, "num_gpu_blocks", None)
        _ov = os.environ.get("MM3_IDXK_NUM_BLOCKS")
        if _ov:
            num_blocks = max(kc.shape[0], int(_ov))
        elif _ng:
            num_blocks = max(kc.shape[0], int(_ng))
        else:
            num_blocks = kc.shape[0]
        if self._idxk_cache is None or self._idxk_cache.shape[0] < num_blocks:
            # 4D ND [num_blocks, block_size, num_kv_heads=1, idx_dim] so the graph-safe
            # paged writer _npu_reshape_and_cache accepts it (a bare 3D cache OOBs:
            # 0x3000035). N=1 makes it byte-identical to the old 3D layout, so the
            # kernel (flat GM pointer) and all .view/.reshape readers are unaffected.
            # BNSD [num_blocks, num_kv_heads=1, block_size, idx_dim] for the graph-safe
            # aclnn writer npu_reshape_and_cache_bnsd. ATB reshape_and_cache is NOT
            # ACL-graph capturable (OperationSetup fails at replay -> 507000). N=1 keeps
            # this byte-identical to the prior layout so kernel/.view/.reshape readers
            # are unaffected; only the declared shape changes for the bnsd op.
            self._idxk_cache = torch.zeros(num_blocks, 1, block_size, idx_dim, device=device, dtype=dtype)
            self._idxk_vcache = None

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if os.environ.get("MM3_MSA_BYPASS", "0") == "1":
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)
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
            nt = attn_metadata.num_actual_tokens
            if not getattr(AscendMSAImpl, "_racdbg", False):
                AscendMSAImpl._racdbg = True
                try:
                    sys.stderr.write(f"[MSA-DBG] kc={tuple(kv_cache[0].shape)} idxk={tuple(self._idxk_cache.shape)} ik={tuple(ik.shape)} slot={tuple(slot.shape)}/{slot.dtype} nt={nt} idx_dim={idx_dim}\n"); sys.stderr.flush()
                except Exception as _e:
                    sys.stderr.write(f"[MSA-DBG] err {_e}\n"); sys.stderr.flush()
            # write index_k of current tokens into the side cache (prefill+decode) via the
            # graph-safe paged-cache writer (same primitive as the main KV cache). Raw
            # advanced-index assignment and index_copy_/ScatterUpdate are NOT cudagraph-safe
            # (MTE OOB on FULL-graph replay); _npu_reshape_and_cache is. idxk is key-only so
            # key==value and key_cache==value_cache (idempotent double write).
            if os.environ.get("MM3_SKIP_IDXK_WRITE", "0") != "1":
                # graph-safe aclnn side-cache write (kvcomp's proven primitive). seq_len
                # = per-request query-token count (decode -> all ones, constant across
                # steps so replay-safe). idxk is key-only: q==k_out single cache.
                _ndt = getattr(attn_metadata, "num_decode_tokens", 0) or 0
                if _ndt == nt and _ndt > 0:
                    # pure decode: exactly 1 query token per request, so per-request
                    # query-len is a constant ones(nt). Building it as a constant avoids
                    # a captured Sub over the NON-persistent query_start_loc (fresh H2D
                    # each step) which read a stale address at FULL-graph replay -> MTE OOB.
                    _qlen = torch.ones(nt, dtype=torch.int32, device=slot.device)
                else:
                    # prefill / mixed runs eager (capture size is decode=1), so reading
                    # query_start_loc here is safe and gives the true per-request lengths.
                    _qsl = attn_metadata.query_start_loc
                    _qlen = (_qsl[1:] - _qsl[:-1]).to(torch.int32)
                _k2 = ik[:nt, 0, :].to(self._idxk_cache.dtype).contiguous()  # [nt, idx_dim]
                torch.ops._C_ascend.npu_reshape_and_cache_bnsd(
                    _k2, self._idxk_cache, slot[:nt].to(torch.int32), _qlen, self._idxk_cache)
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

        # ---------- DECODE (vectorized, FULL-cudagraph capturable) ----------
        if num_decodes == 0 or ndt != num_decodes or num_decodes > _MAXDEC:
            return out
        try:
            from vllm_ascend.models.minimax_m3_msa import msa_decode_vec
            bt = attn_metadata.block_tables
            seqlens = attn_metadata.seq_lens
            kc = kv_cache[0]; vc = kv_cache[1]
            B = num_decodes
            qd = query.reshape(S, nH, hd)[:B]
            iqd = iq.reshape(iq.shape[0], -1, idx_dim)[:B]
            # ---- IndexCache (eager): recompute indexer selection every _FREQ decode
            # steps per sequence; reuse the cached selection otherwise (the local block
            # is always re-added inside the attend fn). Cadence keyed on seq_len growth,
            # so batched out-of-phase sequences each keep their own _FREQ cadence and a
            # stale block-table reuse (seq_len < cached) self-invalidates.
            ic_topk = None
            if _FREQ > 1:
                from vllm_ascend.models.minimax_m3_msa import msa_decode_select
                keys = bt[:B, 0].tolist()
                sls = seqlens[:B].tolist()
                need = [b for b in range(B)
                        if (self._ic_cache.get(keys[b]) is None
                            or sls[b] < self._ic_cache[keys[b]][1]
                            or sls[b] - self._ic_cache[keys[b]][1] >= _FREQ)]
                if need:
                    nidx = torch.tensor(need, device=query.device, dtype=torch.long)
                    fresh = msa_decode_select(iqd[nidx], seqlens[:B][nidx], bt[:B][nidx],
                                              self._idxk_cache, block_size=_BK,
                                              topk_blocks=_KB, init_blocks=_INIT)
                    for j, b in enumerate(need):
                        self._ic_cache[keys[b]] = (fresh[j], sls[b])
                G_ic, K_ic = self._ic_cache[keys[0]][0].shape
                ic_topk = torch.empty(B, G_ic, K_ic, dtype=torch.long, device=query.device)
                for b in range(B):
                    ic_topk[b] = self._ic_cache[keys[b]][0]
                if len(self._ic_cache) > 4096:
                    self._ic_cache = {keys[b]: self._ic_cache[keys[b]] for b in range(B)}
                if not self._ic_logged:
                    sys.stderr.write(f"[MSA] IndexCache ON FREQ={_FREQ}: recompute {len(need)}/{B} this step\n"); sys.stderr.flush()
                    self._ic_logged = True
            if os.environ.get("MM3_DECODE_OPGRAPH", "0") == "1":
                from vllm_ascend.models.minimax_m3_msa import msa_decode_fia_opgraph
                # persistent device seq_lens (in-place-updated graph buffer) so the captured
                # selection op reads the correct seq_len -> numBlocks at FULL-graph replay
                # (positions-derived sl_dev was unreliable -> garbage numBlocks -> the
                # kernel read block_table[b] past its width -> MTE OOB).
                _slg = getattr(attn_metadata, "seq_lens_gpu", None)
                if _slg is not None:
                    sl_dev = _slg[:B].to(torch.int32)
                else:
                    _pos = getattr(self, "_msa_positions", None)
                    sl_dev = (_pos[:B].to(torch.int32) + 1) if _pos is not None else seqlens[:B].to(query.device, dtype=torch.int32)
                _tot = _KB + _LOCAL + _INIT
                if getattr(self, "_sel_buffer", None) is None:
                    self._sel_buffer = torch.zeros(_MAXDEC, iqd.shape[1], _tot, dtype=torch.int32, device=query.device)
                ovo = msa_decode_fia_opgraph(qd, iqd, sl_dev, bt[:B], kc, vc, self._idxk_cache,
                                             block_size=_BK, topk_blocks=_KB, scale=self.scale,
                                             num_heads=nH, num_kv_heads=nKV,
                                             local_blocks=_LOCAL, init_blocks=_INIT,
                                             sel_buffer=self._sel_buffer[:B])
                out[:B] = ovo.reshape(out[:B].shape).to(out.dtype)
            elif os.environ.get("MM3_DECODE_FGRAPH", "0") == "1":
                from vllm_ascend.models.minimax_m3_msa import msa_decode_fia_graph
                _slg = getattr(attn_metadata, "seq_lens_gpu", None)
                if _slg is not None:
                    sl_dev = _slg[:B].to(torch.int32)
                else:
                    _pos = getattr(self, "_msa_positions", None)
                    sl_dev = (_pos[:B].to(torch.int32) + 1) if _pos is not None else seqlens[:B].to(query.device, dtype=torch.int32)
                ovg = msa_decode_fia_graph(qd, iqd, sl_dev, bt[:B], kc, vc, self._idxk_cache,
                                           block_size=_BK, topk_blocks=_KB, scale=self.scale,
                                           num_heads=nH, num_kv_heads=nKV,
                                           local_blocks=_LOCAL, init_blocks=_INIT)
                out[:B] = ovg.reshape(out[:B].shape).to(out.dtype)
            elif os.environ.get("MM3_DECODE_FIA", "0") == "1":
                from vllm_ascend.models.minimax_m3_msa import msa_decode_fia
                ovf = msa_decode_fia(qd, iqd, seqlens[:B], bt[:B], kc, vc, self._idxk_cache,
                                     block_size=_BK, topk_blocks=_KB, scale=self.scale,
                                     num_heads=nH, num_kv_heads=nKV,
                                     local_blocks=_LOCAL, init_blocks=_INIT, topk_idx=ic_topk)
                if os.environ.get("MM3_DECODE_FIA_CHECK", "0") == "1" and not AscendMSAImpl._dec_logged:
                    ovv = msa_decode_vec(qd, iqd, seqlens[:B], bt[:B], kc, vc, self._idxk_cache,
                                         block_size=_BK, topk_blocks=_KB, scale=self.scale,
                                         local_blocks=_LOCAL, init_blocks=_INIT, topk_idx=ic_topk)
                    _diff = (ovf.float() - ovv.float()).abs().max().item()
                    sys.stderr.write(f"[MSA] FIA vs VEC maxabsdiff={_diff:.3e} B={B}\n"); sys.stderr.flush()
                out[:B] = ovf.reshape(out[:B].shape).to(out.dtype)
            else:
                ov = msa_decode_vec(qd, iqd, seqlens[:B], bt[:B], kc, vc, self._idxk_cache,
                                    block_size=_BK, topk_blocks=_KB, scale=self.scale,
                                    local_blocks=_LOCAL, init_blocks=_INIT, topk_idx=ic_topk)
                out[:B] = ov.reshape(out[:B].shape).to(out.dtype)
            if not AscendMSAImpl._dec_logged:
                sys.stderr.write(f"[MSA] decode VEC applied nd={B} (FULL-capable)\n"); sys.stderr.flush()
                AscendMSAImpl._dec_logged = True
        except Exception as e:
            import traceback
            if not AscendMSAImpl._dec_logged:
                sys.stderr.write(f"[MSA] decode VEC ERROR (kept dense): {type(e).__name__}: {e}\n{traceback.format_exc()}\n"); sys.stderr.flush()
                AscendMSAImpl._dec_logged = True
        return out


class AscendMSABackend(AscendAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> type["AscendMSAImpl"]:
        return AscendMSAImpl
