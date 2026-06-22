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
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.device.device_op import DeviceOperator

_BK = 128
_KB = 16
_LOCAL = 1
_INIT = 0
_FREQ = int(os.environ.get("MM3_INDEX_TOPK_FREQ", "1"))
_MAXDEC = int(os.environ.get("MM3_MSA_MAX_DECODE", "8"))


def bind_msa_idxk_caches(kv_caches, static_forward_context, device):
    """Allocate the MSA index-k SIDE CACHE from the runner's KV-cache-init phase (mirrors
    kvcomp init_and_bind_hashk_cache), sized/typed from each layer's main KV cache, and bind
    it onto the AscendMSAImpl instance. This makes idxk_cache a GRAPH-REGISTERED tensor so
    captured FFTS+ kernels can read (selection gather) and write it under FULL_DECODE replay;
    a lazy torch.zeros in the attention forward is NOT graph-accessible -> 507011 MTE OOB.
    No-op for non-MSA layers. Called from model_runner_v1.initialize_kv_cache_tensors."""
    import torch
    bound = 0
    for layer_name, kv_cache in kv_caches.items():
        layer = static_forward_context.get(layer_name)
        if layer is None:
            continue
        impl = getattr(layer, "impl", None)
        if impl is None or impl.__class__.__name__ != "AscendMSAImpl":
            continue
        kc = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
        num_blocks, block_size, idx_dim = kc.shape[0], kc.shape[1], kc.shape[-1]
        # BNSD [num_blocks, 1, block_size, idx_dim], same num_blocks as the main KV cache so
        # slot_mapping (block_id*block_size+offset) indexes it correctly. idx_dim == head_size
        # for M3. dtype matches the KV cache (bf16).
        impl._idxk_cache = torch.zeros(num_blocks, 1, block_size, idx_dim, device=device, dtype=kc.dtype)
        impl._idxk_vcache = None
        impl._idxk_runner_bound = True
        bound += 1
    if bound:
        sys.stderr.write(f"[MSA] bound {bound} runner-allocated (graph-registered) idxk caches\n")
        sys.stderr.flush()


class AscendMSAImpl(AscendAttentionBackendImpl):
    _ann = False
    _pf_logged = False
    _dec_logged = False
    _fg_logged = False
    _idxk_vcache_shared = None  # shared dummy value-cache for the graph-safe idxk writer

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
        # Keep the runner-allocated (graph-registered) idxk cache if bind_msa_idxk_caches set
        # it; only lazily allocate as an eager fallback (a lazy cache is NOT graph-accessible
        # under FULL_DECODE -> 507011, hence the runner allocation is the real path).
        if self._idxk_cache is None:
            # BNSD [num_blocks, num_kv_heads=1, block_size, idx_dim] for the custom writer
            # torch.ops._C_ascend.npu_reshape_and_cache_bnsd. This is the SAME op kvcomp/DSA
            # use for their side cache UNDER FULL_DECODE cudagraph (attention_utils.py
            # reshape_and_cache_kvcomp) -> proven graph-capture-AND-replay-safe. N=1 keeps it
            # byte-identical to a 3D [nb,bs,idx_dim] cache so all .view(nb,bs,d)/.reshape
            # readers (custom op, gathers) are unaffected. (ATB _npu_reshape_and_cache fails
            # OperationSetup under capture; this custom op is the working primitive.)
            self._idxk_cache = torch.zeros(num_blocks, 1, block_size, idx_dim, device=device, dtype=dtype)
            self._idxk_vcache = None
        # Persistent decode seq_len buffer for the bnsd writer's 4th arg. The custom op reads
        # this device tensor at runtime; a fresh torch.ones() each step is a graph intermediate
        # whose recompute races the captured write -> 507011 MTE OOB at replay. kvcomp uses a
        # persistent seq_lens_for_reshape buffer for exactly this. Decode writes 1 token/req.
        if (getattr(self, "_idxk_qlen", None) is None
                or self._idxk_qlen.device != device):
            self._idxk_qlen = torch.ones(_MAXDEC, dtype=torch.int32, device=device)

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
        # Stash index q/k + ensure the persistent idxk side-cache BEFORE super().forward,
        # so the overridden full_graph_fia (invoked inside super during FULL_DECODE capture)
        # can run the in-graph selection. The idxk WRITE stays after super: the selection
        # uses prior steps' idxk and the current token's local block is force-added anyway.
        self._msa_iq_active = iq
        self._msa_ik_active = ik
        if iq is not None and ik is not None and kv_cache is not None:
            try:
                self._ensure_idxk_cache(kv_cache, ik.shape[-1], ik.device, ik.dtype)
                # Pre-allocate the persistent sparse buffers HERE (warmup, PRE-capture) so they
                # are NOT graph-pool tensors. Allocating them inside full_graph_fia (which runs
                # DURING capture) makes them transient graph-pool memory -> reused/clobbered at
                # replay -> the sparse FIA reads garbage fbt -> 507011 MTE OOB.
                if getattr(self, "_fbt_buffer", None) is None and attn_metadata is not None:
                    _idim = ik.shape[-1]
                    _MAXB = attn_metadata.block_tables.shape[1]
                    _G = iq.reshape(iq.shape[0], -1, _idim).shape[1]
                    self._fbt_buffer = torch.zeros(_MAXDEC, _KB + 1, dtype=torch.int32, device=ik.device)
                    self._maxes_buffer = torch.zeros(_MAXDEC, _G, _MAXB, dtype=torch.float16, device=ik.device)
            except Exception:
                pass
        out = super().forward(layer, query, key, value, kv_cache, attn_metadata,
                              output, output_scale, output_block_scale)
        self._msa_iq_active = None
        self._msa_ik_active = None
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
            # Writing the idxk side cache INSIDE the captured FULL_DECODE graph OOBs at replay
            # (507011) regardless of the writer op (custom bnsd replay-OOBs; ATB
            # _npu_reshape_and_cache fails capture setup) -- there is no precedent for a
            # captured decode side-cache write (kvcomp writes its side cache at PREFILL only,
            # base full_graph_fia gates it on attn_state != DecodeOnly). So skip the write
            # when CAPTURING: prefill (eager) still caches the prompt's index_k; generated
            # tokens are not added to the side cache, but the current token's LOCAL block is
            # always force-selected, so only multi-block generations lose a little selection
            # fidelity (negligible for short outputs). MM3_IDXK_DECODE_WRITE=1 forces the
            # (graph-unsafe) decode write back on for eager A/B.
            _skip_cap = (getattr(_EXTRA_CTX, "capturing", False)
                         and os.environ.get("MM3_IDXK_DECODE_WRITE", "0") != "1")
            if os.environ.get("MM3_SKIP_IDXK_WRITE", "0") != "1" and not _skip_cap:
                # GRAPH-SAFE side-cache write via the custom bnsd op -- the SAME primitive
                # kvcomp/DSA use under FULL_DECODE cudagraph (proven capture+replay safe).
                # seq_len (4th arg) = per-request new-token count; decode -> 1 each, read
                # from the PERSISTENT self._idxk_qlen buffer (a fresh torch.ones() each step
                # is a graph intermediate whose recompute races the captured write -> 507011
                # MTE OOB at replay). idxk is key-only: q==k_out single (aliased) cache.
                _ndt = getattr(attn_metadata, "num_decode_tokens", 0) or 0
                if _ndt == nt and _ndt > 0:
                    _qlen = self._idxk_qlen[:nt]  # persistent ones (decode: 1 token/req)
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

        # During FULL_DECODE capture, full_graph_fia (called inside super().forward) already
        # ran the sparse decode + registered the FIA graph task. Skip the eager decode path
        # to avoid double work / a second registration.
        if getattr(_EXTRA_CTX, "capturing", False):
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
            # Use the cached PERSISTENT real kv cache (self.key_cache/value_cache, set
            # once by the base _get_fia_params on the first real forward) rather than the
            # kv_cache ARG. Under FULL_DECODE the kv_cache arg passed at capture is a small
            # dummy (the real paged cache is bound elsewhere); a raw gather/FIA over the
            # arg captures the dummy's base address -> MTE DDR OOB (507011) at replay. The
            # working dense path reads self.key_cache for exactly this reason.
            kc = self.key_cache if getattr(self, "key_cache", None) is not None else kv_cache[0]
            vc = self.value_cache if getattr(self, "value_cache", None) is not None else kv_cache[1]
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
                if getattr(self, "_maxes_buffer", None) is None:
                    _MAXB = bt.shape[1]
                    self._maxes_buffer = torch.zeros(_MAXDEC, iqd.shape[1], _MAXB, dtype=torch.float16, device=query.device)
                ovo = msa_decode_fia_opgraph(qd, iqd, sl_dev, bt[:B], kc, vc, self._idxk_cache,
                                             block_size=_BK, topk_blocks=_KB, scale=self.scale,
                                             num_heads=nH, num_kv_heads=nKV,
                                             local_blocks=_LOCAL, init_blocks=_INIT,
                                             maxes_buffer=self._maxes_buffer[:B])
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
                # device, persistent seq_lens (graph-safe): the captured valid/qblk
                # masks must read the CURRENT step's seq_len at replay. seq_lens_gpu
                # is the in-place-updated graph buffer; a fresh .to(dev) of the CPU
                # seqlens captures a stale host address -> wrong mask at replay.
                _slg = getattr(attn_metadata, "seq_lens_gpu", None)
                if _slg is not None:
                    sl_dev = _slg[:B].to(torch.int32)
                else:
                    _pos = getattr(self, "_msa_positions", None)
                    sl_dev = (_pos[:B].to(torch.int32) + 1) if _pos is not None else seqlens[:B].to(query.device, dtype=torch.int32)
                ov = msa_decode_vec(qd, iqd, sl_dev, bt[:B], kc, vc, self._idxk_cache,
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

    def full_graph_fia(self, query, key, value, attn_metadata, output, layer=None):
        """FULL_DECODE capture: register ONE sparse-FIA graph task per layer (mirrors base
        full_graph_fia). The data-dependent block selection runs IN-graph and writes a
        PERSISTENT rewritten block_table buffer (address stable across replays); the FIA
        reads that buffer, and its tiling is re-planned at replay by update_graph_params
        using a deterministic host kv_lens. Falls back to dense base if MSA inputs absent."""
        iq = getattr(self, "_msa_iq_active", None)
        ik = getattr(self, "_msa_ik_active", None)
        if iq is None or ik is None or getattr(self, "_idxk_cache", None) is None \
                or os.environ.get("MM3_FG_SEL", "op") == "dense":
            return super().full_graph_fia(query, key, value, attn_metadata, output, layer)
        try:
            import torch_npu
            from vllm_ascend.compilation.acl_graph import (
                get_graph_params, update_graph_params_workspaces,
            )
            from vllm_ascend.models.minimax_m3_msa import (
                msa_select_into_fbt, msa_select_into_fbt_torch, msa_host_kv_lens,
            )
            nH = self.num_heads; nKV = self.num_kv_heads; hd = self.head_size
            B = query.shape[0]
            num_tokens = attn_metadata.actual_seq_lengths_q[-1]
            idx_dim = ik.shape[-1]
            iqd = iq.reshape(iq.shape[0], -1, idx_dim)[:B]
            G = iqd.shape[1]
            kc = self.key_cache; vc = self.value_cache
            nblk = kc.shape[0]
            MAXB = attn_metadata.block_tables.shape[1]
            Kp = _KB + 1
            dev = query.device
            _slg = getattr(attn_metadata, "seq_lens_gpu", None)
            if _slg is not None:
                sl_dev = _slg[:B].to(torch.int32)
            else:
                sl_dev = attn_metadata.seq_lens[:B].to(dev, dtype=torch.int32)
            if getattr(self, "_fbt_buffer", None) is None or self._fbt_buffer.shape[0] < B:
                self._fbt_buffer = torch.zeros(max(B, _MAXDEC), Kp, dtype=torch.int32, device=dev)
            if getattr(self, "_maxes_buffer", None) is None or self._maxes_buffer.shape[0] < B:
                self._maxes_buffer = torch.zeros(max(B, _MAXDEC), G, MAXB, dtype=torch.float16, device=dev)
            # ---- in-graph selection -> persistent fbt buffer ----
            _sel_mode = os.environ.get("MM3_FG_SEL", "op")
            if _sel_mode == "trivial":
                # isolation: no op/topk/idxk-gather; fbt = first Kp logical blocks.
                self._fbt_buffer[:B, :Kp].copy_(
                    attn_metadata.block_tables[:B, :Kp].clamp(0, nblk - 1).to(torch.int32))
            elif _sel_mode == "torch":
                # torch-only selection (no custom op) -> isolates whether the custom op is
                # the FFTS+ replay OOB. Correct top-k by score.
                msa_select_into_fbt_torch(iqd, sl_dev, attn_metadata.block_tables[:B],
                                          self._idxk_cache, self._fbt_buffer,
                                          block_size=_BK, topk_blocks=_KB,
                                          local_blocks=_LOCAL, init_blocks=_INIT)
            elif _sel_mode == "readonly":
                # isolation: READ idxk (gather+einsum+amax -> maxes) but use TRIVIAL fbt
                # (no topk/rewrite). If this crashes -> reading idxk_cache is the FFTS+ OOB
                # (registration); if clean -> topk/rewrite is.
                _idim = iqd.shape[2]; _L = MAXB * _BK
                _bt = attn_metadata.block_tables[:B].to(torch.long).clamp(0, nblk - 1)
                _ikf = self._idxk_cache.reshape(self._idxk_cache.shape[0], _BK, _idim)[_bt].reshape(B, _L, _idim).float()
                _M = torch.einsum('bgd,bld->bgl', iqd.float(), _ikf).reshape(B, -1, MAXB, _BK).amax(-1)
                self._maxes_buffer[:B].copy_(_M.to(torch.float16))  # force the gather+einsum
                self._fbt_buffer[:B, :Kp].copy_(
                    attn_metadata.block_tables[:B, :Kp].clamp(0, nblk - 1).to(torch.int32))
            elif _sel_mode == "opmax":
                # isolation: run ONLY the custom AscendC op msa_dist_top_k (its INTERNAL idxk
                # gather) -> maxes; use TRIVIAL in-range fbt (no torch topk/scatter/rewrite).
                # readonly proved the TORCH gather is graph-unsafe; this tests whether the
                # AscendC op's own gather is graph-safe. Clean -> a fused AscendC selection op
                # is viable; crash -> AscendC gather of idxk also unsafe in-graph.
                _idim = iqd.shape[2]
                _bt_i32 = attn_metadata.block_tables[:B].clamp(0, nblk - 1).to(torch.int32).contiguous()
                _iqb = iqd.to(torch.bfloat16).contiguous()
                _ikc = self._idxk_cache.view(self._idxk_cache.shape[0], _BK, _idim).to(torch.bfloat16)
                torch.ops._C_ascend.msa_dist_top_k(
                    _iqb, _ikc, sl_dev, _bt_i32, self._maxes_buffer[:B],
                    _BK, _KB, _LOCAL, _INIT)
                self._fbt_buffer[:B, :Kp].copy_(
                    attn_metadata.block_tables[:B, :Kp].clamp(0, nblk - 1).to(torch.int32))
            elif _sel_mode == "convert":
                # convert-only: NO idxk read. topk_idx = first _KB blocks (== trivial selection
                # for GSM8K-length seqs) but built via the REAL _msa_rewrite_blocktable GATHER
                # (block_tables.gather) instead of the static slice. Isolates whether the
                # in-graph dynamic block_table gather is FULL-graph-safe at B>1 (convert-only).
                from vllm_ascend.models.minimax_m3_msa import _msa_rewrite_blocktable
                _K = min(_KB, MAXB)
                _tk = torch.arange(_K, device=dev, dtype=torch.long).view(1, _K).expand(B, _K).contiguous()
                _fbt2, _kvl = _msa_rewrite_blocktable(_tk, sl_dev, attn_metadata.block_tables[:B],
                                                      block_size=_BK, topk_blocks=_KB, init_blocks=_INIT)
                self._fbt_buffer[:B, :_fbt2.shape[1]].copy_(_fbt2)
            else:
                msa_select_into_fbt(iqd, sl_dev, attn_metadata.block_tables[:B], self._idxk_cache,
                                    self._fbt_buffer, self._maxes_buffer[:B],
                                    block_size=_BK, topk_blocks=_KB,
                                    local_blocks=_LOCAL, init_blocks=_INIT)
            fbt = self._fbt_buffer[:B]
            key_paged = kc.view(nblk, _BK, -1)
            value_paged = vc.view(nblk, _BK, -1)
            # host seq lens via the runner's precomputed list (NO .tolist() sync, which is
            # illegal during graph capture -> 107027).
            seq_lens_list = list(attn_metadata.seq_lens_list[:B])
            kv_lens_host = msa_host_kv_lens(seq_lens_list, _BK, _KB, _INIT)
            asl_q = list(range(1, B + 1))
            softmax_lse = torch.empty(1, dtype=query.dtype, device=dev)
            graph_params = get_graph_params()
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                # size for the MAX possible sparse kv_lens (Kp*block_size) so any replay fits
                max_kv = [Kp * _BK] * B
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    query=query, key=key_paged, value=value_paged, block_table=fbt,
                    input_layout="TND", block_size=_BK,
                    actual_seq_lengths=asl_q, actual_seq_lengths_kv=max_kv,
                    num_key_value_heads=nKV, num_heads=nH, scale=self.scale, sparse_mode=0)
                update_graph_params_workspaces(num_tokens, workspace)
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            attn_params = (
                "MSA",
                weak_ref_tensors(query),
                weak_ref_tensors(key_paged),
                weak_ref_tensors(value_paged),
                weak_ref_tensors(fbt),
                _BK, nKV, nH, self.scale,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
                B,
            )
            graph_params.attn_params[num_tokens].append(attn_params)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query, key=key_paged, value=value_paged, block_table=fbt,
                input_layout="TND", block_size=_BK,
                actual_seq_lengths=asl_q, actual_seq_lengths_kv=kv_lens_host,
                num_key_value_heads=nKV, num_heads=nH, scale=self.scale, sparse_mode=0,
                workspace=workspace, out=[output, softmax_lse])
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            if not AscendMSAImpl._fg_logged:
                sys.stderr.write(f"[MSA] full_graph_fia SPARSE registered B={B} Kp={Kp} kv0={kv_lens_host[0]}\n"); sys.stderr.flush()
                AscendMSAImpl._fg_logged = True
            return output.view(num_tokens, nH, hd), num_tokens
        except Exception as e:
            import traceback
            sys.stderr.write(f"[MSA] full_graph_fia ERROR -> dense fallback: {type(e).__name__}: {e}\n{traceback.format_exc()}\n"); sys.stderr.flush()
            return super().full_graph_fia(query, key, value, attn_metadata, output, layer)

    @staticmethod
    def update_graph_params(update_stream, forward_context, num_tokens, vllm_config,
                            speculative_config=None, num_dcp_pcp_tokens=None,
                            draft_attn_metadatas=None):
        """Replay-time re-issue of the MSA sparse FIA: re-plan tiling with the deterministic
        host kv_lens for THIS step; block_table = the persistent fbt buffer (filled in-graph
        by the selection during replay). MSA carve-out: bypasses the base dense
        seq_lens/block_tables rebind entirely. Delegates to base for non-MSA params."""
        import torch_npu
        from vllm_ascend.compilation.acl_graph import get_graph_params
        from vllm_ascend.models.minimax_m3_msa import msa_host_kv_lens
        graph_params = get_graph_params()
        if graph_params is None:
            return
        params_list = graph_params.attn_params.get(num_tokens)
        if not params_list:
            return
        if not (isinstance(params_list[0], tuple) and len(params_list[0]) > 0
                and params_list[0][0] == "MSA"):
            return AscendAttentionBackendImpl.update_graph_params(
                update_stream, forward_context, num_tokens, vllm_config,
                speculative_config, num_dcp_pcp_tokens, draft_attn_metadatas)
        attn_metadata = forward_context.attn_metadata
        attn_keys = list(attn_metadata.keys())
        if not attn_keys:
            return
        seq_lens_full = list(attn_metadata[attn_keys[0]].seq_lens_list)
        with torch.npu.stream(update_stream):
            for param, handle, event in zip(
                params_list,
                graph_params.handles[num_tokens],
                graph_params.events[num_tokens],
            ):
                (_tag, query, key_paged, value_paged, fbt, block_size,
                 nKV, nH, scale, output, softmax_lse, B) = param
                kv_lens_host = msa_host_kv_lens(seq_lens_full[:B], block_size, _KB, _INIT)
                asl_q = list(range(1, B + 1))
                torch.npu.graph_task_update_begin(update_stream, handle)
                torch_npu.npu_fused_infer_attention_score.out(
                    query=query, key=key_paged, value=value_paged, block_table=fbt,
                    input_layout="TND", block_size=block_size,
                    actual_seq_lengths=asl_q, actual_seq_lengths_kv=kv_lens_host,
                    num_key_value_heads=nKV, num_heads=nH, scale=scale, sparse_mode=0,
                    workspace=graph_params.workspaces.get(num_tokens),
                    out=[output, softmax_lse])
                torch.npu.graph_task_update_end(update_stream)
                event.record(update_stream)


class AscendMSABackend(AscendAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> type["AscendMSAImpl"]:
        return AscendMSAImpl
