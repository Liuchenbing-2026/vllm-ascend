# SPDX-License-Identifier: Apache-2.0
"""Standard-PAGED DSpark speculative proposer for vllm-ascend v0.23.

Original implementation. Plugs the paged DSpark draft (ds_dspark_draft_std) into
the spec-decode framework by subclassing AscendDflashProposer (whose draft-attn
plumbing the paged path needs) and overriding the DSpark-specific steps.

Unlike the eager private-cache proposer, the draft here owns REGISTERED vLLM
attention layers backed by a paged SWA KV cache. Per step:
  set_inputs   : draft query positions (=last_accepted+1+arange), input_ids
                 (bonus token then noise placeholders), context aux+positions,
                 per-gid paged query/context slot maps ; cad.positions re-pointed
                 (iron-law 1); cad.causal=False (non-causal draft gate).
  metadata     : build_for_drafting per draft group -> attaches the non-causal
                 dspark_swa_indices slot table; stashed on the forward context.
  precompute   : draft.precompute_and_store_context_kv(aux, ctx_positions,
                 context_slot_mapping<dict>, ctx_request_slots) -> RoPE'd K into
                 the paged SWA cache at context slots (iron-law 2).
  forward      : draft(input_ids, positions, request_slots, slot_mapping<dict>,
                 block_table<dict>, dspark_query_start_loc/seq_lens/token_to_req)
  sample       : semi-AR Markov loop -> draft token ids [num_reqs, block]
                 (bit-identical to the validated eager _sample_sequential).
"""
from __future__ import annotations

from collections import defaultdict
from copy import copy
from dataclasses import replace
from typing import Any

import torch

from vllm.config import CompilationMode, VllmConfig
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.attention.ds_dspark_meta_std import register_draft_swa_groups

NOISE_FALLBACK = 0


class AscendDSparkPagedProposer(AscendDflashProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner=runner)
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        self.method = "dflash"
        self.parallel_drafting = True
        self.block_size = self.num_speculative_tokens
        self.extra_slots_per_request = self.num_speculative_tokens
        self.net_num_new_slots_per_request = self.num_speculative_tokens
        self.needs_extra_input_slots = True
        self.noise_token_id = int(
            getattr(cfg, "dspark_noise_token_id", None)
            or getattr(cfg, "ptd_token_id", NOISE_FALLBACK) or NOISE_FALLBACK)
        assert self.noise_token_id, "DSpark noise/ptd token id missing (128799 expected)."

        tgt = list(getattr(cfg, "dspark_target_layer_ids", []) or [])
        base_h = vllm_config.speculative_config.draft_model_config.get_hidden_size()
        self.hidden_size = base_h * max(len(tgt), 1)          # 4096*3 = 12288
        self.hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size),
                                         dtype=self.dtype, device=device)
        self._dflash_hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size),
                                                 dtype=self.dtype, device=device)

        qbuf = max(self.max_batch_size * self.num_speculative_tokens, self.max_num_tokens)
        self._qbuf = qbuf
        self.positions = torch.zeros(qbuf, dtype=torch.int32, device=device)
        self._slot_mapping_buffer = torch.zeros(qbuf, dtype=torch.int32, device=device)
        self._request_slots_buffer = torch.zeros(qbuf, dtype=torch.int32, device=device)
        self._context_positions_buffer = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=device)
        self._context_slot_mapping_buffer = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=device)
        self._context_request_slots_buffer = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=device)
        self._token_to_req_buffer = torch.zeros(qbuf, dtype=torch.int32, device=device)
        self.arange_dspark = torch.arange(qbuf + 1, device=device, dtype=torch.int32)
        self._draft_buffer = torch.zeros((self.max_batch_size, self.num_speculative_tokens),
                                         dtype=torch.int64, device=device)
        self._seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        # Stable draft-query metadata handed to the model each step (kernel needs
        # cu_seqlens_q + seqused_kv). Sized max_batch (+1 for the cumulative qsl).
        self._dspark_qsl_buffer = torch.zeros(self.max_batch_size + 1, dtype=torch.int32, device=device)
        self._dspark_seqlen_buffer = torch.zeros(self.max_batch_size, dtype=torch.int32, device=device)

        # per-gid paged tables / slot-mappings (populated each step)
        self._block_tables_by_gid: dict[int, torch.Tensor] = {}
        self._block_tables_by_layer: dict[str, torch.Tensor] = {}
        # The runner delivers each draft SWA group's paged block table + slot map
        # through set_per_group_attn_metadata (model_runner_v1.py: guarded by
        # hasattr(drafter, "set_per_group_attn_metadata")). Without this method the
        # hook is skipped, block_table never reaches the model, and the attention
        # returns zeros (context-blind draft). This is the authoritative source.
        self._dspark_per_group_block_tables: dict[int, torch.Tensor] = {}
        self._dspark_per_group_slot_mappings: dict[int, torch.Tensor] = {}
        self._q_slotmap_by_gid: dict[int, torch.Tensor] = {}
        self._q_slotmap_by_layer: dict[str, torch.Tensor] = {}
        self._ctx_slotmap_by_gid: dict[int, torch.Tensor] = {}
        self._ctx_slotmap_by_layer: dict[str, torch.Tensor] = {}
        self._extra_q_slot_buffers: dict[int, torch.Tensor] = {}
        self._extra_ctx_slot_buffers: dict[int, torch.Tensor] = {}

        # private ring-slot allocation (kept for reset_request_slots bookkeeping)
        sc = getattr(vllm_config, "scheduler_config", None)
        self._max_slots = max(1, int(getattr(sc, "max_num_seqs", self.max_batch_size)
                                     or self.max_batch_size))
        self._req_id_to_slot: dict[str, int] = {}
        self._free_slots = list(range(self._max_slots))
        self._slots_to_reset: list[int] = []
        self._dflash_num_context = 0
        self._runnable = self._run_dspark_model

    # ---- model / backend wiring (paged: draft attn layers DO exist) ----
    def load_model(self, model: torch.nn.Module) -> None:
        # Base load now succeeds because the draft registers real attn layers.
        # Draft stays EAGER this round (rbf16 conc>1 crashes 507011, deferred).
        self.use_cuda_graph = False
        try:
            super().load_model(model)
        finally:
            self.use_cuda_graph = False
        self._runnable = self._run_dspark_model

    def _create_draft_vllm_config(self) -> VllmConfig:
        cfg = super()._create_draft_vllm_config()
        mc = copy(cfg.model_config); mc.enforce_eager = True
        cc = copy(cfg.compilation_config); cc.mode = CompilationMode.NONE
        return replace(cfg, model_config=mc, compilation_config=cc)

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        # Build real AttentionGroups for the 3 draft SWA layers. Sets
        # self.draft_attn_groups / kv_cache_gid / kernel_block_size(=128).
        register_draft_swa_groups(self, kv_cache_config)

    # ---- ring slots (for reset bookkeeping) ----
    def _assign_request_slots(self, bs: int) -> torch.Tensor:
        ib = getattr(self.runner, "input_batch", None)
        if ib is None:
            slots = list(range(bs)); self._slots_to_reset = slots[:]
            return torch.tensor(slots, dtype=torch.int32, device=self.device)
        req_ids = list(ib.req_ids[:bs]); active = set(ib.req_ids[:ib.num_reqs])
        for rid in [r for r in self._req_id_to_slot if r not in active]:
            s = self._req_id_to_slot.pop(rid)
            if s not in self._free_slots:
                self._free_slots.append(s)
        self._free_slots.sort()
        slots, self._slots_to_reset = [], []
        for rid in req_ids:
            if rid not in self._req_id_to_slot:
                if not self._free_slots:
                    raise ValueError("no free DSpark ring slot")
                s = self._free_slots.pop(0)
                self._req_id_to_slot[rid] = s; self._slots_to_reset.append(s)
            slots.append(self._req_id_to_slot[rid])
        return torch.tensor(slots, dtype=torch.int32, device=self.device)

    # ---- paged slot-mapping helpers ----
    def _slot_mapping_from_block_table(self, positions, req_idx, block_table, cache_block_size):
        # slot_id = block_table[req, pos//cbs]*cbs + pos%cbs (cbs = paged --block-size).
        pos = positions.to(torch.long)
        blk = pos // cache_block_size
        off = pos % cache_block_size
        ids = block_table[req_idx].index_select(0, blk)
        return (ids.to(torch.int32) * cache_block_size + off.to(torch.int32))

    def _q_slot_buffer(self, gid: int) -> torch.Tensor:
        if gid == self.kv_cache_gid:
            return self._slot_mapping_buffer
        buf = self._extra_q_slot_buffers.get(gid)
        if buf is None:
            buf = torch.zeros(self._qbuf, dtype=torch.int32, device=self.device)
            self._extra_q_slot_buffers[gid] = buf
        return buf

    def _ctx_slot_buffer(self, gid: int) -> torch.Tensor:
        if gid == self.kv_cache_gid:
            return self._context_slot_mapping_buffer
        buf = self._extra_ctx_slot_buffers.get(gid)
        if buf is None:
            buf = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
            self._extra_ctx_slot_buffers[gid] = buf
        return buf

    def _layer_map(self, gid_map: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for g in self.draft_attn_groups:
            v = gid_map.get(g.kv_cache_group_id)
            if v is not None:
                for name in g.layer_names:
                    out[name] = v
        return out

    def set_per_group_attn_metadata(self, gid: int, block_table, slot_mapping) -> None:
        # Runner hook (model_runner_v1.py:3472-3473). Store the per-group paged
        # block table + slot mapping so _draft_block_tables can source the real
        # draft SWA block table (the missing piece behind attn_out == 0).
        self._dspark_per_group_block_tables[gid] = block_table
        self._dspark_per_group_slot_mappings[gid] = slot_mapping

    @staticmethod
    def _bt_device_tensor(bt, bs: int):
        # v0.23's BlockTable.get_device_tensor takes NO batch_size arg; the old
        # get_device_tensor(bs) call raised -> caught -> None -> block_table never
        # reached the model -> attn returned zeros -> context-blind draft. Mirror
        # #11196: try (bs) then fall back to () .
        for call in (lambda: bt.get_device_tensor(bs), lambda: bt.get_device_tensor()):
            try:
                return call()
            except TypeError:
                continue
            except Exception:
                return None
        return None

    def _draft_block_tables(self, cad, bs: int) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        ib = getattr(getattr(self, "runner", None), "input_batch", None)
        bts = getattr(ib, "block_table", None)
        cad_bt = getattr(cad, "block_table_tensor", None)
        pg = self._dspark_per_group_block_tables
        # If exactly one per-group table was delivered by the runner hook, it IS
        # the (single) draft SWA group's -- use it even if the gid key differs.
        lone_pg = next(iter(pg.values())) if len(pg) == 1 else None
        for g in self.draft_attn_groups:
            gid = g.kv_cache_group_id
            src = None
            # 1) authoritative: what the runner delivered via set_per_group_attn_metadata
            bt = pg.get(gid)
            if bt is not None:
                src = "hook"
            # 2) runner input_batch per-gid table
            if bt is None and bts is not None:
                try:
                    dbt = bts[gid]
                except (IndexError, KeyError, TypeError):
                    dbt = None
                if dbt is not None:
                    bt = self._bt_device_tensor(dbt, bs); src = "runner"
            # 3) single-group fallback: the sole delivered table
            if bt is None and lone_pg is not None:
                bt = lone_pg; src = "lone"
            # 4) last resort: the draft cad's block table
            if bt is None and cad_bt is not None:
                bt = cad_bt; src = "cad"
            if bt is not None:
                out[gid] = bt[:bs]
        return out

    # ---- build draft inputs ----
    def set_inputs_first_pass(self, target_token_ids, next_token_ids, target_positions,
                              target_hidden_states, token_indices_to_sample, cad,
                              num_rejected_tokens_gpu, **kw):
        del target_token_ids, token_indices_to_sample
        bs = int(cad.num_reqs); block = self.num_speculative_tokens; nq = bs * block
        req_slots = self._assign_request_slots(bs)
        self._seed_buffer[:bs].copy_(next_token_ids[:bs].to(torch.int64))
        self._block_tables_by_gid = self._draft_block_tables(cad, bs)
        self._block_tables_by_layer = self._layer_map(self._block_tables_by_gid)
        primary = self.kv_cache_gid

        # --- context: aux hidden + REAL target positions + paged slots (iron-law 2) ---
        cursor = 0
        for r in range(bs):
            a = int(cad.query_start_loc[r].item()); b = int(cad.query_start_loc[r + 1].item())
            n = b - a
            if n <= 0:
                continue
            e = cursor + n
            self._dflash_hidden_states[cursor:e] = target_hidden_states[a:b]
            self._context_positions_buffer[cursor:e] = target_positions[a:b]
            self._context_request_slots_buffer[cursor:e] = int(req_slots[r])
            for g in self.draft_attn_groups:
                gid = g.kv_cache_group_id; bt = self._block_tables_by_gid.get(gid)
                if bt is None:
                    continue
                self._ctx_slot_buffer(gid)[cursor:e] = self._slot_mapping_from_block_table(
                    target_positions[a:b], r, bt, int(g.kv_cache_spec.block_size))
            cursor = e
        self._dflash_num_context = cursor
        self._ctx_slotmap_by_gid = {g.kv_cache_group_id: self._ctx_slot_buffer(g.kv_cache_group_id)[:cursor]
                                    for g in self.draft_attn_groups}
        self._ctx_slotmap_by_layer = self._layer_map(self._ctx_slotmap_by_gid)

        # --- query block (anchor + noise placeholders) ---
        max_len = int(getattr(self.vllm_config.model_config, "max_model_len", 0) or 0)
        eff_seq = cad.seq_lens
        if num_rejected_tokens_gpu is not None:
            eff_seq = eff_seq - num_rejected_tokens_gpu
        next_seq = eff_seq + block
        if max_len > 0:
            next_seq = next_seq.clamp(max=max_len)
        for r in range(bs):
            a = int(cad.query_start_loc[r].item()); b = int(cad.query_start_loc[r + 1].item())
            if num_rejected_tokens_gpu is not None:
                b -= int(num_rejected_tokens_gpu[r].item())
            last_pos = target_positions[b - 1]
            s, e = r * block, (r + 1) * block
            dpos = last_pos + 1 + self.arange_dspark[:block]
            if max_len > 0:
                over = dpos >= max_len
                dpos = torch.where(over, torch.zeros_like(dpos), dpos)
            else:
                over = torch.zeros(block, dtype=torch.bool, device=dpos.device)
            self.positions[s:e] = dpos
            self.input_ids[s] = next_token_ids[r]
            if block > 1:
                self.input_ids[s + 1:e] = self.noise_token_id
            self._request_slots_buffer[s:e] = int(req_slots[r])
            self._token_to_req_buffer[s:e] = r
            for g in self.draft_attn_groups:
                gid = g.kv_cache_group_id; bt = self._block_tables_by_gid.get(gid)
                if bt is None:
                    continue
                sm = self._slot_mapping_from_block_table(dpos, r, bt, int(g.kv_cache_spec.block_size))
                sm.masked_fill_(over, -1)
                self._q_slot_buffer(gid)[s:e] = sm
        self._q_slotmap_by_gid = {g.kv_cache_group_id: self._q_slot_buffer(g.kv_cache_group_id)
                                  for g in self.draft_attn_groups}
        self._q_slotmap_by_layer = self._layer_map(self._q_slotmap_by_gid)

        # --- rewrite cad for the draft block (iron-law 1) ---
        cad.query_start_loc = self.arange_dspark[:bs + 1] * block
        cad.seq_lens = next_seq
        cad.num_actual_tokens = nq
        cad.num_input_tokens = nq
        cad.max_query_len = block
        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [block] * bs
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = block
        if primary in self._q_slotmap_by_gid:
            cad.slot_mapping = self._q_slotmap_by_gid[primary][:nq]
        cad.positions = self.positions[:nq]                     # iron-law 1: DSA q RoPE source
        cad.causal = False                                      # non-causal draft gate
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill
        cad.token_to_req_indices = self._token_to_req_buffer[:nq]

        # stable draft-query metadata for the model (kernel cu_seqlens_q / seqused_kv)
        self._dspark_qsl_buffer[:bs + 1] = self.arange_dspark[:bs + 1] * block
        self._dspark_seqlen_buffer[:bs] = next_seq[:bs].to(torch.int32)
        # TODO(serve-verify): once the draft graph is enabled, also copy
        #   query_start_loc/seq_lens(+cpu) into graph-safe buffers and repoint cad.
        token_idx = torch.arange(nq, dtype=torch.int32, device=self.device)
        return nq, token_idx, cad, None

    @staticmethod
    def _group_metadata_builder(group):
        # vllm AttentionGroup exposes get_metadata_builder() in some versions and
        # a metadata_builders list in others; accept either.
        getter = getattr(group, "get_metadata_builder", None)
        if callable(getter):
            return getter()
        return group.metadata_builders[0]

    # ---- swa metadata: build per-layer + attach dspark_swa_indices ----
    def _build_dsa_metadata(self, cad, num_input_tokens, num_actual_tokens):
        if not self.draft_attn_groups:
            return []
        if num_input_tokens > num_actual_tokens:
            self.positions[num_actual_tokens:num_input_tokens].fill_(0)
            for g in self.draft_attn_groups:
                self._q_slot_buffer(g.kv_cache_group_id)[num_actual_tokens:num_input_tokens].fill_(-1)
            self.input_ids[num_actual_tokens:num_input_tokens].fill_(self.noise_token_id)
            self._token_to_req_buffer[num_actual_tokens:num_input_tokens].fill_(-1)
        cad.positions = self.positions[:num_input_tokens]
        cad.num_input_tokens = num_input_tokens
        cad.num_actual_tokens = num_actual_tokens
        cad.causal = False
        cad.attn_state = AscendAttentionState.ChunkedPrefill
        cad.token_to_req_indices = self._token_to_req_buffer[:num_input_tokens]
        per_layer: dict[str, Any] = {}
        for g in self.draft_attn_groups:
            gid = g.kv_cache_group_id
            cm = copy(cad)
            bt = self._block_tables_by_gid.get(gid)
            if bt is not None:
                cm.block_table_tensor = bt[:cm.num_reqs]
            sm = self._q_slotmap_by_gid.get(gid)
            if sm is not None:
                cm.slot_mapping = sm[:num_input_tokens]
            # builder computes non-causal dspark_swa_indices/lens (== ori_sparse_indices).
            builder = self._group_metadata_builder(g)
            md = builder.build_for_drafting(
                cm, draft_index=1, block_size=g.kv_cache_spec.block_size)
            for name in g.layer_names:
                per_layer[name] = md
        return [per_layer]

    # ---- context-KV precompute into paged cache (iron-law 2) ----
    def _precompute_context_kv(self) -> None:
        n = self._dflash_num_context
        if self._slots_to_reset:
            self.model.reset_request_slots(
                torch.tensor(self._slots_to_reset, dtype=torch.int32, device=self.device))
        ctx_slotmap = self._ctx_slotmap_by_layer if self._ctx_slotmap_by_layer \
            else self._context_slot_mapping_buffer[:n]
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:n],
            self._context_positions_buffer[:n],
            ctx_slotmap,
            self._context_request_slots_buffer[:n])

    def build_model_inputs_first_pass(self, n: int) -> dict[str, Any]:
        self._precompute_context_kv()
        q_slotmap = {name: v[:n] for name, v in self._q_slotmap_by_layer.items()} \
            if self._q_slotmap_by_layer else self._slot_mapping_buffer[:n]
        bs = max(int(n) // self.num_speculative_tokens, 0)
        return dict(
            input_ids=self.input_ids[:n],
            positions=self.positions[:n],
            inputs_embeds=None,
            request_slots=self._request_slots_buffer[:n],
            slot_mapping=q_slotmap,
            block_table=self._block_tables_by_layer or None,
            dspark_query_start_loc=self._dspark_qsl_buffer[:bs + 1],
            dspark_seq_lens=self._dspark_seqlen_buffer[:bs],
            dspark_token_to_req_indices=self._token_to_req_buffer[:n],
        )

    def _run_dspark_model(self, num_input_tokens: int, **kw) -> torch.Tensor:
        del kw
        return self.model(**self.build_model_inputs_first_pass(num_input_tokens))

    # ---- semi-AR Markov sampling (REUSE validated eager math) ----
    def _sample_sequential(self, num_reqs, head_hidden, token_indices, sampling_metadata=None):
        block = self.num_speculative_tokens
        ns = num_reqs * block
        sample_hidden = head_hidden[token_indices[:ns]]
        base = self.model.compute_logits(sample_hidden)
        vocab = base.shape[-1]
        base = base.view(num_reqs, block, vocab)
        prev = self._seed_buffer[:num_reqs]
        for i in range(block):
            emb = self.model.markov_embed(prev)
            bias = self.model.markov_bias(emb)
            self._draft_buffer[:num_reqs, i] = (base[:, i, :] + bias).argmax(dim=-1)
            prev = self._draft_buffer[:num_reqs, i]
        return self._draft_buffer[:num_reqs, :block]

    @torch.inference_mode()
    def dummy_run(self, num_tokens, num_reqs: int = 0, **kw) -> None:
        del kw
        block = self.num_speculative_tokens
        if not num_reqs:
            num_reqs = max(1, min(int(num_tokens) // block, self.max_batch_size))
        nq = min(num_reqs * block, int(self.positions.numel()))
        self.positions[:nq].fill_(0)
        self.input_ids[:nq].fill_(self.noise_token_id)
        self._request_slots_buffer[:nq].fill_(0)
        self._token_to_req_buffer[:nq].fill_(-1)
        self._dflash_num_context = 0
        self._slots_to_reset = [0]
        # TODO(serve-verify): build a minimal paged block table + dsa metadata so
        #   the draft attn layers get a valid (empty-context) swa slot table during
        #   memory profiling; without it _paged_attend has no block_table/kv_cache.
        with set_ascend_forward_context(None, self.vllm_config, num_tokens=nq,
                                        num_actual_tokens=nq, is_draft_model=True):
            fc = get_forward_context()
            if fc is not None:
                fc.moe_layer_index = 0
            self._run_dspark_model(num_input_tokens=nq)

    # ---- orchestration ----
    def _propose(self, target_token_ids, target_positions, target_hidden_states,
                 next_token_ids, token_indices_to_sample, common_attn_metadata,
                 target_model_batch_desc, sampling_metadata, mm_embed_inputs=None,
                 num_rejected_tokens_gpu=None, **kw):
        num_reqs = common_attn_metadata.num_reqs
        nq, token_idx, cad, _ = self.set_inputs_first_pass(
            target_token_ids, next_token_ids, target_positions, target_hidden_states,
            token_indices_to_sample, common_attn_metadata, num_rejected_tokens_gpu)
        num_input_tokens = nq   # draft eager this round -> no graph padding
        md = self._build_dsa_metadata(cad, num_input_tokens, nq)
        with set_ascend_forward_context(md[0] if md else None, self.vllm_config,
                                        num_tokens=num_input_tokens, num_actual_tokens=nq,
                                        is_draft_model=True, draft_attn_metadatas=md):
            fc = get_forward_context()
            if fc is not None:
                fc.moe_layer_index = 0
            head_hidden = self._run_dspark_model(num_input_tokens=num_input_tokens)
            draft = self._sample_sequential(num_reqs, head_hidden, token_idx, sampling_metadata)
        return draft
