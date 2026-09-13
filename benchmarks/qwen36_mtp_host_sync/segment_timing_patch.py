"""Patch + instrument model_runner_v1.py.  Usage: patch.py {probe|A|AB}

Rounds 3-11 localised the whole MTP step regression to two blocking host waits at the
top of _prepare_inputs, both waiting on the same device milestone (step N's sampling):

    A  :1103-1104  num_accepted_tokens_event.synchronize()            28.92 ms measured
    B  :1243-1244  -> _correct_optimistic_seq_lens_cpu -> :1451
                   valid_sampled_token_count_event.synchronize()      ~0, masked by A

Measured step 55.03 ms vs device 35.65 ms: host and device do not overlap.

mode probe : instrumentation only, nothing else changes. Establishes the paired
             baseline and, via the new seq_sync segment, shows B reading ~0 today.
mode A     : replace A's host round-trip with the device-side equivalent.
             The block's entire semantics is one permutation,
                 out[i] = src[prev_positions[i]] if prev_positions[i] >= 0 else 1
             with the tail filled with 1. Every LIVE consumer of the result is already
             a device tensor (:1148 update_num_computed_tokens_for_batch_change,
             :3090 the GDN builder); the only reader of the .np mirror is the
             mamba_cache_mode == "align" branch at :2014-2021, which is dead here
             (mode is "none"), and the patch is gated on that anyway.
             CRITICAL: _update_states_after_model_execute writes
             num_accepted_tokens.gpu on global_stream() (:2392-2399), and
             global_stream() is never joined back to the default stream (it appears
             exactly twice in the file). The .synchronize() being removed was the ONLY
             cross-stream ordering edge, so it is replaced by a non-blocking
             current_stream().wait_event(...) -- without that this is a write-write
             race that would silently feed wrong accepted counts to 30 GDN layers.
mode AB    : additionally skip B. This is LOSSY by construction: _seq_lens_cpu feeds
             FIA's actual_seq_lengths_kv (attention_v1.py:306-309 -> :831), so leaving
             optimistic_seq_lens_cpu uncorrected overstates each request's KV length by
             exactly this step's rejection count. Ship only if the greedy byte-exact
             check passes and acceptance length holds.
"""
import sys

PATH = "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
MODE = sys.argv[1] if len(sys.argv) > 1 else "probe"
assert MODE in ("probe", "A", "AB"), MODE

HELPER = '''

# ---- NTSEG probe v11 ------------------------------------------------------
from time import perf_counter as _nt_pc


class _NTSegStats:
    KEYS = ("prolog", "sync_ip", "upd_st", "prep_in", "acc_sync", "seq_sync",
            "fwd", "sample", "bookkeep", "draft", "fwd_dev", "draft_dev")

    def __init__(self):
        self.acc = {"d": {}, "p": {}}
        self.wall = {"d": 0.0, "p": 0.0}
        self.nsteps = {"d": 0, "p": 0}
        self.last = None
        self.last_b = None
        self.b = "d"
        self.evs = {}

    def add(self, k, dt):
        a = self.acc[self.b]
        a[k] = a.get(k, 0.0) + dt
        a[k + "_n"] = a.get(k + "_n", 0) + 1

    def tick(self, num_tokens, num_reqs):
        self.b = "d" if (num_tokens is not None and num_tokens <= 512) else "p"
        t = _nt_pc()
        if self.last is not None and self.last_b is not None:
            self.wall[self.last_b] += t - self.last
            if self.last_b == self.b:
                a = self.acc[self.b]
                a["step"] = a.get("step", 0.0) + (t - self.last)
                a["step_n"] = a.get("step_n", 0) + 1
        self.last = t
        self.last_b = self.b
        self.nsteps[self.b] += 1
        if (self.nsteps["d"] + self.nsteps["p"]) % 200 == 0:
            out = []
            for b in ("d", "p"):
                a = self.acc[b]
                parts = []
                for k in ("step",) + self.KEYS:
                    c = a.get(k + "_n", 0)
                    if c:
                        parts.append("%s=%.2f/%d" % (k, a[k] * 1000.0 / c, c))
                if parts:
                    out.append("%s[%s]" % (b, " ".join(parts)))
            logger.info("NTSEG4 nd=%d np=%d ntok=%s nreq=%s | %s",
                        self.nsteps["d"], self.nsteps["p"], num_tokens, num_reqs,
                        " ".join(out))
            self.acc = {"d": {}, "p": {}}
            self.wall = {"d": 0.0, "p": 0.0}

    def dev_begin(self, k):
        try:
            import torch as _t
            key = (self.b, k)
            rec = self.evs.get(key)
            if rec is None:
                rec = [_t.npu.Event(enable_timing=True),
                       _t.npu.Event(enable_timing=True), False]
                self.evs[key] = rec
            if rec[2] and rec[1].query():
                self.add(k + "_dev", rec[0].elapsed_time(rec[1]) / 1000.0)
                rec[2] = False
            if not rec[2]:
                rec[0].record()
        except Exception:
            pass

    def dev_end(self, k):
        try:
            rec = self.evs.get((self.b, k))
            if rec is not None and not rec[2]:
                rec[1].record()
                rec[2] = True
        except Exception:
            pass


_NT = _NTSegStats()
# ---- end NTSEG probe ------------------------------------------------------
'''

ACC_ORIG = '''        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
            # Async mode: condense() reordered indices, use prev_positions mapping
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # Non-async mode: use values directly
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()'''

ACC_PROBE = ACC_ORIG.replace(
    '''        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()''',
    '''        if self.num_accepted_tokens_event is not None:
            _nt_ac0 = _nt_pc()
            self.num_accepted_tokens_event.synchronize()
            _NT.add("acc_sync", _nt_pc() - _nt_ac0)''')

ACC_FAST = '''        if self.num_accepted_tokens_event is not None:
            _nt_ac0 = _nt_pc()
            _nt_fast = (
                self.use_async_scheduling
                and prev_req_id_to_index
                and self.cache_config.mamba_cache_mode != "align"
            )
            if _nt_fast:
                # Device-side equivalent of the host round-trip this replaces:
                #     out[i] = src[prev[i]] if prev[i] >= 0 else 1,  tail = 1
                # Every live consumer of the result is already a device tensor
                # (:1148, :3090); only the dead align branch reads the .np mirror.
                # The removed .synchronize() was the ONLY ordering edge against
                # _update_states_after_model_execute, which writes this tensor on
                # global_stream() and never joins back -- hence the wait_event.
                torch.npu.current_stream().wait_event(self.num_accepted_tokens_event)
                if prev_positions_gpu is None:
                    self.prev_positions.copy_to_gpu(num_reqs)
                    prev_positions_gpu = self.prev_positions.gpu[:num_reqs]
                _nt_idx = prev_positions_gpu
                _nt_src = self.num_accepted_tokens.gpu.clone()
                _nt_g = _nt_src.index_select(0, _nt_idx.clamp_min(0).long())
                self.num_accepted_tokens.gpu[:num_reqs] = torch.where(
                    _nt_idx >= 0, _nt_g, torch.ones_like(_nt_g)
                )
                self.num_accepted_tokens.gpu[num_reqs:].fill_(1)
            else:
                self.num_accepted_tokens_event.synchronize()
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.np[num_reqs:].fill(1)
                self.num_accepted_tokens.copy_to_gpu()
            _NT.add("acc_sync", _nt_pc() - _nt_ac0)'''

SEQ_ORIG = '''        if self._needs_seq_lens_cpu_sync and async_spec_decode_active:
            self._correct_optimistic_seq_lens_cpu(num_reqs)'''

SEQ_TIMED = '''        if self._needs_seq_lens_cpu_sync and async_spec_decode_active:
            _nt_sl0 = _nt_pc()
            self._correct_optimistic_seq_lens_cpu(num_reqs)
            _NT.add("seq_sync", _nt_pc() - _nt_sl0)'''

SEQ_OFF = '''        if False and self._needs_seq_lens_cpu_sync and async_spec_decode_active:
            _nt_sl0 = _nt_pc()
            self._correct_optimistic_seq_lens_cpu(num_reqs)
            _NT.add("seq_sync", _nt_pc() - _nt_sl0)'''

EDITS = [
    (  # target forward: tick, prologue close-out, host dispatch, device span
        """            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )""",
        """            _NT.tick(num_tokens_padded, getattr(batch_desc, "num_reqs", None))
            _NT.add("prolog", _nt_pc() - getattr(self, "_nt_em0", _nt_pc()))
            _NT.dev_begin("fwd")
            _nt_t0 = _nt_pc()
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )
            _NT.add("fwd", _nt_pc() - _nt_t0)
            _NT.dev_end("fwd")""",
    ),
    (
        """    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:""",
        """    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        self._nt_em0 = _nt_pc()""",
    ),
    (
        """        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():""",
        """        with record_function_or_nullcontext("prepare input"):
            _nt_s0 = _nt_pc()
            with self.synchronize_input_prep():
                _NT.add("sync_ip", _nt_pc() - _nt_s0)""",
    ),
    (
        """                deferred_state_corrections_fn = self._update_states(
                    scheduler_output
                )""",
        """                _nt_u0 = _nt_pc()
                deferred_state_corrections_fn = self._update_states(
                    scheduler_output
                )
                _NT.add("upd_st", _nt_pc() - _nt_u0)
                self._nt_pi0 = _nt_pc()""",
    ),
    (
        """                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )""",
        """                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )
                _NT.add("prep_in", _nt_pc() - getattr(self, "_nt_pi0", _nt_pc()))""",
    ),
    (
        """        with record_function_or_nullcontext("sample_token"):
            sampler_output = self._sample(logits, spec_decode_metadata)""",
        """        with record_function_or_nullcontext("sample_token"):
            _nt_t0 = _nt_pc()
            sampler_output = self._sample(logits, spec_decode_metadata)
            _NT.add("sample", _nt_pc() - _nt_t0)""",
    ),
    (
        """        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(""",
        """        _nt_tb = _nt_pc()
        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(""",
    ),
    (
        """            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        with record_function_or_nullcontext("draft_token"):""",
        """            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )
        _NT.add("bookkeep", _nt_pc() - _nt_tb)

        with record_function_or_nullcontext("draft_token"):""",
    ),
    (
        """                if use_padded_batch and not early_pp_padded_drafter:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)""",
        """                if use_padded_batch and not early_pp_padded_drafter:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    _NT.dev_begin("draft")
                    _nt_t0 = _nt_pc()
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                    _NT.add("draft", _nt_pc() - _nt_t0)
                    _NT.dev_end("draft")""",
    ),
    (ACC_ORIG, ACC_PROBE if MODE == "probe" else ACC_FAST),
    (SEQ_ORIG, SEQ_OFF if MODE == "AB" else SEQ_TIMED),
]

src = open(PATH, encoding="utf-8").read()
if "NTSEG probe v11" in src:
    print("ALREADY_PATCHED_ABORT")
    sys.exit(2)

LOGGER_ANCHOR = "from vllm.logger import logger"
i = src.find(LOGGER_ANCHOR)
if i < 0:
    print("LOGGER_ANCHOR_NOT_FOUND")
    sys.exit(3)
i += len(LOGGER_ANCHOR)
src = src[:i] + HELPER + src[i:]

for n, (old, new) in enumerate(EDITS, 1):
    if src.count(old) != 1:
        print("EDIT_%d_COUNT=%d_ABORT" % (n, src.count(old)))
        sys.exit(4)
    src = src.replace(old, new)

open(PATH, "w", encoding="utf-8").write(src)
print("PATCHED mode=%s edits=%d" % (MODE, len(EDITS)))
