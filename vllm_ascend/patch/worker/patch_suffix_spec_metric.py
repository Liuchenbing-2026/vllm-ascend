#
# Patch: Correct acceptance rate metric for suffix/ngram speculative decoding
# with padded proposals.
#
# The suffix proposer pads draft proposals to num_speculative_tokens with
# vocab_size tokens for FULL_DECODE_ONLY graph capture.  These padding
# tokens are always rejected but are counted in num_draft_tokens by the
# scheduler, which inflates the denominator and makes the acceptance rate
# appear lower than it actually is.
#
# This patch wraps Scheduler.update_from_output to detect vocab_size
# padding tokens in scheduled_spec_decode_tokens and include them in
# num_invalid_spec_tokens.  make_spec_decoding_stats then correctly
# excludes them from the acceptance rate metric, while the KV-cache
# correction (num_computed_tokens -= num_rejected) remains unaffected
# because it is computed BEFORE make_spec_decoding_stats is called.
#
# Why not patch update_draft_token_ids_in_output (the previous approach)?
#   That method is ONLY called in the async + structured-output (deferred
#   sampling) path.  For sync scheduling or async-without-structured-output,
#   it is never invoked, so the metric stays diluted.  Patching
#   update_from_output covers ALL scheduling paths.
#

from vllm.v1.core.sched.scheduler import Scheduler

_orig_update_from_output = Scheduler.update_from_output


def _patched_update_from_output(self, scheduler_output, model_runner_output):
    # Before the original logic runs, count vocab_size padding tokens
    # in the scheduled spec decode tokens and register them as invalid.
    # This ensures make_spec_decoding_stats subtracts them from
    # num_draft_tokens for the acceptance rate metric, without touching
    # the num_rejected / num_computed_tokens KV-cache correction.
    sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    if sched_spec_tokens:
        vocab_size = self.vllm_config.model_config.get_vocab_size()
        num_invalid = scheduler_output.num_invalid_spec_tokens
        changed = False

        for req_id, spec_token_ids in sched_spec_tokens.items():
            padding_count = sum(1 for t in spec_token_ids
                                if t >= vocab_size)
            if padding_count > 0:
                if num_invalid is None:
                    num_invalid = {}
                num_invalid[req_id] = (num_invalid.get(req_id, 0)
                                       + padding_count)
                changed = True

        if changed:
            scheduler_output.num_invalid_spec_tokens = num_invalid

    return _orig_update_from_output(self, scheduler_output,
                                    model_runner_output)


Scheduler.update_from_output = _patched_update_from_output
