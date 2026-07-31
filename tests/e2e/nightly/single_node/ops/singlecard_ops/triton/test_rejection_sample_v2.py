"""Tests for the MRV2 NPU rejection sampler
(vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils.rejection_sample).

Guards the non-greedy probability-ratio test: the kernel draws the uniform u
on-device via tl.rand(seed, pos), so acceptance must follow
p_target(draft) > u * q_draft(draft). A regression to a degenerate u (e.g. the
former hardcoded u=0.0) would unconditionally accept every draft token and
break the losslessness of speculative decoding.
"""

import gc

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

DEVICE = "npu"
NEG_INF = -1e9


def _run_rejection_sample(target_logits: torch.Tensor, draft_tokens: torch.Tensor, num_reqs: int, num_steps: int):
    """Drive rejection_sample with the MRV2 caller's tensor layout.

    Per request there are num_steps + 1 logits rows (one per draft token plus
    the bonus position); draft_sampled mirrors input_ids[logits_indices], so
    the draft token verified by row start + i sits at start + i + 1.
    """
    tokens_per_req = num_steps + 1
    num_logits = num_reqs * tokens_per_req
    assert target_logits.shape[0] == num_logits

    draft_sampled = torch.zeros(num_logits, dtype=torch.int32, device=DEVICE)
    draft_sampled.view(num_reqs, tokens_per_req)[:, 1:] = draft_tokens

    cu_num_logits = torch.arange(num_reqs + 1, dtype=torch.int32, device=DEVICE) * tokens_per_req
    pos = torch.arange(num_logits, dtype=torch.int32, device=DEVICE)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = idx_mapping.repeat_interleave(tokens_per_req)
    expanded_local_pos = torch.arange(tokens_per_req, dtype=torch.int32, device=DEVICE).repeat(num_reqs)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(12345, 12345 + num_reqs, dtype=torch.int64, device=DEVICE)

    sampled, num_sampled = rejection_sample(
        target_logits,
        None,  # draft_logits: one-hot draft, q(draft_token) = 1
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_steps,
    )
    torch.npu.synchronize()
    return sampled, num_sampled


def _make_inputs(num_reqs: int, num_steps: int, vocab_size: int, draft_target_prob: float):
    """Target logits giving the draft token draft_target_prob mass, with the
    remainder on a single disjoint token (so acceptance prob == the given
    value under the one-hot draft distribution)."""
    generator = torch.Generator().manual_seed(0)
    draft_tokens = torch.randint(0, vocab_size // 2, (num_reqs, num_steps), generator=generator, dtype=torch.int32).to(
        DEVICE
    )

    num_logits = num_reqs * (num_steps + 1)
    target_logits = torch.full((num_logits, vocab_size), NEG_INF, dtype=torch.float32, device=DEVICE)
    rows = target_logits.view(num_reqs, num_steps + 1, vocab_size)
    other_token = vocab_size - 1  # disjoint from draft token range
    if draft_target_prob < 1.0:
        log_ratio = torch.tensor(draft_target_prob / (1.0 - draft_target_prob)).log().item()
    else:
        log_ratio = -NEG_INF
    rows[:, :, other_token] = 0.0
    rows[:, :num_steps, :].scatter_(
        2,
        draft_tokens.unsqueeze(-1).long(),
        torch.full_like(draft_tokens, log_ratio, dtype=torch.float32).unsqueeze(-1),
    )
    return target_logits, draft_tokens


@torch.inference_mode()
def test_certain_drafts_all_accepted():
    """p_target(draft) == 1 => ratio test passes for every u in (0, 1).

    num_sampled includes the resampled/bonus token (_insert_resampled_kernel
    stores accepted + 1), so all-accepted means num_steps + 1."""
    num_reqs, num_steps, vocab_size = 8, 3, 2048
    target_logits, draft_tokens = _make_inputs(num_reqs, num_steps, vocab_size, draft_target_prob=1.0)

    sampled, num_sampled = _run_rejection_sample(target_logits, draft_tokens, num_reqs, num_steps)

    assert torch.equal(num_sampled, torch.full((num_reqs,), num_steps + 1, dtype=num_sampled.dtype, device=DEVICE))
    assert torch.equal(sampled[:, :num_steps], draft_tokens.to(sampled.dtype))
    gc.collect()
    torch.npu.empty_cache()


@torch.inference_mode()
def test_improbable_drafts_all_rejected():
    """p_target(draft) ~= 1e-13 => acceptance requires u < 1e-13, which the
    fixed seeds never produce, so num_sampled == 1 (just the resampled token
    at the first rejected step). The former hardcoded u=0.0 accepted every
    draft here (log(0) = -inf => num_sampled == num_steps + 1), so this is
    the regression guard for that bug."""
    num_reqs, num_steps, vocab_size = 8, 3, 2048
    target_logits, draft_tokens = _make_inputs(num_reqs, num_steps, vocab_size, draft_target_prob=1e-13)

    _, num_sampled = _run_rejection_sample(target_logits, draft_tokens, num_reqs, num_steps)

    assert torch.equal(num_sampled, torch.ones(num_reqs, dtype=num_sampled.dtype, device=DEVICE))
    gc.collect()
    torch.npu.empty_cache()


@torch.inference_mode()
def test_half_prob_drafts_acceptance_rate():
    """p_target(draft) == 0.5 => E[num_sampled] = 1 + sum_{i=1..k} 0.5^i
    ~= 1.94 for k=4. Loose bounds reject both degenerate extremes (u ~ 0
    accepts all -> mean 5.0; u ~ 1 rejects all -> mean 1.0) without being
    seed-sensitive."""
    num_reqs, num_steps, vocab_size = 128, 4, 2048
    target_logits, draft_tokens = _make_inputs(num_reqs, num_steps, vocab_size, draft_target_prob=0.5)

    sampled, num_sampled = _run_rejection_sample(target_logits, draft_tokens, num_reqs, num_steps)

    mean_sampled = num_sampled.float().mean().item()
    assert 1.5 < mean_sampled < 2.4, f"mean sampled tokens {mean_sampled} outside expected range"

    # Accepted prefixes (all but the final resampled/bonus token) must be the
    # draft tokens themselves.
    num_sampled_cpu = num_sampled.cpu()
    sampled_cpu = sampled.cpu()
    draft_cpu = draft_tokens.cpu().to(sampled_cpu.dtype)
    for r in range(num_reqs):
        n = int(num_sampled_cpu[r]) - 1
        assert torch.equal(sampled_cpu[r, :n], draft_cpu[r, :n])
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
