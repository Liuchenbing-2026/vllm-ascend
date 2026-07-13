from types import SimpleNamespace

import torch

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


def _metadata(
    *,
    internal_cpu: torch.Tensor,
    public_cpu: torch.Tensor | None,
    upper_bound: torch.Tensor | None,
):
    return SimpleNamespace(
        num_reqs=2,
        _seq_lens_cpu=internal_cpu,
        seq_lens_cpu=public_cpu,
        seq_lens_cpu_upper_bound=upper_bound,
        parallel_drafting_seq_lens_cpu_valid=False,
    )


def test_markov_seq_lens_cpu_subtracts_current_rejections_once():
    proposer = AscendDflashProposer.__new__(AscendDflashProposer)
    proposer.uses_markov_head = True
    original = torch.tensor([105, 207], dtype=torch.int32)
    public = original.clone()
    upper_bound = original.clone()
    metadata = _metadata(internal_cpu=original, public_cpu=public, upper_bound=upper_bound)

    proposer._update_markov_seq_lens_cpu(
        metadata,
        num_query_per_req=5,
        num_rejected_tokens_gpu=torch.tensor([2, 0]),
        num_rejected_tokens_cpu=torch.tensor([2, 0]),
    )

    assert metadata.parallel_drafting_seq_lens_cpu_valid
    assert metadata._seq_lens_cpu.tolist() == [108, 212]
    assert metadata._seq_lens_cpu.dtype == torch.int32
    assert metadata.seq_lens_cpu.tolist() == [108, 212]
    assert metadata.seq_lens_cpu_upper_bound.tolist() == [110, 212]
    assert original.tolist() == [105, 207]
    assert public.tolist() == [105, 207]


def test_markov_seq_lens_cpu_missing_current_counts_keeps_device_fallback():
    proposer = AscendDflashProposer.__new__(AscendDflashProposer)
    proposer.uses_markov_head = True
    original = torch.tensor([105, 207], dtype=torch.int64)
    metadata = _metadata(internal_cpu=original, public_cpu=None, upper_bound=original.clone())

    proposer._update_markov_seq_lens_cpu(
        metadata,
        num_query_per_req=5,
        num_rejected_tokens_gpu=torch.tensor([2, 0]),
        num_rejected_tokens_cpu=None,
    )

    assert not metadata.parallel_drafting_seq_lens_cpu_valid
    assert metadata._seq_lens_cpu is original
    assert metadata.seq_lens_cpu is None


def test_plain_dflash_does_not_opt_into_cpu_fast_path():
    proposer = AscendDflashProposer.__new__(AscendDflashProposer)
    proposer.uses_markov_head = False
    original = torch.tensor([105, 207], dtype=torch.int64)
    metadata = _metadata(internal_cpu=original, public_cpu=None, upper_bound=original.clone())

    proposer._update_markov_seq_lens_cpu(
        metadata,
        num_query_per_req=5,
        num_rejected_tokens_gpu=torch.tensor([2, 0]),
        num_rejected_tokens_cpu=torch.tensor([2, 0]),
    )

    assert not metadata.parallel_drafting_seq_lens_cpu_valid
    assert metadata._seq_lens_cpu is original


def test_full_graph_dp_padding_aligns_exact_cpu_mirror_with_device_batch():
    proposer = AscendDflashProposer.__new__(AscendDflashProposer)
    internal = torch.tensor([108, 212], dtype=torch.int32)
    public = internal.clone()
    upper_bound = torch.tensor([110, 212], dtype=torch.int32)
    metadata = _metadata(
        internal_cpu=internal,
        public_cpu=public,
        upper_bound=upper_bound,
    )
    metadata.parallel_drafting_seq_lens_cpu_valid = True

    proposer._adjust_exact_parallel_drafting_seq_lens_cpu(
        metadata,
        desired_size=3,
    )

    assert metadata._seq_lens_cpu.tolist() == [108, 212, 0]
    assert metadata.seq_lens_cpu.tolist() == [108, 212, 0]
    assert metadata.seq_lens_cpu_upper_bound.tolist() == [110, 212, 0]
    assert metadata._seq_lens_cpu.dtype == torch.int32
