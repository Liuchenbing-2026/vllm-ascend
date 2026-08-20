# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

from vllm_ascend.spec_decode.gemma4_proposer import AscendGemma4Proposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def _vllm_config(*, gemma4: bool = False, step3p5: bool = False):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            use_gemma4_mtp=lambda: gemma4,
            use_step3p5_mtp=lambda: step3p5,
        )
    )


def test_only_gemma4_opts_into_per_group_kv_groups():
    """Gemma4's draft layers span a sliding-window and a full-attention KV
    cache group, so it needs a block table and slot-mapping buffer per group.
    Every other proposer must keep the single-buffer behaviour, otherwise its
    ACL graph captures one slot-mapping address and runtime writes another."""
    assert AscendGemma4Proposer.uses_per_group_kv_groups is True
    assert AscendSpecDecodeBaseProposer.uses_per_group_kv_groups is False


def test_get_spec_decode_method_dispatches_gemma4():
    from vllm_ascend.spec_decode import get_spec_decode_method

    with (
        patch("vllm_ascend.spec_decode.AscendGemma4Proposer", return_value="gemma4") as g4,
        patch("vllm_ascend.spec_decode.AscendStep3p5MTPProposer", return_value="step3p5") as s35,
        patch("vllm_ascend.spec_decode.AscendEagleProposer", return_value="eagle") as eagle,
    ):
        assert get_spec_decode_method("mtp", _vllm_config(gemma4=True), "cpu", None) == "gemma4"
        g4.assert_called_once()
        s35.assert_not_called()
        eagle.assert_not_called()


def test_get_spec_decode_method_falls_back_when_not_gemma4():
    from vllm_ascend.spec_decode import get_spec_decode_method

    with (
        patch("vllm_ascend.spec_decode.AscendGemma4Proposer", return_value="gemma4") as g4,
        patch("vllm_ascend.spec_decode.AscendStep3p5MTPProposer", return_value="step3p5"),
        patch("vllm_ascend.spec_decode.AscendEagleProposer", return_value="eagle"),
    ):
        assert get_spec_decode_method("mtp", _vllm_config(step3p5=True), "cpu", None) == "step3p5"
        assert get_spec_decode_method("mtp", _vllm_config(), "cpu", None) == "eagle"
        # A proposer built without a speculative_config must not crash the
        # dispatch on the gemma4 probe.
        assert get_spec_decode_method("mtp", SimpleNamespace(speculative_config=None), "cpu", None) == "eagle"
        g4.assert_not_called()


def test_centroids_graph_capture_is_disabled_on_ascend():
    """Upstream captures ``torch.cuda.CUDAGraph()`` for the centroids
    ``get_top_tokens`` path, which raises ``Tried to instantiate dummy base
    class CUDAGraph`` on NPU. The override must be a no-op that leaves
    ``_centroids_sizes`` empty so ``_greedy_sample`` takes the ordinary path."""
    assert AscendGemma4Proposer._setup_centroids_cuda_graphs is not Gemma4Proposer._setup_centroids_cuda_graphs

    proposer = AscendGemma4Proposer.__new__(AscendGemma4Proposer)
    proposer._centroids_sizes = []
    proposer._setup_centroids_cuda_graphs()
    assert proposer._centroids_sizes == []


def test_upstream_per_group_block_table_api_is_present():
    """``NPUModelRunner._prepare_inputs`` calls ``set_per_group_block_table``
    and ``AscendGemma4Proposer`` reads ``_per_group_block_tables``; both are
    owned by upstream ``Gemma4Proposer``. Pin them so an upstream rename fails
    here instead of at model-load time."""
    assert hasattr(Gemma4Proposer, "set_per_group_block_table")
    assert hasattr(Gemma4Proposer, "build_per_group_and_layer_attn_metadata")
