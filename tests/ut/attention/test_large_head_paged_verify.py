# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU unit tests for head_dim=512 speculative paged verify and decode."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
)
from vllm_ascend.attention.large_head_paged_verify import (
    AscendLargeHeadPagedAttentionBackendImpl,
    AscendLargeHeadPagedBackend,
    AscendLargeHeadPagedMetadata,
    AscendLargeHeadPagedMetadataBuilder,
    LargeHeadPagedVerifyParamProvider,
    build_paged_verify_mask,
    large_head_decode_graph_params,
    large_head_verify_graph_params,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    expand_pa_per_token_rows,
    requires_contiguous_pa_kv_cache,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.device.utils import FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE, _gather_paged_kv_to_dense
from vllm_ascend.platform import NPUPlatform, _should_use_large_head_paged_backend


def _selector_config(**overrides):
    defaults = dict(
        head_size=FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=128,
        use_mla=False,
        has_sliding_window=False,
        sliding_window=None,
        use_sparse=False,
        use_pcp=False,
        use_dcp=False,
        use_kv_connector=False,
        has_sink=False,
        use_non_causal=False,
        use_per_head_quant_scales=False,
        use_adaptive_verification=False,
        use_rswa=False,
        use_hisparse=False,
        attn_type=AttentionType.DECODER,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mock_vllm_config(
    *,
    with_spec: bool = True,
    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    runner_type: str = "generate",
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type=runner_type,
            max_model_len=8192,
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(model_type="gemma4", architectures=["Gemma4ForConditionalGeneration"]),
            hf_text_config=SimpleNamespace(model_type="gemma4_text"),
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        cache_config=SimpleNamespace(block_size=8, cache_dtype="auto"),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True, max_num_seqs=8, max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        speculative_config=(
            SimpleNamespace(
                num_speculative_tokens=3,
                parallel_drafting=False,
                speculator_config=SimpleNamespace(parallel_drafting=False),
                use_dspark=lambda: False,
            )
            if with_spec
            else None
        ),
        kv_transfer_config=None,
        quant_config=None,
        use_v2_model_runner=True,
    )


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3])
def test_select_large_head_paged_backend_enabled_on_a2_a3(monkeypatch, device):
    cfg = _mock_vllm_config(with_spec=True)
    selector = _selector_config()
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.platform.get_current_hardware_profile", lambda: get_hardware_profile(device))
    assert _should_use_large_head_paged_backend(selector) is True

    monkeypatch.setattr("vllm_ascend.platform._validate_fa3_backend", lambda *_: False)
    monkeypatch.setattr("vllm_ascend.platform._validate_flash_attn_backend", lambda *_: False)
    monkeypatch.setattr("vllm_ascend.platform._validate_flash_attn_npu_3", lambda *_: False)
    assert (
        NPUPlatform._select_attn_backend_cls(selector)
        == "vllm_ascend.attention.large_head_paged_verify.AscendLargeHeadPagedBackend"
    )


@pytest.mark.parametrize(
    ("device", "with_spec", "runner_type", "selector_overrides"),
    [
        (AscendDeviceType.A5, True, "generate", {}),
        (AscendDeviceType._310P, True, "generate", {}),
        (AscendDeviceType.A2, False, "generate", {}),
        (AscendDeviceType.A2, True, "pooling", {}),
        (AscendDeviceType.A2, True, "generate", {"head_size": 256}),
        (AscendDeviceType.A2, True, "generate", {"has_sliding_window": True}),
        (AscendDeviceType.A2, True, "generate", {"attn_type": AttentionType.ENCODER_ONLY}),
        (AscendDeviceType.A2, True, "generate", {"use_mla": True}),
        (AscendDeviceType.A2, True, "generate", {"use_sparse": True}),
        (AscendDeviceType.A2, True, "generate", {"use_pcp": True}),
        (AscendDeviceType.A2, True, "generate", {"use_dcp": True}),
        (AscendDeviceType.A2, True, "generate", {"use_kv_connector": True}),
        (AscendDeviceType.A2, True, "generate", {"has_sink": True}),
        (AscendDeviceType.A2, True, "generate", {"use_non_causal": True}),
        (AscendDeviceType.A2, True, "generate", {"use_per_head_quant_scales": True}),
        (AscendDeviceType.A2, True, "generate", {"use_adaptive_verification": True}),
        (AscendDeviceType.A2, True, "generate", {"use_rswa": True}),
        (AscendDeviceType.A2, True, "generate", {"use_hisparse": True}),
    ],
)
def test_select_large_head_paged_backend_exclusions(monkeypatch, device, with_spec, runner_type, selector_overrides):
    cfg = _mock_vllm_config(with_spec=with_spec, runner_type=runner_type)
    selector = _selector_config(**selector_overrides)
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.platform.get_current_hardware_profile", lambda: get_hardware_profile(device))
    assert _should_use_large_head_paged_backend(selector) is False


def test_kv_cache_shape_and_contiguous_cache_contract(monkeypatch):
    a2_profile = get_hardware_profile(AscendDeviceType.A2)
    a5_profile = get_hardware_profile(AscendDeviceType.A5)

    with (
        patch("vllm_ascend.attention.large_head_paged_verify.get_current_hardware_profile", return_value=a2_profile),
        patch("vllm.config.get_current_vllm_config_or_none", return_value=_mock_vllm_config()),
    ):
        shape_bnsd = AscendLargeHeadPagedBackend.get_kv_cache_shape(16, 128, 4, 512)
        assert shape_bnsd == (2, 16, 4, 128, 512)

    with (
        patch("vllm_ascend.attention.large_head_paged_verify.get_current_hardware_profile", return_value=a2_profile),
        patch(
            "vllm.config.get_current_vllm_config_or_none",
            return_value=_mock_vllm_config(cudagraph_mode=CUDAGraphMode.FULL),
        ),
    ):
        shape_full = AscendLargeHeadPagedBackend.get_kv_cache_shape(16, 128, 4, 512)
        assert shape_full == (2, 16, 128, 4, 512)

    with (
        patch("vllm_ascend.attention.large_head_paged_verify.get_current_hardware_profile", return_value=a5_profile),
        patch("vllm.config.get_current_vllm_config_or_none", return_value=_mock_vllm_config()),
    ):
        shape_a5 = AscendLargeHeadPagedBackend.get_kv_cache_shape(16, 128, 4, 512)
        assert shape_a5 == (2, 16, 128, 4, 512)

    cfg = _mock_vllm_config(with_spec=True)
    spec = FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=512, dtype=torch.bfloat16)
    impl = AscendLargeHeadPagedAttentionBackendImpl.__new__(AscendLargeHeadPagedAttentionBackendImpl)
    impl.sliding_window = None
    assert requires_contiguous_pa_kv_cache(SimpleNamespace(impl=impl), cfg, spec) is True

    padded_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=512,
        dtype=torch.bfloat16,
        page_size_padded=spec.real_page_size_bytes + 64,
    )
    assert requires_contiguous_pa_kv_cache(SimpleNamespace(impl=impl), cfg, padded_spec) is False

    base_impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    base_impl.sliding_window = None
    assert requires_contiguous_pa_kv_cache(SimpleNamespace(impl=base_impl), cfg, spec) is False


def test_build_paged_verify_mask_oracle_and_pad_row_safety():
    # Batch: real row seq=6, real minimal seq=4 (k=3 -> qlen=4), padded/dummy rows seq=1 and seq=0.
    seq_lens = torch.tensor([6, 4, 1, 0], dtype=torch.int32)
    block_table = torch.zeros((4, 1), dtype=torch.int32)
    mask = build_paged_verify_mask(
        seq_lens,
        block_table,
        num_speculative_tokens=3,
        block_size=8,
        device=torch.device("cpu"),
    )
    assert mask.shape == (4, 1, 4, 8)
    assert mask.dtype == torch.bool

    expected = torch.ones((4, 1, 4, 8), dtype=torch.bool)
    # seq=6, qlen=4 -> allowed last positions: [2, 3, 4, 5]
    for j in range(4):
        expected[0, 0, j, : 2 + j + 1] = False
        expected[1, 0, j, : j + 1] = False
        # Padded/dummy rows clamp_min(0) -> position 0 stays unmasked (False) to prevent all-True NaNs.
        expected[2, 0, j, 0] = False
        expected[3, 0, j, 0] = False

    torch.testing.assert_close(mask, expected)


def test_expand_pa_per_token_rows_oracle_and_idempotency():
    metadata = AscendLargeHeadPagedMetadata(
        block_tables=torch.tensor([[10, 11], [20, 21], [99, 99]], dtype=torch.int32),
        actual_seq_lengths_q=[4, 6, 6],
        seq_lens_list=[10, 5, 0],
        seq_lens=torch.tensor([10, 5, 0], dtype=torch.int32),
    )
    assert expand_pa_per_token_rows(metadata) is True
    assert metadata.seq_lens.tolist() == [7, 8, 9, 10, 4, 5]
    torch.testing.assert_close(
        metadata.block_tables,
        torch.tensor([[10, 11], [10, 11], [10, 11], [10, 11], [20, 21], [20, 21]], dtype=torch.int32),
    )
    # Second call must be idempotent.
    assert expand_pa_per_token_rows(metadata) is True
    assert metadata.seq_lens.tolist() == [7, 8, 9, 10, 4, 5]


def test_gather_paged_kv_to_dense_bnsd_matches_natural():
    num_blocks, block_size, num_kv_heads, head_size = 4, 8, 2, 16
    natural_k = torch.randn((num_blocks, block_size, num_kv_heads, head_size), dtype=torch.float32)
    natural_v = torch.randn((num_blocks, block_size, num_kv_heads, head_size), dtype=torch.float32)
    bnsd_k = natural_k.permute(0, 2, 1, 3).contiguous()
    bnsd_v = natural_v.permute(0, 2, 1, 3).contiguous()
    block_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    seq_lens = [10, 5]

    out_nat_k, out_nat_v = _gather_paged_kv_to_dense(
        natural_k, natural_v, block_table, seq_lens, num_kv_heads, head_size
    )
    out_bnsd_k, out_bnsd_v = _gather_paged_kv_to_dense(bnsd_k, bnsd_v, block_table, seq_lens, num_kv_heads, head_size)
    torch.testing.assert_close(out_nat_k, out_bnsd_k)
    torch.testing.assert_close(out_nat_v, out_bnsd_v)

    # Explicit use_bnsd handles square dims where block_size == num_kv_heads.
    sq_nat_k = torch.randn((3, 4, 4, 8), dtype=torch.float32)
    sq_nat_v = torch.randn((3, 4, 4, 8), dtype=torch.float32)
    sq_bnsd_k = sq_nat_k.permute(0, 2, 1, 3).contiguous()
    sq_bnsd_v = sq_nat_v.permute(0, 2, 1, 3).contiguous()
    bt_sq = torch.tensor([[1, 2]], dtype=torch.int32)
    exp_k, exp_v = _gather_paged_kv_to_dense(sq_nat_k, sq_nat_v, bt_sq, [6], 4, 8, use_bnsd=False)
    act_k, act_v = _gather_paged_kv_to_dense(sq_bnsd_k, sq_bnsd_v, bt_sq, [6], 4, 8, use_bnsd=True)
    torch.testing.assert_close(exp_k, act_k)
    torch.testing.assert_close(exp_v, act_v)


def test_metadata_builder_capture_vs_runtime_mask(monkeypatch):
    cfg = _mock_vllm_config(with_spec=True, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    spec = FullAttentionSpec(block_size=8, num_kv_heads=2, head_size=512, dtype=torch.bfloat16)
    monkeypatch.setattr(
        "vllm_ascend.attention.attention_v1.get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    monkeypatch.setattr("vllm_ascend.attention.utils.get_current_vllm_config", lambda: cfg)
    builder = AscendLargeHeadPagedMetadataBuilder(spec, ["layer.59"], cfg, torch.device("cpu"))
    common = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([6, 8], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=2,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        max_seq_len=8,
        causal=True,
        attn_state=AscendAttentionState.ChunkedPrefill,
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int32),
    )

    cap_meta = builder.build_for_cudagraph_capture(common)
    assert cap_meta.paged_verify_mask is None
    assert cap_meta.use_pa_graph is False

    rt_meta = builder.build(0, common)
    assert rt_meta.paged_verify_mask is not None
    assert rt_meta.paged_verify_mask.shape == (2, 1, 4, 16)
    assert 16 in builder._kpos_cache and 4 in builder._qpos_cache


def test_bnsd_verify_capture_param_provider_and_eager_routing(monkeypatch):
    cfg = _mock_vllm_config(with_spec=True, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1.get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.attention.utils.get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1.is_pcp_decode_sharding_enabled", lambda *_: False)
    monkeypatch.setattr("vllm_ascend.attention.attention_v1.needs_layer_aware_fia_graph_replay", lambda: True)
    monkeypatch.setattr(
        "vllm_ascend.attention.large_head_paged_verify.get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    monkeypatch.setattr(
        "vllm_ascend.attention.attention_v1.get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )

    impl = AscendLargeHeadPagedAttentionBackendImpl(
        num_heads=8,
        head_size=512,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )
    impl._layer_name = "layer.59"
    assert impl.use_bnsd_kv_cache is True

    k_cache = torch.zeros((4, 2, 8, 512), dtype=torch.bfloat16)
    v_cache = torch.zeros((4, 2, 8, 512), dtype=torch.bfloat16)
    impl.key_cache, impl.value_cache = k_cache, v_cache

    query = torch.randn((8, 8, 512), dtype=torch.bfloat16)
    key = torch.randn((8, 2, 512), dtype=torch.bfloat16)
    value = torch.randn((8, 2, 512), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    cap_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[4, 8],
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        seq_lens_list=[6, 8],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_prefills=0,
        causal=True,
    )

    captured_ws = torch.zeros(128, dtype=torch.uint8)
    get_ws_mock = MagicMock(return_value=captured_ws)
    reg_mock = MagicMock()
    monkeypatch.setattr(
        "torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace",
        get_ws_mock,
        raising=False,
    )
    monkeypatch.setattr("vllm_ascend.attention.large_head_paged_verify.register_task", reg_mock)
    monkeypatch.setattr(
        "vllm_ascend.attention.large_head_paged_verify.get_capture_resource",
        lambda _key, factory, use_max_workspace=False: factory(),
    )

    prev_capturing = _EXTRA_CTX.capturing
    try:
        _EXTRA_CTX.capturing = True
        impl.forward_fused_infer_attention(query, key, value, cap_meta, output)
    finally:
        _EXTRA_CTX.capturing = prev_capturing

    get_ws_mock.assert_called_once()
    assert reg_mock.call_count == 1
    op_arg, kwargs_arg, provider_arg = reg_mock.call_args[0]
    assert op_arg is torch_npu.npu_fused_infer_attention_score_v2.out
    assert kwargs_arg["input_layout"] == "BNSD"
    assert kwargs_arg["sparse_mode"] == 0
    assert kwargs_arg["block_size"] == 8
    assert kwargs_arg["actual_seq_qlen"] == [4, 4]
    assert kwargs_arg["atten_mask"].shape == (2, 1, 4, 16)
    assert provider_arg == LargeHeadPagedVerifyParamProvider("layer.59")

    # Provider raises if multi-query verify replay lacks matching paged_verify_mask.
    with pytest.raises(RuntimeError, match="paged_verify_mask"):
        provider_arg.resolve({"layer.59": cap_meta})

    rt_mask = build_paged_verify_mask(cap_meta.seq_lens, block_table, 3, 8, torch.device("cpu"))
    rt_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[4, 8],
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        seq_lens_list=[6, 8],
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_prefills=0,
        causal=True,
        paged_verify_mask=rt_mask,
    )
    resolved = provider_arg.resolve({"layer.59": rt_meta})
    assert resolved["atten_mask"] is rt_mask
    assert resolved["actual_seq_kvlen"] == [6, 8]

    # Eager BNSD verify path invokes npu_fused_infer_attention_score_v2 in BNSD without dense gather.
    fia_v2_mock = MagicMock(return_value=(torch.ones((2, 8, 4, 512), dtype=torch.bfloat16), torch.zeros(1)))
    monkeypatch.setattr("torch_npu.npu_fused_infer_attention_score_v2", fia_v2_mock)
    out = impl.forward_fused_infer_attention(query, key, value, rt_meta, output)
    assert out is output
    assert fia_v2_mock.call_count == 1
    assert fia_v2_mock.call_args.kwargs["input_layout"] == "BNSD"
    assert fia_v2_mock.call_args.kwargs["atten_mask"] is rt_mask
    torch.testing.assert_close(output, torch.ones_like(output))

    # Eager draft decode (qlen=1, _EXTRA_CTX.is_draft_model=True) also uses BNSD paged op.
    fia_v2_mock.reset_mock()
    fia_v2_mock.return_value = (torch.full((2, 8, 1, 512), 2.0, dtype=torch.bfloat16), torch.zeros(1))
    draft_q = torch.randn((2, 8, 512), dtype=torch.bfloat16)
    draft_out = torch.empty_like(draft_q)
    draft_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[1, 2],
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        seq_lens_list=[6, 8],
        attn_state=AscendAttentionState.DecodeOnly,
        num_prefills=0,
        causal=True,
    )
    prev_is_draft = _EXTRA_CTX.is_draft_model
    try:
        _EXTRA_CTX.is_draft_model = True
        impl.forward_fused_infer_attention(draft_q, draft_q[:, :2], draft_q[:, :2], draft_meta, draft_out)
    finally:
        _EXTRA_CTX.is_draft_model = prev_is_draft
    assert fia_v2_mock.call_count == 1
    assert fia_v2_mock.call_args.kwargs["input_layout"] == "BNSD"
    assert fia_v2_mock.call_args.kwargs["actual_seq_qlen"] == [1, 1]
    torch.testing.assert_close(draft_out, torch.full_like(draft_out, 2.0))

    # PrefillNoCache must never enter eager BNSD verify even with uniform qlen=4.
    fia_v2_mock.reset_mock()
    nocache_meta = AscendLargeHeadPagedMetadata(
        block_tables=block_table,
        actual_seq_lengths_q=[4, 8],
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        seq_lens_list=[4, 4],
        attn_state=AscendAttentionState.PrefillNoCache,
        num_prefills=2,
        causal=True,
        paged_verify_mask=rt_mask,
    )
    assert impl._should_eager_bnsd_verify(nocache_meta, query.shape[0]) is False

    # FULL mode helpers emit PagedAttention kwargs with expanded block tables.
    pa_meta = AscendLargeHeadPagedMetadata(
        block_tables=torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]], dtype=torch.int32),
        actual_seq_lengths_q=[2, 4],
        seq_lens=torch.tensor([5, 6, 7, 8], dtype=torch.int32),
        seq_lens_list=[6, 8],
        use_pa_graph=True,
    )
    pa_verify = large_head_verify_graph_params("layer.59", pa_meta)
    assert set(pa_verify) == {"block_table", "context_lens"}
    assert pa_verify["block_table"].shape[0] == 4
    pa_decode = large_head_decode_graph_params(pa_meta, [11, 12])
    assert set(pa_decode) == {"block_table", "context_lens"}
    assert pa_decode["context_lens"].tolist() == [11, 12]
