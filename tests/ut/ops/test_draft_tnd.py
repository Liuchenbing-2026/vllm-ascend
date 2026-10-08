# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class NoScalarRead(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("length read back to host")
        return func(*args, **(kwargs or {}))


class TestDraftTnd(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/draft_tnd.py"
        spec = importlib.util.spec_from_file_location("draft_tnd_under_test", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.query = torch.zeros(6, 4, 128, dtype=torch.bfloat16)
        self.key = torch.zeros(4, 128, 256, dtype=self.query.dtype)
        self.kwargs = dict(
            seq_lens=torch.tensor([127, 129], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 6], dtype=torch.int32),
            block_table=torch.tensor([[3, 1], [0, 2]], dtype=torch.int32),
            block_size=128,
            num_heads=4,
            num_kv_heads=2,
            scale=128**-0.5,
            causal=False,
            sliding_window=None,
            attn_mask=None,
            cache={},
        )
        self.ops = SimpleNamespace(
            _npu_fused_infer_attention_score_v2_sink_metadata=Mock(return_value=torch.zeros(1024, dtype=torch.int32)),
            npu_fused_infer_attention_score_v2_sink=Mock(return_value=(torch.ones_like(self.query), None)),
        )
        npu = SimpleNamespace(
            get_stream_limit=lambda stream: dict(cube_core_num=20, vector_core_num=40), current_stream=lambda: None
        )
        self.enterContext(patch.object(torch, "npu", npu, create=True))
        self.enterContext(patch.object(torch.ops, "_C_ascend", self.ops))

    def call(self):
        with NoScalarRead():
            return self.module.draft_tnd_attention(self.query, self.key, self.key, **self.kwargs)

    def test_ragged_tnd_reuses_metadata_only_within_step(self):
        first = self.call()
        self.call()
        metadata_op = self.ops._npu_fused_infer_attention_score_v2_sink_metadata
        self.assertEqual(metadata_op.call_count, 1)
        args = metadata_op.call_args.kwargs
        torch.testing.assert_close(args["actual_seq_lengths"], torch.tensor([2, 6]))
        torch.testing.assert_close(args["actual_seq_lengths_kv"], torch.tensor([127, 129]))
        self.assertEqual(args["aic_core_num"], 20)
        self.assertEqual(self.ops.npu_fused_infer_attention_score_v2_sink.call_args.kwargs["input_layout"], "TND")
        self.assertIsNotNone(first)
        # A new build with identical shapes must not reuse stale lengths.
        self.kwargs["seq_lens"].copy_(torch.tensor([128, 130]))
        self.kwargs["cache"] = {}
        self.call()
        self.assertEqual(metadata_op.call_count, 2)
        torch.testing.assert_close(metadata_op.call_args.kwargs["actual_seq_lengths_kv"], torch.tensor([128, 130]))
        torch.testing.assert_close(args["actual_seq_lengths_kv"], torch.tensor([127, 129]))

    def test_missing_operator_preserves_fallback(self):
        del self.ops.npu_fused_infer_attention_score_v2_sink
        self.assertIsNone(self.call())
        self.ops._npu_fused_infer_attention_score_v2_sink_metadata.assert_not_called()

    def test_noncausal_window_preserves_fallback(self):
        self.kwargs["sliding_window"] = 16
        self.assertIsNone(self.call())
        self.ops._npu_fused_infer_attention_score_v2_sink_metadata.assert_not_called()

    def test_causal_window_parameters(self):
        self.kwargs.update(causal=True, sliding_window=16, attn_mask=torch.ones(2048, 2048, dtype=torch.int8))
        self.call()
        for op in (
            self.ops._npu_fused_infer_attention_score_v2_sink_metadata,
            self.ops.npu_fused_infer_attention_score_v2_sink,
        ):
            self.assertEqual(op.call_args.kwargs["sparse_mode"], 4)
            self.assertEqual(op.call_args.kwargs["pre_tokens"], 16)
            self.assertEqual(op.call_args.kwargs["next_tokens"], 0)

    def test_metadata_runtime_errors_are_not_hidden(self):
        self.ops._npu_fused_infer_attention_score_v2_sink_metadata.side_effect = RuntimeError("bad metadata")
        with self.assertRaisesRegex(RuntimeError, "bad metadata"):
            self.call()
        self.ops.npu_fused_infer_attention_score_v2_sink.assert_not_called()

    def test_noncontiguous_query_preserves_fallback(self):
        self.query = torch.zeros(6, 4, 256, dtype=self.query.dtype)[..., ::2]
        self.assertIsNone(self.call())

    def test_head_dim_256_does_not_launch_unsupported_tiling(self):
        self.query = torch.zeros(6, 4, 256, dtype=self.query.dtype)
        self.key = torch.zeros(4, 128, 512, dtype=self.query.dtype)
        self.assertIsNone(self.call())
        self.ops._npu_fused_infer_attention_score_v2_sink_metadata.assert_not_called()
        self.ops.npu_fused_infer_attention_score_v2_sink.assert_not_called()


if __name__ == "__main__":
    unittest.main()
