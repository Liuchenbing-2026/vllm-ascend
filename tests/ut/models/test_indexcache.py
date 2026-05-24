# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_ascend.models._indexcache.compute_skip_topk.

Covers the 5 decision paths created by PR #8398 + dflash_mtp adaptation:
1. use_index_cache=False (master switch off)
2. pattern is None -> freq-based decision
3. pattern given, layer_idx in range
4. pattern given, layer_idx out of range
5. has_indexer=False (non-sparse layer guard)

Pure-Python, no torch / no vllm-ascend runtime needed.
"""
import pytest

from vllm_ascend.models._indexcache import compute_skip_topk


class TestComputeSkipTopk:
    # ---- Path 1: master switch off -----------------------------------------
    def test_use_index_cache_false_always_returns_false(self):
        """use_index_cache=False -> skip_topk=False on every layer; runtime
        path is byte-equivalent to pre-PR-#8398 behavior."""
        for layer_idx in range(0, 128):
            assert compute_skip_topk(
                layer_idx,
                has_indexer=True,
                use_index_cache=False,
                index_topk_freq=4,
                index_topk_pattern="S" * 128,
            ) is False, f"layer {layer_idx}"

    # ---- Path 2: pattern=None, freq path ------------------------------------
    def test_freq_path_layer_0_never_skips(self):
        """freq path uses max(layer_idx-1, 0) % freq; layer 0 gives 0%freq=0
        -> never skip (first sparse layer always recomputes)."""
        for freq in (1, 2, 3, 4, 8):
            assert compute_skip_topk(
                0, has_indexer=True, use_index_cache=True,
                index_topk_freq=freq,
            ) is False, f"freq={freq}"

    def test_freq_path_pattern_freq_4(self):
        """freq=4: layer 1 keep (0%4=0), 2,3,4 skip, 5 keep (4%4=0), ..."""
        expected = {1: False, 2: True, 3: True, 4: True,
                    5: False, 6: True, 7: True, 8: True, 9: False}
        for layer_idx, want in expected.items():
            assert compute_skip_topk(
                layer_idx, has_indexer=True, use_index_cache=True,
                index_topk_freq=4,
            ) is want, f"layer {layer_idx} expected {want}"

    def test_freq_one_never_skips(self):
        """freq=1 -> (x-1)%1 == 0 always -> skip_topk=False on every layer;
        equivalent to disabling the feature even with master switch on."""
        for layer_idx in range(0, 64):
            assert compute_skip_topk(
                layer_idx, has_indexer=True, use_index_cache=True,
                index_topk_freq=1,
            ) is False, f"layer {layer_idx}"

    # ---- Path 3: explicit pattern, in range ---------------------------------
    def test_pattern_in_range_S_skips_F_keeps(self):
        """'S' at position layer_idx -> skip; anything else (F/A/etc) -> keep."""
        pattern = "FSFSFSXY"
        expected = [False, True, False, True, False, True, False, False]
        for layer_idx, want in enumerate(expected):
            assert compute_skip_topk(
                layer_idx, has_indexer=True, use_index_cache=True,
                index_topk_pattern=pattern,
            ) is want, f"layer {layer_idx} pattern='{pattern[layer_idx]}'"

    def test_pattern_overrides_freq_when_both_given(self):
        """When pattern is non-None, freq is ignored entirely."""
        pattern = "FFFF"  # all keep
        for layer_idx, ch in enumerate(pattern):
            got = compute_skip_topk(
                layer_idx, has_indexer=True, use_index_cache=True,
                index_topk_freq=2,  # would skip layer 2,3 in freq path
                index_topk_pattern=pattern,
            )
            assert got is False, f"layer {layer_idx} (pattern char '{ch}')"

    # ---- Path 4: pattern out of range ---------------------------------------
    def test_pattern_out_of_range_returns_false(self):
        """layer_idx >= len(pattern) -> skip_topk=False (defensive default)."""
        pattern = "SSS"
        for layer_idx in (3, 4, 10, 63, 127):
            assert compute_skip_topk(
                layer_idx, has_indexer=True, use_index_cache=True,
                index_topk_pattern=pattern,
            ) is False, f"layer {layer_idx}"

    def test_pattern_negative_layer_idx_returns_false(self):
        """layer_idx < 0 should never happen but guard against it."""
        assert compute_skip_topk(
            -1, has_indexer=True, use_index_cache=True,
            index_topk_pattern="SSSS",
        ) is False

    # ---- Path 5: non-sparse layer guard -------------------------------------
    def test_no_indexer_returns_false(self):
        """compress_ratio<=1 (no indexer) -> never enter IndexCache path,
        regardless of master switch / pattern / freq."""
        cases = [
            dict(use_index_cache=False),
            dict(use_index_cache=True, index_topk_freq=4),
            dict(use_index_cache=True, index_topk_pattern="SSSS"),
            dict(use_index_cache=True, index_topk_freq=2,
                 index_topk_pattern="SFSF"),
        ]
        for kw in cases:
            for layer_idx in (0, 1, 5, 10):
                assert compute_skip_topk(
                    layer_idx, has_indexer=False, **kw,
                ) is False, f"layer {layer_idx} kw={kw}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
