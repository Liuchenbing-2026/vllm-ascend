# SPDX-License-Identifier: Apache-2.0
"""Regression tests for platform lookahead replay_boundaries patch on vLLM v0.34.0."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from vllm_ascend.patch.platform import patch_lookahead_block_hashes as _patch_mod
except ImportError:
    _patch_path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_lookahead_block_hashes.py"
    _patch_spec = importlib.util.spec_from_file_location("_standalone_patch_lookahead_block_hashes", _patch_path)
    assert _patch_spec is not None and _patch_spec.loader is not None
    _patch_mod = importlib.util.module_from_spec(_patch_spec)
    _patch_spec.loader.exec_module(_patch_mod)

_PATCH_MARKER = _patch_mod._PATCH_MARKER
apply_lookahead_block_hashes_patch = _patch_mod.apply_lookahead_block_hashes_patch


def _make_v034_coordinator_module():
    """Build a v0.34.0-shaped coordinator module with 3-arg replay_boundaries."""

    class _Manager:
        def __init__(self, *, enable_caching: bool = True) -> None:
            self.enable_caching = enable_caching
            self.calls: list[tuple[int, int, tuple[int, ...]]] = []

        def cache_blocks(self, request, num_cacheable_tokens, retention_interval=None, replay_boundaries=()):
            self.calls.append((request.num_prompt_tokens, num_cacheable_tokens, tuple(replay_boundaries)))

    class FakeKVCacheCoordinator:
        def __init__(self, lookahead: int, drop_eagle_group_ids=()):
            self.block_pool = SimpleNamespace(num_hash_lookahead_tokens=lookahead)
            self.drop_eagle_group_ids = set(drop_eagle_group_ids)
            self.scheduler_block_size = 128
            self.retention_interval = None
            self.single_type_managers = [_Manager(enable_caching=True), _Manager(enable_caching=False)]

        def get_num_cacheable_tokens(self, num_computed_tokens: int, group_id: int) -> int:
            return num_computed_tokens

        def cache_blocks(self, request, num_computed_tokens: int) -> None:
            boundaries = coordinator_mod.replay_boundaries(
                request.num_prompt_tokens,
                self.scheduler_block_size,
                use_eagle=bool(self.drop_eagle_group_ids),
            )
            for group_id, manager in enumerate(self.single_type_managers):
                if not manager.enable_caching:
                    continue
                manager.cache_blocks(
                    request,
                    self.get_num_cacheable_tokens(num_computed_tokens, group_id),
                    retention_interval=self.retention_interval,
                    replay_boundaries=boundaries,
                )

    def _legacy_replay_boundaries(
        num_prompt_tokens: int, scheduler_block_size: int, use_eagle: bool
    ) -> tuple[int, ...]:
        if not use_eagle:
            return (num_prompt_tokens - 1,)
        block = scheduler_block_size
        resend = (num_prompt_tokens - 1) // block * block
        extension = num_prompt_tokens // block * block
        return tuple(sorted({max(resend - block, 0), max(extension - block, 0)}))

    coordinator_mod = SimpleNamespace(
        replay_boundaries=_legacy_replay_boundaries,
        KVCacheCoordinator=FakeKVCacheCoordinator,
    )
    return coordinator_mod


def _make_spec(
    *,
    num_speculative_tokens: int = 1,
    multi_module_mtp: bool = False,
    eagle: bool = True,
    eagle_block_drop: bool = True,
    disable_lookahead_block_hashes: bool = False,
):
    return SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        disable_lookahead_block_hashes=disable_lookahead_block_hashes,
        use_eagle=lambda: eagle,
        use_eagle_block_drop=lambda: eagle_block_drop,
        use_multi_module_mtp=lambda: multi_module_mtp,
        speculator_config=SimpleNamespace(
            num_prefill_lookahead_tokens=num_speculative_tokens if multi_module_mtp else 1,
            num_hash_lookahead_tokens=num_speculative_tokens if multi_module_mtp else 1,
        ),
    )


class TestPatchLookaheadBlockHashes(unittest.TestCase):
    def test_v034_patch_wraps_replay_boundaries_and_cache_blocks_and_is_idempotent(self):
        coordinator_mod = _make_v034_coordinator_module()
        with patch.object(_patch_mod, "vllm_version_is", return_value=True):
            self.assertTrue(apply_lookahead_block_hashes_patch(coordinator_module=coordinator_mod))
            self.assertFalse(apply_lookahead_block_hashes_patch(coordinator_module=coordinator_mod))

        self.assertTrue(getattr(coordinator_mod.replay_boundaries, _PATCH_MARKER, False))
        self.assertIn("num_hash_lookahead_tokens", inspect.signature(coordinator_mod.replay_boundaries).parameters)
        self.assertTrue(getattr(coordinator_mod.KVCacheCoordinator.cache_blocks, _PATCH_MARKER, False))

        self.assertEqual(coordinator_mod.replay_boundaries(1537, 512, False, 0), (1536,))
        self.assertEqual(coordinator_mod.replay_boundaries(1537, 512, False, 1), (1536,))
        self.assertEqual(coordinator_mod.replay_boundaries(1537, 512, False, 3), (1534,))
        self.assertEqual(coordinator_mod.replay_boundaries(1537, 512, True, 3), (1024,))
        self.assertEqual(coordinator_mod.replay_boundaries(1536, 512, True, 3), (512, 1024))
        # Short prompts under multi-module MTP lookahead must clamp to 0 instead of negatives.
        self.assertEqual(coordinator_mod.replay_boundaries(1, 512, False, 3), (0,))
        self.assertEqual(coordinator_mod.replay_boundaries(2, 512, False, 3), (0,))
        self.assertEqual(coordinator_mod.replay_boundaries(3, 512, False, 3), (0,))
        self.assertEqual(coordinator_mod.replay_boundaries(4, 512, False, 3), (1,))

        req = SimpleNamespace(num_prompt_tokens=1537)
        coord1 = coordinator_mod.KVCacheCoordinator(lookahead=1)
        coord1.cache_blocks(req, 1537)
        self.assertEqual(coord1.single_type_managers[0].calls, [(1537, 1537, (1536,))])
        self.assertEqual(coord1.single_type_managers[1].calls, [])

        coord3 = coordinator_mod.KVCacheCoordinator(lookahead=3)
        coord3.cache_blocks(req, 1537)
        self.assertEqual(coord3.single_type_managers[0].calls, [(1537, 1537, (1534,))])

        short_req = SimpleNamespace(num_prompt_tokens=2)
        coord3_short = coordinator_mod.KVCacheCoordinator(lookahead=3)
        coord3_short.cache_blocks(short_req, 2)
        self.assertEqual(coord3_short.single_type_managers[0].calls, [(2, 2, (0,))])

        coord_eagle = coordinator_mod.KVCacheCoordinator(lookahead=3, drop_eagle_group_ids={0})
        coord_eagle.cache_blocks(req, 1537)
        self.assertEqual(coord_eagle.single_type_managers[0].calls, [(1537, 1537, (1408,))])

    def test_non_v034_is_strictly_noop(self):
        coordinator_mod = _make_v034_coordinator_module()
        orig_replay = coordinator_mod.replay_boundaries
        orig_cache_blocks = coordinator_mod.KVCacheCoordinator.cache_blocks
        with patch.object(_patch_mod, "vllm_version_is", return_value=False):
            applied = apply_lookahead_block_hashes_patch(coordinator_module=coordinator_mod)
        self.assertFalse(applied)
        self.assertIs(coordinator_mod.replay_boundaries, orig_replay)
        self.assertIs(coordinator_mod.KVCacheCoordinator.cache_blocks, orig_cache_blocks)
        self.assertFalse(getattr(coordinator_mod.replay_boundaries, _PATCH_MARKER, False))
        self.assertFalse(getattr(coordinator_mod.KVCacheCoordinator.cache_blocks, _PATCH_MARKER, False))

    def test_live_vllm_config_and_block_hasher_integration(self):
        """Verify live lookahead config resolution, replay_boundaries, and block hasher alignment."""
        import vllm.v1.core.kv_cache_coordinator as live_coord_mod
        from vllm.config import KVTransferConfig, VllmConfig
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

        from vllm_ascend.utils import vllm_version_is

        self.assertTrue(callable(getattr(KVConnectorBase_V1, "supports_lookahead_block_hashes", None)))
        self.assertTrue(callable(getattr(MultiConnector, "supports_lookahead_block_hashes", None)))

        expected_patched = bool(vllm_version_is("0.34.0"))
        self.assertEqual(getattr(live_coord_mod.replay_boundaries, _PATCH_MARKER, False), expected_patched)
        self.assertEqual(
            getattr(live_coord_mod.KVCacheCoordinator.cache_blocks, _PATCH_MARKER, False),
            expected_patched,
        )
        self.assertEqual(live_coord_mod.replay_boundaries(1537, 512, False, 3), (1534,))

        class _LookaheadConnector(KVConnectorBase_V1):
            @classmethod
            def supports_lookahead_block_hashes(cls, kv_transfer_config) -> bool:
                return True

        class _OptOutConnector(KVConnectorBase_V1):
            @classmethod
            def supports_lookahead_block_hashes(cls, kv_transfer_config) -> bool:
                return False

        registry = getattr(KVConnectorFactory, "_registry", None)
        orig_registry = dict(registry) if isinstance(registry, dict) else None
        fake_registry = {
            "LookaheadConnector": lambda: _LookaheadConnector,
            "MultiConnector": lambda: MultiConnector,
            "OptOutConnector": lambda: _OptOutConnector,
        }

        def resolve_live_lookahead(ktc) -> int:
            cfg = object.__new__(VllmConfig)
            object.__setattr__(cfg, "speculative_config", _make_spec())
            object.__setattr__(cfg, "kv_transfer_config", ktc)
            object.__setattr__(cfg, "cache_config", SimpleNamespace(enable_prefix_caching=True))
            return int(cfg._resolve_num_hash_lookahead_tokens())

        try:
            with patch.object(
                KVConnectorFactory,
                "get_connector_class",
                side_effect=lambda ktc: fake_registry[ktc.kv_connector](),
            ):
                if orig_registry is not None and isinstance(registry, dict):
                    registry.clear()
                    registry.update(fake_registry)

                ktc_lookahead = KVTransferConfig(kv_connector="LookaheadConnector", kv_role="kv_both")
                ktc_multi = KVTransferConfig(
                    kv_connector="MultiConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={
                        "connectors": [{"kv_connector": "LookaheadConnector", "kv_role": "kv_both"}]
                    },
                )
                ktc_multi_opt_out = KVTransferConfig(
                    kv_connector="MultiConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={
                        "connectors": [
                            {"kv_connector": "LookaheadConnector", "kv_role": "kv_both"},
                            {"kv_connector": "OptOutConnector", "kv_role": "kv_both"},
                        ]
                    },
                )

                self.assertTrue(_LookaheadConnector.supports_lookahead_block_hashes(ktc_lookahead))
                self.assertTrue(MultiConnector.supports_lookahead_block_hashes(ktc_multi))
                self.assertFalse(MultiConnector.supports_lookahead_block_hashes(ktc_multi_opt_out))

                lookahead = resolve_live_lookahead(ktc_lookahead)
                self.assertEqual(lookahead, 1)
                self.assertEqual(resolve_live_lookahead(ktc_multi), 1)
                self.assertEqual(resolve_live_lookahead(ktc_multi_opt_out), 0)
        finally:
            if orig_registry is not None and isinstance(registry, dict):
                registry.clear()
                registry.update(orig_registry)

        hash_fn = lambda obj: hashlib.sha256(repr(obj).encode("utf-8")).digest()
        init_none_hash(hash_fn)

        def make_req(toks: list[int]):
            return SimpleNamespace(
                _all_token_ids=toks,
                all_token_ids=toks,
                num_tokens=len(toks),
                block_hashes=[],
                mm_features=[],
                lora_request=None,
                cache_salt=None,
                prompt_embeds=None,
                prompt_is_token_ids=None,
            )

        base_tokens = list(range(1537))
        fork256_tokens = list(base_tokens)
        fork256_tokens[256] += 10000
        fork257_tokens = list(base_tokens)
        fork257_tokens[257] += 10000

        hasher = get_request_block_hasher(128, hash_fn, lookahead)
        base_hashes = hasher(make_req(base_tokens))
        fork256_hashes = hasher(make_req(fork256_tokens))
        fork257_hashes = hasher(make_req(fork257_tokens))
        if len(base_hashes) >= 3:
            self.assertEqual(base_hashes[:1], fork256_hashes[:1])
            self.assertNotEqual(base_hashes[1], fork256_hashes[1])
            self.assertEqual(base_hashes[:2], fork257_hashes[:2])
            self.assertNotEqual(base_hashes[2], fork257_hashes[2])

    def test_platform_init_registers_lookahead_before_kv_cache_coordinator(self):
        init_path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/__init__.py"
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
        imported_modules = [
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        ]
        lookahead_mod = "vllm_ascend.patch.platform.patch_lookahead_block_hashes"
        coord_mod = "vllm_ascend.patch.platform.patch_kv_cache_coordinator"
        self.assertIn(lookahead_mod, imported_modules)
        self.assertIn(coord_mod, imported_modules)
        self.assertLess(imported_modules.index(lookahead_mod), imported_modules.index(coord_mod))
