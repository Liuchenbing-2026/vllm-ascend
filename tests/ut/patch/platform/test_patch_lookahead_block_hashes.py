# SPDX-License-Identifier: Apache-2.0
"""Regression tests for platform lookahead block-hash backport on vLLM v0.33.x."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
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


def _make_v033_targets():
    """Build v0.33.0-shaped classes lacking vllm#64096 and vllm#65448."""

    class FakeKVConnectorBase_V1:
        pass

    class FakeKVTransferConfig:
        def __init__(self, **kwargs):
            self.kv_connector = kwargs.get("kv_connector")
            self.engine_id = kwargs.get("engine_id", "engine-0")
            self.kv_connector_extra_config = kwargs.get("kv_connector_extra_config", {})
            self.kv_connector_module_path = kwargs.get("kv_connector_module_path")

    class FakeFactory:
        _registry: dict[str, type] = {}

        @classmethod
        def get_connector_class(cls, kv_transfer_config):
            name = kv_transfer_config.kv_connector
            if name not in cls._registry:
                raise ValueError(f"Unsupported connector type: {name}")
            return cls._registry[name]

    class FakeMultiConnector(FakeKVConnectorBase_V1):
        @classmethod
        def _get_connector_classes_and_configs(cls, vllm_config):
            ktcs = vllm_config.kv_transfer_config.kv_connector_extra_config.get("connectors", [])
            ret = []
            for ktc in ktcs:
                temp_config = copy.copy(vllm_config)
                temp_config.kv_transfer_config = FakeKVTransferConfig(**ktc)
                ret.append((FakeFactory.get_connector_class(temp_config.kv_transfer_config), temp_config))
            return ret

    class FakeVllmConfig:
        def __init__(
            self,
            *,
            speculative_config=None,
            kv_transfer_config=None,
            enable_prefix_caching: bool = True,
        ) -> None:
            self.speculative_config = speculative_config
            self.kv_transfer_config = kv_transfer_config
            self.cache_config = SimpleNamespace(enable_prefix_caching=enable_prefix_caching)

        @property
        def num_prefill_lookahead_tokens(self) -> int:
            spec = self.speculative_config
            if spec is None or not spec.use_eagle():
                return 0
            if spec.use_multi_module_mtp():
                return spec.num_speculative_tokens
            return 1

        @property
        def num_hash_lookahead_tokens(self) -> int:
            spec = self.speculative_config
            if (
                spec is None
                or not spec.use_eagle_block_drop()
                or spec.disable_lookahead_block_hashes
                or not self.cache_config.enable_prefix_caching
            ):
                return 0
            if self.kv_transfer_config is not None:
                return 0
            return self.num_prefill_lookahead_tokens

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

    class FakeAscendStoreConnector(FakeKVConnectorBase_V1):
        @classmethod
        def supports_lookahead_block_hashes(cls, kv_transfer_config) -> bool:
            return True

    class FakeUnsupportedConnector(FakeKVConnectorBase_V1):
        pass

    FakeFactory._registry["AscendStoreConnector"] = FakeAscendStoreConnector
    FakeFactory._registry["MooncakeConnectorStoreV1"] = FakeAscendStoreConnector
    FakeFactory._registry["UnsupportedConnector"] = FakeUnsupportedConnector
    FakeFactory._registry["MultiConnector"] = FakeMultiConnector

    return SimpleNamespace(
        VllmConfig=FakeVllmConfig,
        KVConnectorBase_V1=FakeKVConnectorBase_V1,
        MultiConnector=FakeMultiConnector,
        KVConnectorFactory=FakeFactory,
        KVTransferConfig=FakeKVTransferConfig,
        coordinator_mod=coordinator_mod,
        AscendStoreConnector=FakeAscendStoreConnector,
        UnsupportedConnector=FakeUnsupportedConnector,
    )


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
    def test_v033_patch_enables_lookahead_for_supporting_connectors_and_multi_connector(self):
        t = _make_v033_targets()
        self.assertFalse(hasattr(t.KVConnectorBase_V1, "supports_lookahead_block_hashes"))

        applied = apply_lookahead_block_hashes_patch(
            vllm_config_cls=t.VllmConfig,
            kv_connector_base_cls=t.KVConnectorBase_V1,
            multi_connector_cls=t.MultiConnector,
            factory_cls=t.KVConnectorFactory,
            kv_transfer_config_cls=t.KVTransferConfig,
            coordinator_module=t.coordinator_mod,
        )
        self.assertTrue(applied)
        # Re-applying must be strictly idempotent.
        self.assertFalse(
            apply_lookahead_block_hashes_patch(
                vllm_config_cls=t.VllmConfig,
                kv_connector_base_cls=t.KVConnectorBase_V1,
                multi_connector_cls=t.MultiConnector,
                factory_cls=t.KVConnectorFactory,
                kv_transfer_config_cls=t.KVTransferConfig,
                coordinator_module=t.coordinator_mod,
            )
        )

        # Single AscendStoreConnector and MooncakeConnectorStoreV1 alias (L=1 and L=3),
        # even when enable_prefix_caching=False (matching vllm#64096).
        for conn_name in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
            for enable_pc in (True, False):
                cfg1 = t.VllmConfig(
                    speculative_config=_make_spec(num_speculative_tokens=3),
                    kv_transfer_config=t.KVTransferConfig(kv_connector=conn_name),
                    enable_prefix_caching=enable_pc,
                )
                self.assertEqual(cfg1.num_hash_lookahead_tokens, 1)

                cfg3 = t.VllmConfig(
                    speculative_config=_make_spec(num_speculative_tokens=3, multi_module_mtp=True),
                    kv_transfer_config=t.KVTransferConfig(kv_connector=conn_name),
                    enable_prefix_caching=enable_pc,
                )
                self.assertEqual(cfg3.num_hash_lookahead_tokens, 3)

        # Unsupported connector stays on L=0.
        unsupported_cfg = t.VllmConfig(
            speculative_config=_make_spec(),
            kv_transfer_config=t.KVTransferConfig(kv_connector="UnsupportedConnector"),
        )
        self.assertEqual(unsupported_cfg.num_hash_lookahead_tokens, 0)
        self.assertFalse(t.KVConnectorBase_V1.supports_lookahead_block_hashes(unsupported_cfg.kv_transfer_config))

        # MultiConnector with all supporting children resolves L=1 and propagates to child configs.
        multi_ok_cfg = t.VllmConfig(
            speculative_config=_make_spec(),
            kv_transfer_config=t.KVTransferConfig(
                kv_connector="MultiConnector",
                kv_connector_extra_config={"connectors": [{"kv_connector": "AscendStoreConnector"}]},
            ),
        )
        self.assertEqual(multi_ok_cfg.num_hash_lookahead_tokens, 1)
        children_ok = t.MultiConnector._get_connector_classes_and_configs(multi_ok_cfg)
        self.assertEqual(children_ok[0][1].num_hash_lookahead_tokens, 1)

        # Mixed MultiConnector must keep 0 on parent AND on the supporting child (zero drift).
        multi_mixed_cfg = t.VllmConfig(
            speculative_config=_make_spec(),
            kv_transfer_config=t.KVTransferConfig(
                kv_connector="MultiConnector",
                kv_connector_extra_config={
                    "connectors": [
                        {"kv_connector": "AscendStoreConnector"},
                        {"kv_connector": "UnsupportedConnector"},
                    ]
                },
            ),
        )
        self.assertEqual(multi_mixed_cfg.num_hash_lookahead_tokens, 0)
        children_mixed = t.MultiConnector._get_connector_classes_and_configs(multi_mixed_cfg)
        self.assertEqual(children_mixed[0][1].num_hash_lookahead_tokens, 0)
        self.assertEqual(children_mixed[1][1].num_hash_lookahead_tokens, 0)

        # Empty MultiConnector returns 0.
        empty_multi = t.VllmConfig(
            speculative_config=_make_spec(),
            kv_transfer_config=t.KVTransferConfig(kv_connector="MultiConnector", kv_connector_extra_config={}),
        )
        self.assertEqual(empty_multi.num_hash_lookahead_tokens, 0)

        # Disabled / non-drop / no-speculative / kv_transfer_config=None gating.
        self.assertEqual(
            t.VllmConfig(
                speculative_config=None,
                kv_transfer_config=t.KVTransferConfig(kv_connector="AscendStoreConnector"),
            ).num_hash_lookahead_tokens,
            0,
        )
        self.assertEqual(
            t.VllmConfig(
                speculative_config=_make_spec(eagle_block_drop=False),
                kv_transfer_config=t.KVTransferConfig(kv_connector="AscendStoreConnector"),
            ).num_hash_lookahead_tokens,
            0,
        )
        self.assertEqual(
            t.VllmConfig(
                speculative_config=_make_spec(disable_lookahead_block_hashes=True),
                kv_transfer_config=t.KVTransferConfig(kv_connector="AscendStoreConnector"),
            ).num_hash_lookahead_tokens,
            0,
        )
        self.assertEqual(
            t.VllmConfig(
                speculative_config=_make_spec(), kv_transfer_config=None, enable_prefix_caching=True
            ).num_hash_lookahead_tokens,
            1,
        )
        self.assertEqual(
            t.VllmConfig(
                speculative_config=_make_spec(), kv_transfer_config=None, enable_prefix_caching=False
            ).num_hash_lookahead_tokens,
            0,
        )

    def test_v033_patch_lazy_plugin_load_and_unknown_connector_fallback(self):
        t = _make_v033_targets()
        t.KVConnectorFactory._registry.pop("AscendStoreConnector")
        load_calls = 0

        def fake_load_plugins() -> None:
            nonlocal load_calls
            load_calls += 1
            t.KVConnectorFactory._registry["AscendStoreConnector"] = t.AscendStoreConnector

        apply_lookahead_block_hashes_patch(
            vllm_config_cls=t.VllmConfig,
            kv_connector_base_cls=t.KVConnectorBase_V1,
            multi_connector_cls=t.MultiConnector,
            factory_cls=t.KVConnectorFactory,
            kv_transfer_config_cls=t.KVTransferConfig,
            coordinator_module=t.coordinator_mod,
            load_plugins_fn=fake_load_plugins,
        )

        # First lookup misses initial registry -> triggers load_plugins_fn -> succeeds with L=1.
        cfg = t.VllmConfig(
            speculative_config=_make_spec(),
            kv_transfer_config=t.KVTransferConfig(kv_connector="AscendStoreConnector"),
        )
        self.assertEqual(cfg.num_hash_lookahead_tokens, 1)
        self.assertEqual(load_calls, 1)

        # Completely unknown connector must safely fall back to 0 without raising ValueError.
        unknown_cfg = t.VllmConfig(
            speculative_config=_make_spec(),
            kv_transfer_config=t.KVTransferConfig(kv_connector="CompletelyUnknownConnector"),
        )
        self.assertEqual(unknown_cfg.num_hash_lookahead_tokens, 0)
        self.assertEqual(load_calls, 2)

    def test_v033_patch_replay_boundaries_and_cache_blocks_for_multi_module_mtp(self):
        t = _make_v033_targets()
        apply_lookahead_block_hashes_patch(
            vllm_config_cls=t.VllmConfig,
            kv_connector_base_cls=t.KVConnectorBase_V1,
            multi_connector_cls=t.MultiConnector,
            factory_cls=t.KVConnectorFactory,
            kv_transfer_config_cls=t.KVTransferConfig,
            coordinator_module=t.coordinator_mod,
        )

        self.assertEqual(t.coordinator_mod.replay_boundaries(1537, 512, False, 0), (1536,))
        self.assertEqual(t.coordinator_mod.replay_boundaries(1537, 512, False, 1), (1536,))
        self.assertEqual(t.coordinator_mod.replay_boundaries(1537, 512, False, 3), (1534,))

        req = SimpleNamespace(num_prompt_tokens=1537)
        coord1 = t.coordinator_mod.KVCacheCoordinator(lookahead=1)
        coord1.cache_blocks(req, 1537)
        self.assertEqual(coord1.single_type_managers[0].calls, [(1537, 1537, (1536,))])
        self.assertEqual(coord1.single_type_managers[1].calls, [])

        coord3 = t.coordinator_mod.KVCacheCoordinator(lookahead=3)
        coord3.cache_blocks(req, 1537)
        self.assertEqual(coord3.single_type_managers[0].calls, [(1537, 1537, (1534,))])

    def test_post64096_targets_are_strictly_noop(self):
        class Post64096KVConnectorBase_V1:
            @classmethod
            def supports_lookahead_block_hashes(cls, kv_transfer_config) -> bool:
                return False

        class Post64096MultiConnector(Post64096KVConnectorBase_V1):
            @classmethod
            def supports_lookahead_block_hashes(cls, kv_transfer_config) -> bool:
                return True

            @classmethod
            def _get_connector_classes_and_configs(cls, vllm_config):
                return []

        class Post64096VllmConfig:
            def _resolve_num_hash_lookahead_tokens(self) -> int:
                return 0

            @property
            def num_hash_lookahead_tokens(self) -> int:
                return 0

        def post_replay(num_prompt_tokens, scheduler_block_size, use_eagle, num_hash_lookahead_tokens=0):
            return (num_prompt_tokens - max(1, num_hash_lookahead_tokens),)

        class PostKVCacheCoordinator:
            def cache_blocks(self, request, num_computed_tokens):
                return None

        coord_mod = SimpleNamespace(replay_boundaries=post_replay, KVCacheCoordinator=PostKVCacheCoordinator)
        orig_prop = Post64096VllmConfig.num_hash_lookahead_tokens
        orig_multi_get = Post64096MultiConnector.__dict__["_get_connector_classes_and_configs"]
        orig_cache_blocks = PostKVCacheCoordinator.cache_blocks

        changed = apply_lookahead_block_hashes_patch(
            vllm_config_cls=Post64096VllmConfig,
            kv_connector_base_cls=Post64096KVConnectorBase_V1,
            multi_connector_cls=Post64096MultiConnector,
            factory_cls=SimpleNamespace(),
            kv_transfer_config_cls=SimpleNamespace,
            coordinator_module=coord_mod,
        )
        self.assertFalse(changed)
        self.assertIs(Post64096VllmConfig.num_hash_lookahead_tokens, orig_prop)
        prop_fget = getattr(Post64096VllmConfig.num_hash_lookahead_tokens, "fget", None)
        self.assertFalse(getattr(prop_fget, _PATCH_MARKER, False))
        self.assertIs(Post64096MultiConnector.__dict__["_get_connector_classes_and_configs"], orig_multi_get)
        self.assertIs(coord_mod.replay_boundaries, post_replay)
        self.assertIs(PostKVCacheCoordinator.cache_blocks, orig_cache_blocks)

    def test_live_vllm_config_and_block_hasher_integration(self):
        from vllm.config import KVTransferConfig, VllmConfig
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
            AscendStoreConnector,
        )

        self.assertTrue(callable(getattr(KVConnectorBase_V1, "supports_lookahead_block_hashes", None)))
        self.assertTrue(callable(getattr(MultiConnector, "supports_lookahead_block_hashes", None)))

        class _OptOutConnector(KVConnectorBase_V1):
            @classmethod
            def supports_lookahead_block_hashes(cls, kv_transfer_config) -> bool:
                return False

        registry = getattr(KVConnectorFactory, "_registry", None)
        orig_registry = dict(registry) if isinstance(registry, dict) else None
        fake_registry = {
            "AscendStoreConnector": lambda: AscendStoreConnector,
            "MultiConnector": lambda: MultiConnector,
            "OptOutConnector": lambda: _OptOutConnector,
        }

        def resolve_live_lookahead(ktc) -> int:
            cfg = object.__new__(VllmConfig)
            object.__setattr__(cfg, "speculative_config", _make_spec())
            object.__setattr__(cfg, "kv_transfer_config", ktc)
            object.__setattr__(cfg, "cache_config", SimpleNamespace(enable_prefix_caching=True))
            resolve_fn = getattr(cfg, "_resolve_num_hash_lookahead_tokens", None)
            if callable(resolve_fn):
                return int(resolve_fn())
            return int(cfg.num_hash_lookahead_tokens)

        try:
            with patch.object(
                KVConnectorFactory,
                "get_connector_class",
                side_effect=lambda ktc: fake_registry[ktc.kv_connector](),
            ):
                if orig_registry is not None and isinstance(registry, dict):
                    registry.clear()
                    registry.update(fake_registry)

                ktc_ascend = KVTransferConfig(kv_connector="AscendStoreConnector", kv_role="kv_both")
                ktc_multi = KVTransferConfig(
                    kv_connector="MultiConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={
                        "connectors": [{"kv_connector": "AscendStoreConnector", "kv_role": "kv_both"}]
                    },
                )
                ktc_multi_opt_out = KVTransferConfig(
                    kv_connector="MultiConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={
                        "connectors": [
                            {"kv_connector": "AscendStoreConnector", "kv_role": "kv_both"},
                            {"kv_connector": "OptOutConnector", "kv_role": "kv_both"},
                        ]
                    },
                )

                self.assertTrue(AscendStoreConnector.supports_lookahead_block_hashes(ktc_ascend))
                self.assertTrue(MultiConnector.supports_lookahead_block_hashes(ktc_multi))
                self.assertFalse(MultiConnector.supports_lookahead_block_hashes(ktc_multi_opt_out))

                lookahead = resolve_live_lookahead(ktc_ascend)
                self.assertEqual(lookahead, 1)
                self.assertEqual(resolve_live_lookahead(ktc_multi), 1)
                if not hasattr(VllmConfig, "_resolve_num_hash_lookahead_tokens"):
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
