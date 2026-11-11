# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# DSpine draft model for the vLLM-Ascend plugin (main / v1 runner).
#
# DSpine (``Draft in Parallel, Condition Through Depth``) is a DFlash parallel
# draft backbone plus a per-layer "code bridge": after each draft layer a learned
# low-rank write injects a quantized feature of every position into the residual
# stream, so the causal conditioning chain unfolds over network depth while all
# block positions are still drafted in parallel.
#
# Main's ``DFlashQwen3Model`` already resolves per-layer causality and sliding
# windows from the draft config, so this subclass only adds the bridge and the
# pre-write stash. The serial last-write refinement lives in the proposer.
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

from vllm_ascend.models.dspine_math import (
    BRIDGE_PREFIX,
    bridge_layer_write,
    resolve_dspine_spec,
)

logger = init_logger(__name__)


class DSpineCodeBridge(nn.Module):
    """One learned residual write after every active draft layer.

    Parameter names and shapes match the checkpoint export exactly under
    ``selected_prefix_bridge.`` so the weights load without a name map.
    """

    def __init__(self, config) -> None:
        super().__init__()
        spec = resolve_dspine_spec(config)
        self.spec = spec
        self.hidden_size = spec.hidden_size
        self.num_layers = spec.num_layers
        self.code_size = spec.code_size
        self.rank = spec.rank
        self.eps = spec.eps
        self.block_size = spec.block_size
        self.trained_block_size = spec.block_size
        self.active_layers = frozenset(spec.active_layers)
        self.terminal_residual_rank = spec.terminal_residual_rank
        self.head_topk = spec.head_topk

        bf16 = torch.bfloat16
        self.prev = nn.Linear(spec.code_size, spec.rank, bias=False, dtype=bf16)
        self.gate = nn.Linear(spec.code_size, spec.rank, bias=True, dtype=bf16)
        self.predecessor_gate = nn.Linear(spec.code_size, spec.rank, bias=False, dtype=bf16)
        # Terminal writes are disabled in DSpine checkpoints; the module is kept so
        # the checkpoint's (zero) projection still has a destination.
        self.out = nn.Linear(spec.rank, spec.code_size, bias=False, dtype=bf16)
        self.register_buffer("codebook", torch.empty(spec.vocab_size, spec.code_size, dtype=bf16))
        self.register_buffer("read_weight", torch.empty(spec.code_size, spec.hidden_size, dtype=bf16))
        self.register_buffer(
            "early_read_weights",
            torch.empty(spec.num_layers - 1, spec.code_size, spec.hidden_size, dtype=bf16),
        )
        self.feature_write_weights = nn.Parameter(
            torch.empty(spec.num_layers, spec.hidden_size, spec.rank, dtype=bf16),
            requires_grad=False,
        )
        residual_rank = spec.terminal_residual_rank
        self.register_buffer(
            "terminal_residual_embedding",
            torch.empty(spec.vocab_size, residual_rank, dtype=bf16) if residual_rank else None,
        )
        self.register_buffer(
            "terminal_residual_projection",
            torch.empty(spec.rank, residual_rank, dtype=bf16) if residual_rank else None,
        )
        for name in ("independent_write_version", "terminal_disabled_version"):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))
        self.register_buffer("packed_weight", None, persistent=False)
        self._prepared = False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strict loader: every bridge tensor must be present with the right shape."""
        destinations = {name: t for name, t in self.named_parameters()}
        destinations.update({name: t for name, t in self.named_buffers() if t is not None})
        destinations.pop("packed_weight", None)

        loaded: set[str] = set()
        for name, tensor in weights:
            if name not in destinations:
                raise ValueError(f"Unexpected DSpine code-bridge weight: {name!r}")
            dest = destinations[name]
            if tuple(dest.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"DSpine code-bridge shape mismatch for {name}: {tuple(tensor.shape)} vs {tuple(dest.shape)}"
                )
            dest.data.copy_(tensor)
            loaded.add(name)
        missing = set(destinations) - loaded
        if missing:
            raise ValueError(f"Missing DSpine code-bridge tensors: {sorted(missing)}")
        return loaded

    @torch.no_grad()
    def prepare(self) -> None:
        """Validate the checkpoint and build the fused [3R, C] sender projection."""
        if int(self.independent_write_version.item()) != 1:
            raise ValueError("Unsupported DSpine independent_write_version")
        if int(self.terminal_disabled_version.item()) != 1:
            raise ValueError("Unsupported DSpine terminal_disabled_version")
        if self.out.weight.count_nonzero().item():
            raise ValueError("Terminal-disabled DSpine checkpoint contains a nonzero terminal projection")
        packed = torch.cat((self.gate.weight, self.predecessor_gate.weight, self.prev.weight), dim=0).contiguous()
        self.packed_weight = packed
        self._prepared = True

    def layer_read_weight(self, layer: int) -> torch.Tensor:
        return self.read_weight if layer == self.num_layers - 1 else self.early_read_weights[layer]

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
        next_norm: nn.Module,
        stash: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._prepared:
            raise RuntimeError("DSpine code-bridge weights were not prepared")
        normed, out, state = bridge_layer_write(
            hidden,
            residual,
            input_ids,
            layer,
            self,
            next_norm.weight,
            next_norm.variance_epsilon,
        )
        if stash is not None:
            stash.update(state)
        return normed, out


class DSpineQwen3DecoderLayer(DFlashQwen3DecoderLayer):
    """DFlash layer whose input RMSNorm may already have been applied by a fused code write."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_normed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            if residual is None:
                residual = hidden_states
            return hidden_states, residual
        if input_normed:
            if residual is None:
                raise RuntimeError("A pre-normed DSpine layer input needs its residual")
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DSpineQwen3Model(DFlashQwen3Model):
    """DFlash draft backbone + code bridge.

    The forward returns the last write's normalized hidden states for the whole
    block and records the pre-write state in ``last_write_stash`` for the tail.
    """

    decoder_layer_cls = DSpineQwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix)
        if self.quant_config is not None:
            raise ValueError("The DSpine draft does not support quantized weights")
        self.selected_prefix_bridge = DSpineCodeBridge(self.config)
        self.trained_block_size = int(self.selected_prefix_bridge.trained_block_size)
        self.block_size = int(self.selected_prefix_bridge.block_size)
        self.last_write_stash: dict[str, torch.Tensor] | None = None
        # ``DFlashQwen3Model`` is decorated with ``@support_torch_compile``. The
        # bridge forward keeps Python-side state (the stash) and the refinement
        # tail has data-dependent gathers, so the draft runs eagerly.
        self.do_not_compile = True
        self.compiled = False

    def set_block_size(self, block_size: int) -> None:
        """Serve with ``block_size`` rows per request block (2..trained block size)."""
        block = int(block_size)
        if not 2 <= block <= self.trained_block_size:
            raise ValueError(
                f"DSpine serves block sizes 2..{self.trained_block_size} (the trained block size); got {block}"
            )
        self.block_size = block
        self.selected_prefix_bridge.block_size = block

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds
        residual: torch.Tensor | None = None
        input_normed = False
        last = len(self.layers) - 1
        bridge = self.selected_prefix_bridge
        self.last_write_stash = None

        for index, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                input_normed=input_normed,
            )
            input_normed = False
            if index in bridge.active_layers and hidden_states.numel():
                next_norm = self.layers[index + 1].input_layernorm if index < last else self.norm
                stash: dict[str, torch.Tensor] | None = {} if index == last else None
                hidden_states, residual = bridge(
                    hidden_states,
                    residual,
                    input_ids,
                    index,
                    next_norm,
                    stash,
                )
                if stash is not None:
                    self.last_write_stash = stash
                input_normed = True

        if hidden_states.numel() and not input_normed:
            raise RuntimeError("DSpine's last layer must end with a (norm-fused) code write")
        return hidden_states


class DSpineQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """Causal-LM wrapper exposing the DFlash proposer contract to vLLM."""

    model_cls = DSpineQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # The checkpoint ships no embed_tokens / lm_head; both are shared with the
        # target model by the proposer (``has_own_lm_head=False`` is what the main
        # runner checks before sharing).
        self.has_own_lm_head = False
        self.has_own_embed_tokens = False

    def get_draft_attn_causal(self) -> list[bool]:
        # DSpine checkpoints are trained with causal block attention
        # (draft_block_causal); the config patch also sets is_causal for the
        # per-layer attention modules.
        return [True] * len(self.model.layers)

    def _read_mask_embedding(self) -> torch.Tensor | None:
        # DSpine checkpoints never ship the optional mask_embedding.pt override.
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        def backbone_weights():
            for name, tensor in weights:
                short = name.removeprefix("model.")
                if short in ("draft_block_causal_version", "last_write_refine_version"):
                    if int(tensor.item()) != 1:
                        raise ValueError(f"Unsupported DSpine checkpoint version tensor: {short}={int(tensor.item())}")
                    continue
                yield name, tensor

        super().load_weights(backbone_weights())
        bridge = self.model.selected_prefix_bridge
        bridge.prepare()
        logger.info(
            "Loaded DSpine draft: layers=%d active_layers=%s code_size=%d rank=%d head_topk=%d "
            "block_size=%d terminal_residual_rank=%d",
            bridge.num_layers,
            sorted(i + 1 for i in bridge.active_layers),
            bridge.code_size,
            bridge.rank,
            bridge.head_topk,
            bridge.block_size,
            bridge.terminal_residual_rank,
        )