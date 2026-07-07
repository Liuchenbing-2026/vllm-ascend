# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Clean-room DSpark (DeepSeek-V4 semi-autoregressive speculative decoding)
# draft model for the vLLM-Ascend plugin.
#
# The draft weights are embedded inside the DeepSeek-V4 target checkpoint under
# the ``mtp.{i}.*`` namespace. A single parallel backbone pass produces base
# logits for a whole block of draft positions (non-causal within the block),
# and a low-rank Markov head refines each position sequentially with a
# prefix-dependent bias (see ``DSparkDeepseekV4Proposer._sample_sequential``).
#
# This module is written from the published DSpark algorithm description and
# the vLLM-Ascend DSV4 / DFlash infrastructure; it does not derive from any
# third-party port.
import logging
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import PretrainedConfig
from vllm.config import VllmConfig

logger = logging.getLogger(__name__)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix

from .deepseek_v4 import DeepseekV2DecoderLayer, DeepseekV2MixtureOfExperts


def _draft_quant_config(vllm_config: VllmConfig) -> QuantizationConfig | None:
    """Quantization config for the draft submodules.

    The DSpark draft weights (attention projections, ``main_proj``) are stored
    W8A8 inside the quantized target checkpoint, so we keep them quantized
    rather than silently reading them as bf16 (which corrupts the draft output).

    Exception: an rbf16 draft has been dequantized to bf16, so its projections
    must be built unquantized regardless of the target's quantization.
    """
    draft_hf_config = vllm_config.model_config.hf_config
    if getattr(draft_hf_config, "dspark_mtp_dequantized_to_bf16", False):
        return None
    return vllm_config.quant_config


def _load_global_rotation(vllm_config: VllmConfig) -> torch.Tensor | None:
    """Load the QuaRot global rotation matrix Q from the target checkpoint.

    The path is declared by the quant description at
    ``optional/quarot/rotation_map/global_rotation`` and is relative to the
    model directory, matching the convention used for Eagle3 QuaRot drafts.
    Returns None when the checkpoint is not QuaRot-rotated.
    """
    quant_config = vllm_config.quant_config
    if quant_config is None:
        return None
    quant_description = getattr(quant_config, "quant_description", None)
    if not quant_description:
        return None
    try:
        rel = quant_description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    except (KeyError, TypeError):
        return None
    model_path = Path(vllm_config.model_config.model)
    rotation_path = model_path / rel
    try:
        return load_file(rotation_path)["global_rotation"]
    except Exception:
        logger.exception("Failed to load DSpark global rotation from %s", rotation_path)
        raise


class DSparkMarkovHead(nn.Module):
    """Low-rank Markov correction head.

    ``w1`` embeds the previously produced token (indexed in the *target*
    vocabulary) to a rank-``r`` vector; ``w2`` projects it back to a bias over
    the *draft* vocabulary. The two vocabularies differ only when the
    checkpoint ships a reduced draft vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.w1 = VocabParallelEmbedding(vocab_size, rank)
        self.w2 = ParallelLMHead(
            draft_vocab_size,
            rank,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "w2"),
        )
        self.bias_processor = LogitsProcessor(draft_vocab_size)

    def embed(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        return self.w1(prev_token_ids)

    def bias(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.bias_processor(self.w2, embedded)


class DSparkDeepseekV4Model(nn.Module):
    """DSpark parallel draft backbone over the DeepSeek-V4 MLA stack."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: PretrainedConfig = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        self.config = config
        quant_config = _draft_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        self.num_dspark_layers = getattr(config, "num_nextn_predict_layers", 1)
        self.hc_mult = getattr(config, "hc_mult", 1)
        self.rms_norm_eps = config.rms_norm_eps

        # Target layers whose hidden states are concatenated as the drafter's
        # context input. Falls back to a single (already-combined) stream.
        self.target_layer_ids = list(
            getattr(config, "dspark_target_layer_ids", []) or []
        )
        combine_fan_in = self.hidden_size * max(len(self.target_layer_ids), 1)

        # bug#4: main_proj is W8A8 in our checkpoint; keep it quantized.
        self.main_proj = ReplicatedLinear(
            combine_fan_in,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
            return_bias=False,
        )
        self.main_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # Reuse the target's DSV4 decoder block for the draft layers. The
        # construction prefix stays in the checkpoint's native ``mtp.{i}``
        # namespace so quant-description lookups hit the checkpoint's own keys
        # and extract_dsv4_layer_index offsets the config-array index past the
        # target stack. The nn module path comes from the ModuleDict key, so the
        # parameters live at ``model.layers.{i}.*`` (matching _remap_dspark_name).
        self.layers = nn.ModuleDict(
            {
                str(i): DeepseekV2DecoderLayer(
                    vllm_config,
                    f"mtp.{i}",
                    config=config,
                    is_draft_layer=True,
                )
                for i in range(self.num_dspark_layers)
            }
        )

        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        self.logits_processor = LogitsProcessor(self.vocab_size)

        self.markov_head = DSparkMarkovHead(
            self.vocab_size,
            self.draft_vocab_size,
            getattr(config, "dspark_markov_rank", 256),
            maybe_prefix(prefix, "markov_head"),
            quant_config=quant_config,
        )

        # bug#5: the W8A8 checkpoint is QuaRot-rotated (target hidden + draft
        # attention are rotated) but main_proj stores *un-rotated* weights. The
        # global rotation matrix Q is loaded from the target checkpoint and used
        # to bring the incoming (rotated) hidden states into main_proj's basis
        # and back. Absent Q (non-QuaRot checkpoint), the alignment is a no-op.
        self.register_buffer("global_rotation", None, persistent=False)
        self.set_global_rotation(_load_global_rotation(vllm_config))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def set_global_rotation(self, rotation: torch.Tensor | None) -> None:
        self.global_rotation = rotation

    def combine_hidden_states(self, context_states: torch.Tensor) -> torch.Tensor:
        """Project + norm the concatenated target hidden states.

        Order is proj-then-norm. When a QuaRot rotation is present the rotated
        input is de-rotated per hidden block before ``main_proj`` and the result
        is re-rotated after ``main_norm`` so the drafter's KV basis matches the
        rotated attention weights.
        """
        x = context_states
        if self.global_rotation is not None:
            x = self._apply_rotation(x, inverse=True)
        x = self.main_proj(x)
        x = self.main_norm(x)
        if self.global_rotation is not None:
            x = self._apply_rotation(x, inverse=False)
        return x

    def _apply_rotation(self, x: torch.Tensor, inverse: bool) -> torch.Tensor:
        q = self.global_rotation.to(torch.float32)
        rot = q.t() if inverse else q
        blocks = x.view(x.shape[0], -1, self.hidden_size).to(torch.float32)
        blocks = blocks @ rot
        return blocks.reshape(x.shape).to(x.dtype)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Project context hidden states to K/V and write the draft KV cache.

        ``context_states`` is the *already-combined* draft hidden stream (the
        proposer applies ``combine_hidden_states`` once, before this call). The
        context tokens carry no query, so this runs the KV-producer half of each
        draft layer's MLA path (wkv -> kv_norm -> RoPE -> scatter into the
        sliding-window cache) once per layer. Variable context shapes keep this
        out of the captured graph; it runs eager. When ``context_slot_mapping``
        is None (dummy/profile run) the projections run but nothing is written.
        """
        combined = context_states
        for i in range(self.num_dspark_layers):
            attn = self.layers[str(i)].self_attn
            self._insert_layer_context_kv(
                attn, combined, context_positions, context_slot_mapping
            )

    def _insert_layer_context_kv(
        self,
        attn: nn.Module,
        combined: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None,
    ) -> None:
        """Write one draft layer's context K/V into its sliding-window cache.

        Mirrors the KV-producer tail of the DSA MLA path
        (``DSAAttention._mla_prolog_multistream``): wkv projection, kv RMSNorm,
        partial RoPE, then ``DeviceOperator.dsa_kv_compress_scatter`` into the
        layer's SWA cache. Runs eager because the context shape varies per step.

        NOTE (on-device bring-up): the RoPE cos/sin registry
        (``get_cos_and_sin_dsa``) is populated by the attention metadata builder;
        the exact group key for the draft SWA layer must be confirmed on NPU.
        This is the single hardware-coupled seam in the port.
        """
        # Lazy imports keep NPU init out of model-inspection subprocesses.
        from vllm_ascend.device.device_op import DeviceOperator
        from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa

        kv = attn.wkv(combined)
        kv = attn.kv_norm(kv)
        kv = kv.view(-1, 1, attn.nope_head_dim + attn.rope_head_dim)

        cos_sin = get_cos_and_sin_dsa(context_positions)
        cos, sin = self._select_layer_cos_sin(cos_sin, attn)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )

        if context_slot_mapping is None:
            return
        DeviceOperator.dsa_kv_compress_scatter(
            attn.swa_cache_layer.kv_cache, kv, context_slot_mapping
        )

    @staticmethod
    def _select_layer_cos_sin(cos_sin, attn):
        """Pick the (cos, sin) pair for this draft layer's default rope group."""
        # get_cos_and_sin_dsa returns {config_key: {group_name: (cos, sin)}} or a
        # single (cos, sin). Draft SWA layers use the "default" rope group.
        if isinstance(cos_sin, tuple):
            return cos_sin
        for groups in cos_sin.values():
            if "default" in groups:
                return groups["default"]
            return next(iter(groups.values()))
        raise RuntimeError("No rope cos/sin registered for the DSpark draft layer")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        residual = None
        for i in range(self.num_dspark_layers):
            hidden_states, residual = self.layers[str(i)](
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.embed_tokens, self.norm(hidden_states))

    # --- Markov head hooks used by the proposer's serial sampling loop ---
    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.embed_tokens, self.norm(hidden_states))

    def markov_embed(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(prev_token_ids)

    def markov_bias(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(embedded)


class DSparkDeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts):
    """Causal-LM wrapper exposing the contract the DSpark proposer relies on."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Aliased to the target lm_head/embedding by the proposer's sharing
        # logic; DSpark uses the full target vocabulary, so no d2t remap.
        self.lm_head = self.model.embed_tokens
        self.logits_processor = self.model.logits_processor
        self.draft_id_to_target_id = None
        # DFlash-family drafters embed the mask/noise token instead of a learned
        # mask hidden state, so no mask_hidden parameter is needed.
        self.mask_hidden = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_draft_logits(hidden_states)

    def markov_embed(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(prev_token_ids)

    def markov_bias(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.model.markov_bias(embedded)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: identity. A reduced-vocab checkpoint would add its
        # draft_id_to_target_id offset here.
        return draft_ids

    def set_global_rotation(self, rotation: torch.Tensor | None) -> None:
        self.model.set_global_rotation(rotation)

    def _remap_dspark_name(self, name: str) -> str | None:
        """Map ``mtp.{i}.*`` checkpoint keys to draft parameter paths.

        Head/context-combiner parameters live at model level; everything else
        is a per-layer decoder block. ``confidence_head.*`` and non-``mtp.*``
        keys belong to the target and are dropped.
        """
        if not name.startswith("mtp."):
            return None
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
        stage, rest = parts[1], parts[2]
        if rest.startswith("confidence_head"):
            return None

        head_level = (
            "main_proj",
            "main_norm",
            "norm",
            "markov_head",
            "markov_w1",
            "markov_w2",
        )
        if rest.startswith(head_level):
            rest = rest.replace("markov_w1", "markov_head.w1")
            rest = rest.replace("markov_w2", "markov_head.w2")
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            (".w1.", ".gate_proj.", None),
            (".w2.", ".down_proj.", None),
            (".w3.", ".up_proj.", None),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name or "t2d" in name:
                continue
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped

            # bug#3 companion: normalize checkpoint substrings to param names.
            for src, dst in ((".attn.", ".self_attn."), (".ffn.", ".mlp.")):
                if src in name and ".self_attn." not in name:
                    name = name.replace(src, dst)
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")
            if ".gate.bias" in name:
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

            # Only per-layer decoder blocks take the fused/stacked mapping;
            # head params (e.g. markov_head.w1) must never hit the w1 rule.
            handled = False
            if name.startswith("model.layers."):
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    candidate = name.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    param = params_dict[candidate]
                    if shard_id is None:
                        param.weight_loader(param, weight)
                    else:
                        param.weight_loader(param, weight, shard_id)
                    loaded.add(candidate)
                    handled = True
                    break
            if handled:
                continue

            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None or name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded.add(name)
        return loaded
