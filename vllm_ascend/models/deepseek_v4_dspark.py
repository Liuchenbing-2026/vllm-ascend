# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Clean-room DSpark (DeepSeek-V4 semi-autoregressive speculative decoding)
# draft model for the vLLM-Ascend plugin.
#
# The draft weights are embedded inside the DeepSeek-V4 target checkpoint under
# the ``mtp.{stage}.*`` namespace (3 stages). One parallel backbone pass over a
# block of query slots produces base logits for the whole block; a low-rank
# Markov head then refines each position sequentially. The backbone consumes an
# aux hidden built by the target from the mean-over-hc-copies of layers
# [40,41,42], combined by ``main_proj``/``main_norm``.
#
# Written from the DSpark algorithm spec and the vLLM-Ascend DSV4 infrastructure.
import logging
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
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

from vllm_ascend.utils import enable_dsa_cp

from .deepseek_v4 import DeepseekV2DecoderLayer, DeepseekV2MixtureOfExperts

logger = logging.getLogger(__name__)


def _draft_quant_config(vllm_config: VllmConfig) -> QuantizationConfig | None:
    """Quantization config for the draft attention / main_proj projections.

    The W8A8 draft stores these quantized inside the target checkpoint. An rbf16
    draft has been dequantized to bf16, so its projections must be built
    unquantized; the flag lives on the draft's own hf_config.
    """
    draft_hf_config = vllm_config.speculative_config.draft_model_config.hf_config
    if getattr(draft_hf_config, "dspark_mtp_dequantized_to_bf16", False):
        return None
    return vllm_config.quant_config


def _load_global_rotation(vllm_config: VllmConfig) -> torch.Tensor | None:
    """Load the optional QuaRot rotation Q from the target checkpoint.

    Only the W8A8 draft (whose main_proj stores un-rotated weights) needs a
    runtime rotation; the rbf16 draft is already in the rotated basis and has no
    quant description, so this returns None (inert).
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
    rotation_path = Path(vllm_config.model_config.model) / rel
    try:
        return load_file(rotation_path)["global_rotation"]
    except Exception:
        logger.exception("Failed to load DSpark global rotation from %s", rotation_path)
        raise


class DSparkMarkovHead(nn.Module):
    """Low-rank Markov correction head (rank r).

    ``w1`` embeds the previously produced token (shared/target vocabulary) to a
    rank-r vector; ``w2`` projects it back to a bias over the same vocabulary.
    """

    def __init__(self, vocab_size: int, rank: int, prefix: str) -> None:
        super().__init__()
        self.w1 = VocabParallelEmbedding(vocab_size, rank)
        self.w2 = ParallelLMHead(
            vocab_size, rank, quant_config=None, prefix=maybe_prefix(prefix, "w2")
        )
        self.bias_processor = LogitsProcessor(vocab_size)

    def embed(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        return self.w1(prev_token_ids)

    def bias(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.bias_processor(self.w2, embedded)


class DSparkDeepseekV4Model(nn.Module):
    """DSpark parallel draft backbone over the DeepSeek-V4 hc MLA stack."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: PretrainedConfig = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        self.config = config
        quant_config = _draft_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        # 3 decoder stages (n_mtp_layers), not num_nextn_predict_layers.
        self.num_dspark_layers = getattr(config, "n_mtp_layers", 3)

        # main_proj combines the concatenated per-layer aux hidden from the
        # target's dspark_target_layer_ids (each mean-over-hc = hidden wide).
        self.target_layer_ids = list(
            getattr(config, "dspark_target_layer_ids", []) or []
        )
        combine_fan_in = self.hidden_size * max(len(self.target_layer_ids), 1)
        self.main_proj = ReplicatedLinear(
            combine_fan_in,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
            return_bias=False,
        )
        self.main_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # Reuse the DSV4 decoder block for the draft stages. The construction
        # prefix stays in the checkpoint's native ``mtp.{i}`` namespace so quant
        # lookups resolve and extract_dsv4_layer_index offsets the config index
        # past the target stack (making the draft layers dense sliding-window).
        # The nn param path comes from the ModuleDict key: model.layers.{i}.
        self.layers = nn.ModuleDict(
            {
                str(i): DeepseekV2DecoderLayer(
                    vllm_config, f"mtp.{i}", config=config, is_draft_layer=True
                )
                for i in range(self.num_dspark_layers)
            }
        )

        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # Head-level hc reduction (collapse the 4 hc copies -> hidden) before
        # the draft lm_head, mirroring the target's hc_head.
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        self.logits_processor = LogitsProcessor(self.vocab_size)
        self.markov_head = DSparkMarkovHead(
            self.vocab_size,
            getattr(config, "dspark_markov_rank", 256),
            maybe_prefix(prefix, "markov_head"),
        )

        self.register_buffer("global_rotation", None, persistent=False)
        self.set_global_rotation(_load_global_rotation(vllm_config))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def set_global_rotation(self, rotation: torch.Tensor | None) -> None:
        self.global_rotation = rotation

    def _apply_rotation(self, x: torch.Tensor, inverse: bool) -> torch.Tensor:
        q = self.global_rotation.to(torch.float32)
        rot = q.t() if inverse else q
        blocks = x.view(x.shape[0], -1, self.hidden_size).to(torch.float32) @ rot
        return blocks.reshape(x.shape).to(x.dtype)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_norm(main_proj(aux)) -> main_x [T, hidden].

        ``aux_hidden_states`` is [T, len(target_layer_ids)*hidden]. For the rbf16
        draft no rotation is applied (weights already in the rotated basis).
        """
        x = aux_hidden_states
        if self.global_rotation is not None:
            x = self._apply_rotation(x, inverse=True)
        x = self.main_norm(self.main_proj(x))
        if self.global_rotation is not None:
            x = self._apply_rotation(x, inverse=False)
        return x

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Write the combined context hidden (main_x [T, hidden]) as MLA KV into
        each draft stage's sliding-window cache. Context tokens carry no query.
        Runs eager (variable context shapes)."""
        for i in range(self.num_dspark_layers):
            attn = self.layers[str(i)].self_attn
            self._insert_layer_context_kv(
                attn, f"mtp.{i}.self_attn.attn", context_states,
                context_positions, context_slot_mapping,
            )

    def _insert_layer_context_kv(
        self,
        attn: nn.Module,
        rope_layer_name: str,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None,
    ) -> None:
        kv = attn.wkv(main_x)          # [T, head_dim=512]
        kv = attn.kv_norm(kv)
        kv = kv.view(-1, 1, attn.nope_head_dim + attn.rope_head_dim)  # [T,1,512]

        # Profile/dummy run (no slot mapping): projections only, no RoPE/scatter
        # (the DSA rope registry is not populated outside a real forward).
        if context_slot_mapping is None:
            return

        from vllm_ascend.device.device_op import DeviceOperator
        from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa

        cos_proxy, sin_proxy = get_cos_and_sin_dsa(context_positions)
        cos, sin = cos_proxy[rope_layer_name], sin_proxy[rope_layer_name]
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1), cos, sin, rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        swa = attn.swa_cache_layer
        slot_mapping = DeviceOperator.format_dsa_slot_mapping(
            context_slot_mapping, swa.block_size
        )
        DeviceOperator.dsa_kv_compress_scatter(swa.kv_cache, kv, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # hc-expand the query embeds to the hyper-connection residual stream.
        hidden_states = inputs_embeds.unsqueeze(1).repeat(1, self.hc_mult, 1)
        residual = None
        for i in range(self.num_dspark_layers):
            hidden_states, residual = self.layers[str(i)](
                positions, hidden_states, residual, None
            )
        return hidden_states.flatten(1)  # [T, hc_mult*hidden]

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse the 4 hc copies to hidden via a learned gated weighted sum."""
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        mixes = F.linear(x, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)


class DSparkDeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts):
    """Causal-LM wrapper exposing the DFlash/proposer contract."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Aliased to the target embedding/lm_head by the proposer's sharing.
        self.lm_head = self.model.embed_tokens
        self.logits_processor = self.model.logits_processor
        self.draft_id_to_target_id = None
        self.mask_hidden = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def precompute_and_store_context_kv(
        self, context_states, context_positions, context_slot_mapping=None
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def _head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden is [N, hc_mult*hidden]; collapse hc, norm, project with lm_head.
        xc = self.model.hc_head(hidden.view(-1, self.model.hc_mult, self.model.hidden_size))
        return self.logits_processor(self.lm_head, self.model.norm(xc))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self._head_logits(hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._head_logits(hidden_states)

    def markov_embed(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(prev_token_ids)

    def markov_bias(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(embedded)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids  # shared vocab: identity

    def set_global_rotation(self, rotation: torch.Tensor | None) -> None:
        self.model.set_global_rotation(rotation)

    def _remap_dspark_name(self, name: str) -> str | None:
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
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "markov_head",
        )
        if rest.startswith(head_level):
            rest = rest.replace("markov_w1", "w1").replace("markov_w2", "w2")
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        config = self.config
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        heads_per_rank = config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            model=self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=config.n_routed_experts,
            num_redundant_experts=0,
        )
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name or "t2d" in name:
                continue
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped

            if ".attn_norm." in name:
                name = name.replace(".attn_norm.", ".input_layernorm.")
            if ".ffn_norm." in name:
                name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
            if ".attn." in name and ".self_attn." not in name:
                name = name.replace(".attn.", ".self_attn.")
            if ".ffn." in name:
                name = name.replace(".ffn.", ".mlp.")
            for src, dst in ((".w1.", ".gate_proj."), (".w2.", ".down_proj."), (".w3.", ".up_proj.")):
                name = name.replace(src, dst)
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")
            if ".gate.bias" in name:
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

            if "attn_sink" in name:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                sliced = weight if enable_dsa_cp() else weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(sliced)
                loaded.add(name)
                continue

            handled = False
            if name.startswith("model.layers.") and "mlp.experts." not in name:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    candidate = name.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    params_dict[candidate].weight_loader(params_dict[candidate], weight, shard_id)
                    loaded.add(candidate)
                    handled = True
                    break
            if handled:
                continue

            is_expert = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                is_expert = True
                candidate = name.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                params_dict[candidate].weight_loader(
                    params_dict[candidate], weight, candidate,
                    shard_id=shard_id, expert_id=expert_id, return_success=False,
                )
                loaded.add(candidate)
                break
            if is_expert:
                continue

            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None or name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded.add(name)
        return loaded
