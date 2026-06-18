#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# MiniMax-M3 — Phase-1 text-only, DENSE full-attention model for vllm-ascend.
# ============================================================================
#
# Target: vllm v0.21.0 / vllm-ascend releases/v0.21.0rc.
#
# There is NO public M3 reference implementation. This file mirrors the
# closest available templates:
#   * vllm deepseek_v2 (sigmoid+bias router / shared experts / dense+MoE mix /
#     FusedMoE wiring / DecoderLayer-Model-ForCausalLM structure / load_weights)
#   * vllm qwen3_moe   (per-head qk_norm: RMSNorm over head_dim on reshaped q/k)
#   * vllm gpt_oss     (swigluoai activation: alpha=1.702, limit=7.0)
#   * vllm-ascend deepseek_v4 (Ascend integration idioms, MixtureOfExperts,
#     load_weights name-remapping, FusedMoE.make_expert_params_mapping)
#
# PHASE-1 SCOPE / DELIBERATE SIMPLIFICATIONS
# ------------------------------------------
#   * Attention is plain GQA dense full attention via vllm.attention.Attention
#     (the Ascend backend is selected automatically). The MSA "lightning
#     indexer" (index_q/k_proj/norm) modules are CREATED so the checkpoint
#     tensors map and load, but they are NEVER called in forward(). They are
#     placeholders for Phase-2 block-sparse attention. See "MSA INDEXER".
#   * Layers 0,1,2 are DENSE MLP (dense_intermediate_size=12288). Layers 3..59
#     are MoE (128 routed experts, top_k=4, sigmoid router + bias, 1 shared
#     expert). Split is driven by config.moe_layer_freq == [0,0,0, 1*57].
#
# MEASURED CHECKPOINT FACTS (from safetensors headers on the w8a8 weights):
#   prefix `language_model.model.` (VL wrapper) -> stripped to `model.` here.
#   * self_attn.{q,k,v,o}_proj.weight  I8 (+ .weight_scale/.weight_offset F32)
#   * self_attn.{q,k}_norm.weight      BF16 [128] (per-head)
#   * self_attn.index_{q,k}_proj.weight BF16, index_{q,k}_norm.weight BF16 [128]
#   * block_sparse_moe.gate.weight     F32 [128,6144]
#   * block_sparse_moe.e_score_correction_bias F32 [128]
#   * block_sparse_moe.experts.E.{w1,w2,w3}.weight I8 (+scale/offset)
#         w1 -> gate_proj, w2 -> down_proj, w3 -> up_proj
#   * block_sparse_moe.shared_experts.{gate_proj,up_proj}.weight I8 (+scale/off)
#   * block_sparse_moe.shared_experts.down_proj.weight BF16  (NOT quantized!)
#
# QUANTIZATION
# ------------
#   int8 w8a8 weights carry sibling `.weight_scale` + `.weight_offset` (F32,
#   per-output-channel). The math is owned by vllm-ascend's modelslim quant
#   layer (vllm_ascend/quantization/modelslim_config.py). Our job here is to
#   build int8 linears WITH quant_config and route .weight_scale/.weight_offset
#   to the right params; bf16 tensors (all norms, index_*, shared down_proj,
#   gate, embed, final norm) bypass quant (quant_config=None on those layers).
#
#   *** LOAD-TIME PREREQUISITE (NOT in this file) ***
#   modelslim_config.py must learn about "minimax_m3" so the FusedMoE / linear
#   quant-method lookup remaps `mlp` -> `block_sparse_moe` and strips expert
#   indices, exactly like the existing "minimax_m2" branch. See the big
#   TODO(verify) block near get_quant_method usage and the report. Without it
#   the int8 experts/attention will not find their quant scheme.

from collections.abc import Iterable
import os
import typing

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SwigluOAIAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config access helpers
# ---------------------------------------------------------------------------
def _text_config(hf_config):
    """Return the text sub-config.

    The top-level config is the VL wrapper (model_type=minimax_m3_vl). All the
    text fields live under `.text_config`. We fall back to the top config so
    this works for a hypothetical text-only checkpoint too.
    """
    return getattr(hf_config, "text_config", hf_config)


def _cfg(config, *names, default=None):
    """Get the first present attribute among `names` from `config`."""
    for n in names:
        if hasattr(config, n) and getattr(config, n) is not None:
            return getattr(config, n)
    return default


# ---------------------------------------------------------------------------
# swigluoai activation on CONTIGUOUS [gate | up] halves
# ---------------------------------------------------------------------------
class MiniMaxM3SwiGLUOAI(nn.Module):
    """GPT-OSS clamped-SwiGLU ("swigluoai") on contiguous gate|up halves.

    Input gate_up: [*, 2*I]  ->  output [*, I].
        gate, up = gate_up.chunk(2, -1)          # first half gate, second up
        out = (clamp(up, +-limit) + 1) * gate' * sigmoid(alpha * gate')
        where gate' = clamp(gate, max=limit)

    Named `act_fn` and used as a standalone module because vllm-ascend's fused
    MoE shared-expert path calls `shared_experts.act_fn(gate_up)` directly
    (ops/fused_moe/fused_moe.py:_shared_experts_part2). MergedColumnParallelLinear
    yields CONTIGUOUS halves (NOT the ::2/1::2 interleave that vllm's
    SwigluOAIAndMul assumes), so we chunk(2) rather than stride-slice.
    TODO(verify): gate-vs-up ordering once generation is checked; swap if wrong.
    """

    def __init__(self, alpha: float = 1.702, limit: float = 7.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.limit = float(limit)

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return (up + 1.0) * glu


# ---------------------------------------------------------------------------
# Dense MLP (layers 0,1,2) and shared-expert MLP — swigluoai activation
# ---------------------------------------------------------------------------
class MiniMaxM3MLP(nn.Module):
    """Gated MLP with the GPT-OSS clamped-SwiGLU ("swigluoai") activation.

    M3 uses hidden_act="swigluoai", swiglu_alpha=1.702, swiglu_limit=7.0.
    `SwigluOAIAndMul` expects the gate/up to be INTERLEAVED on the last dim
    (gate = x[..., ::2], up = x[..., 1::2]) — see vllm activation.py.

    TODO(verify): MergedColumnParallelLinear concatenates [gate, up] as two
    contiguous halves, NOT interleaved. So `SwigluOAIAndMul.forward` (which
    slices ::2 / 1::2) would mismatch the MergedColumnParallelLinear layout.
    GPT-OSS sidesteps this because its fused weight is already interleaved on
    disk. For M3 the checkpoint stores SEPARATE gate_proj / up_proj, so after
    MergedColumnParallelLinear they are contiguous halves. We therefore split
    explicitly and call the clamped-swiglu math directly rather than relying on
    the interleaved ::2/1::2 slicing. Confirm gate vs up ordering once weights
    load (swap if outputs look wrong).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_alpha: float,
        swiglu_limit: float,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        # REQUIRED attribute name `act_fn`: vllm-ascend's fused MoE shared-expert
        # path calls `shared_experts.act_fn(gate_up)` directly. Operates on the
        # contiguous [gate | up] halves produced by MergedColumnParallelLinear.
        self.act_fn = MiniMaxM3SwiGLUOAI(swiglu_alpha, swiglu_limit)

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# MoE block (layers 3..59): 128 routed experts (sigmoid + bias) + 1 shared
# ---------------------------------------------------------------------------
class MiniMaxM3MoE(nn.Module):
    """MiniMax-M3 sparse MoE block.

    Mirrors DeepseekV2MoE/DeepseekV4MoE but with M3 specifics:
      * scoring_func = "sigmoid", use_routing_bias = True
      * e_score_correction_bias [128]  (-> gate.e_score_correction_bias)
      * routed_scaling_factor = 2.0
      * NO expert groups (num_expert_group=None -> use_grouped_topk=False)
      * activation = "swigluoai"
      * 1 shared expert (gate/up int8, down bf16), output = routed + shared
    """

    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = float(
            _cfg(config, "routed_scaling_factor", default=1.0)
        )
        self.swiglu_alpha = float(_cfg(config, "swiglu_alpha", default=1.702))
        self.swiglu_limit = float(_cfg(config, "swiglu_limit", default=7.0))

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()

        # M3 names the routed-expert count num_local_experts; keep deepseek
        # fallbacks for robustness.
        self.n_routed_experts: int = int(
            _cfg(config, "num_local_experts", "n_routed_experts", "num_experts")
        )
        self.n_shared_experts: int = int(
            _cfg(config, "n_shared_experts", default=1)
        )
        self.top_k: int = int(_cfg(config, "num_experts_per_tok", default=4))
        # Expert intermediate. M3 uses `intermediate_size` for experts (3072);
        # deepseek uses `moe_intermediate_size`.
        self.moe_intermediate_size: int = int(
            _cfg(config, "moe_intermediate_size", "intermediate_size", default=3072)
        )
        self.shared_intermediate_size: int = int(
            _cfg(config, "shared_intermediate_size", default=self.moe_intermediate_size)
        )

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        # ---- Router gate ----------------------------------------------------
        # gate.weight is F32 [128, 6144], NOT quantized -> quant_config=None.
        # ReplicatedLinear (router logits are tiny; replicate across TP) matches
        # the deepseek_v4 idiom (it forces fp32 router math via precast).
        self.gate = ReplicatedLinear(
            int(config.hidden_size),
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # Tell the quant/loader path the gate weight should stay fp32.
        self.gate.precast_fp32_weight = True

        use_routing_bias = bool(_cfg(config, "use_routing_bias", default=True))
        if use_routing_bias:
            # Checkpoint key block_sparse_moe.e_score_correction_bias [128] F32
            # is remapped to gate.e_score_correction_bias in load_weights.
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # ---- EPLB / physical-expert bookkeeping (mirrors deepseek) ----------
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        # ---- Shared expert --------------------------------------------------
        # gate_proj/up_proj are int8 (quant_config passed), down_proj is bf16 in
        # the checkpoint. We pass quant_config to the whole MLP; the modelslim
        # quant layer marks down_proj as FLOAT in quant_model_description.json
        # and so loads it unquantized automatically (is_layer_skipped_ascend).
        # TODO(verify): confirm the quant json marks shared_experts.down_proj as
        # FLOAT; if not, down_proj will try int8 and fail. The header shows it
        # is BF16, so the json should say FLOAT.
        self.shared_experts = MiniMaxM3MLP(
            hidden_size=int(config.hidden_size),
            intermediate_size=self.shared_intermediate_size * self.n_shared_experts,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_limit=self.swiglu_limit,
            quant_config=quant_config,
            reduce_results=False,  # we reduce after adding routed output
            prefix=f"{prefix}.shared_experts",
        )

        # ---- Routed experts via FusedMoE -----------------------------------
        # M3: sigmoid scoring + correction bias, no expert grouping.
        scoring_func = str(_cfg(config, "scoring_func", default="sigmoid"))
        # norm_topk_prob defaults False for M3 (sigmoid gate not renormalized);
        # routed_scaling_factor applied to routed output instead.
        # M3 ALWAYS renormalizes top-k routing weights (transformers + SGLang ref).
        # renormalize=True also routes vllm-ascend select_experts to the NPU fused
        # gating kernel (moe_gating_top_k norm_type=1 sigmoid + bias_opt): bias used
        # for SELECTION only, weights = raw sigmoid, then renorm, then *scaling once.
        # renormalize=False fell into _native_select_experts (bias leaks into weights,
        # no renorm) -> over-amplified routed output in all 57 MoE layers -> garbage.
        renormalize = True
        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=self.top_k,
            hidden_size=int(config.hidden_size),
            intermediate_size=self.moe_intermediate_size,
            renormalize=renormalize,
            quant_config=quant_config,
            # No expert groups for M3.
            use_grouped_topk=False,
            num_expert_group=None,
            topk_group=None,
            prefix=f"{prefix}.experts",
            scoring_func=scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            activation="swigluoai",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits computed by the gate (fp32) — mirror deepseek_v4 which
        # runs the gate in fp32 explicitly for numerical stability.
        router_logits = torch.nn.functional.linear(
            hidden_states.float(), self.gate.weight
        )

        fused_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        # FusedMoE returns either a plain tensor (routed only) or a
        # (shared_out, routed_out) tuple when shared_experts is fused.
        # IMPORTANT: vLLM FusedMoE ALREADY applies routed_scaling_factor to the
        # routed output internally (it scales topk_weights; vllm-ascend passes
        # routed_scaling_factor into fused_experts). Multiplying again here was a
        # DOUBLE-APPLY bug (routed contribution became x4 instead of x2) that
        # corrupted every MoE layer's output -> garbage generation. So we only
        # add the (unscaled) shared-expert output; do NOT re-scale routed.
        if isinstance(fused_out, tuple):
            shared_output, final_hidden_states = fused_out
            if shared_output is not None:
                final_hidden_states = final_hidden_states + shared_output
            if self.tp_size > 1:
                final_hidden_states = (
                    self.experts.maybe_all_reduce_tensor_model_parallel(
                        final_hidden_states
                    )
                )
        else:
            final_hidden_states = fused_out

        return final_hidden_states.view(num_tokens, hidden_dim)


# ---------------------------------------------------------------------------
# Attention — dense GQA full attention + per-head qk_norm + partial rotary.
# ---------------------------------------------------------------------------
class MiniMaxM3Attention(nn.Module):
    """GQA attention.

    q_proj out = 8192 (64 heads * 128), k/v_proj out = 512 (4 kv heads * 128),
    o_proj in = 8192. head_dim 128. partial rotary 0.5 -> rotary_dim 64.

    Per-head qk_norm: GemmaRMSNorm over head_dim=128 applied to q/k reshaped to
    [*, num_heads, 128] (pattern from qwen3_moe, norm class from gemma3).

    NOTE(weights): q/k/v/o_proj are int8 w8a8 (quant_config). q_norm/k_norm are
    bf16. We keep q_proj/k_proj/v_proj as SEPARATE ColumnParallelLinear layers
    (NOT fused QKVParallelLinear) because per-head qk_norm must be applied to q
    and k separately AND the checkpoint stores them separately with their own
    int8 scales/offsets — fusing would require fusing the scales too. This is
    the simplest correct mapping.
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        max_position_embeddings: int,
        rope_theta: float,
        partial_rotary_factor: float,
        rms_norm_eps: float,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = hidden_size
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(_cfg(config, "head_dim", default=128))

        assert self.total_num_heads % tp_size == 0, (
            f"num_heads {self.total_num_heads} not divisible by tp {tp_size}"
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Fused QKV projection (int8 w8a8). QKVParallelLinear correctly handles
        # GQA when total_num_kv_heads (4) < tp_size (8) by REPLICATING kv heads
        # across ranks. Plain ColumnParallelLinear would evenly shard the
        # 512-wide k/v into 64/rank under tp=8, splitting a 128-dim head across
        # ranks -> `k.view(*, k.shape[-1]//128, 128)` gets a 0 dim -> crash.
        # The checkpoint stores SEPARATE q/k/v_proj (each int8 + scale/offset);
        # they load into the fused qkv via packed_modules_mapping + the
        # ("qkv_proj", "{q,k,v}_proj", "{q,k,v}") stacked mapping in load_weights.
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head qk_norm (bf16). Gemma (1+w) semantics per use_gemma_norm.
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)

        # Partial rotary: rotary_dim = head_dim * partial_rotary_factor = 64.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters={
                "rope_theta": rope_theta,
                "rope_type": "default",
                "partial_rotary_factor": partial_rotary_factor,
            },
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # ---- MSA INDEXER (Phase-2 ONLY — created so weights map; NOT used) ---
        # These exist for layers 3..59 in the checkpoint. They are bf16 and NOT
        # quantized. We build them as ReplicatedLinear (no TP) + per-head
        # GemmaRMSNorm so the tensors load, but forward() does NOT call them.
        # Phase 2 (block-sparse lightning indexer) will wire them in.
        sa_cfg = getattr(config, "sparse_attention_config", None) or {}
        self.has_index = bool(sa_cfg.get("use_sparse_attention", False)) and (
            self._layer_has_index(prefix, sa_cfg)
        )
        if self.has_index:
            index_dim = int(sa_cfg.get("sparse_index_dim", 128))
            num_index_heads = int(sa_cfg.get("sparse_num_index_heads", 4))
            # index_q_proj.weight [512,6144] = num_index_heads(4) * index_dim(128)
            self.index_q_proj = ReplicatedLinear(
                hidden_size,
                num_index_heads * index_dim,
                bias=False,
                quant_config=None,  # bf16, not quantized
                prefix=f"{prefix}.index_q_proj",
            )
            # index_k_proj.weight [128,6144] = index_dim (single shared index k)
            self.index_k_proj = ReplicatedLinear(
                hidden_size,
                index_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.index_k_proj",
            )
            self.index_q_norm = GemmaRMSNorm(index_dim, eps=rms_norm_eps)
            self.index_k_norm = GemmaRMSNorm(index_dim, eps=rms_norm_eps)
            self._msa_nih = int(num_index_heads)
            self._msa_idim = int(index_dim)
        else:
            self.index_q_proj = None
            self.index_k_proj = None
            self.index_q_norm = None
            self.index_k_norm = None

    @staticmethod
    def _layer_has_index(prefix: str, sa_cfg: dict) -> bool:
        """Layers 0,1,2 are dense full attn with NO index_* tensors; 3..59 have
        them. Driven by sparse_attention_config.sparse_attention_freq
        == [0,0,0, 1*57]."""
        try:
            layer_idx = int(prefix.split(".")[-2])  # ...layers.N.self_attn
        except (ValueError, IndexError):
            return False
        freq = sa_cfg.get("sparse_attention_freq")
        if isinstance(freq, (list, tuple)) and layer_idx < len(freq):
            return bool(freq[layer_idx])
        # Fallback: first 3 dense, rest sparse.
        return layer_idx >= 3

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head qk_norm over head_dim (reshape -> norm -> reshape back).
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)

        # PHASE 1: dense full attention only. The MSA indexer (self.index_*) is
        # intentionally NOT invoked here. Phase 2 will compute index scores and
        # gather a sparse KV subset before / instead of this dense call.
        if self.has_index:
            _iq = self.index_q_proj(hidden_states)[0].view(-1, self._msa_nih, self._msa_idim)
            _ik = self.index_k_proj(hidden_states)[0].view(-1, 1, self._msa_idim)
            _iq = self.index_q_norm(_iq)
            _ik = self.index_k_norm(_ik)
            _impl = getattr(self.attn, "impl", None)
            if _impl is not None:
                _impl._msa_iq = _iq
                _impl._msa_ik = _ik
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------
class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = int(config.hidden_size)
        rms_norm_eps = float(_cfg(config, "rms_norm_eps", default=1e-6))
        max_position_embeddings = int(
            _cfg(config, "max_position_embeddings", default=1048576)
        )
        rope_theta = float(_cfg(config, "rope_theta", default=5_000_000.0))
        # partial_rotary_factor 0.5 -> rotary_dim 64 (head_dim 128).
        partial_rotary_factor = float(
            _cfg(config, "partial_rotary_factor", default=0.5)
        )

        layer_idx = int(prefix.split(".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = MiniMaxM3Attention(
            config=config,
            hidden_size=self.hidden_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            rms_norm_eps=rms_norm_eps,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # Dense vs MoE: moe_layer_freq == [0,0,0, 1*57]. 1 -> MoE, 0 -> dense.
        if self._is_moe_layer(config, layer_idx):
            self.mlp = MiniMaxM3MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            dense_intermediate = int(
                _cfg(
                    config,
                    "dense_intermediate_size",
                    "intermediate_size",
                    default=12288,
                )
            )
            self.mlp = MiniMaxM3MLP(
                hidden_size=self.hidden_size,
                intermediate_size=dense_intermediate,
                swiglu_alpha=float(_cfg(config, "swiglu_alpha", default=1.702)),
                swiglu_limit=float(_cfg(config, "swiglu_limit", default=7.0)),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        # All norms use Gemma (1+w) semantics (use_gemma_norm=true).
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            self.hidden_size, eps=rms_norm_eps
        )

    @staticmethod
    def _is_moe_layer(config, layer_idx: int) -> bool:
        freq = _cfg(config, "moe_layer_freq")
        if isinstance(freq, (list, tuple)) and layer_idx < len(freq):
            return bool(freq[layer_idx])
        # Fallback: first_k_dense_replace dense layers then MoE.
        first_dense = _cfg(config, "first_k_dense_replace", default=3)
        return layer_idx >= int(first_dense)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm transformer block with fused add-RMSNorm residual passing,
        # identical structure to qwen3_moe / deepseek.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@support_torch_compile
class MiniMaxM3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        config = _text_config(hf_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        self.vocab_size = int(config.vocab_size)

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            int(config.num_hidden_layers),
            lambda prefix: MiniMaxM3DecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(
                config.hidden_size,
                eps=float(_cfg(config, "rms_norm_eps", default=1e-6)),
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided."
                    )
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        _dbg = bool(os.environ.get('MM3_DEBUG')) and hidden_states.shape[0] > 1
        if _dbg:
            try:
                _h = hidden_states.float()
                print(f'[MM3DBG] T={hidden_states.shape[0]} embed hs_norm={_h.norm().item():.4f} absmax={_h.abs().max().item():.4f}', flush=True)
            except Exception as _e:
                print(f'[MM3DBG] embed err {_e}', flush=True)
        for _li, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if _dbg and (_li < 6 or _li % 10 == 0 or _li >= 57):
                try:
                    _h = hidden_states.float(); _r = residual.float()
                    print(f'[MM3DBG] L{_li:02d} hs_norm={_h.norm().item():.4f} hs_absmax={_h.abs().max().item():.4f} res_norm={_r.norm().item():.4f} res_absmax={_r.abs().max().item():.4f}', flush=True)
                except Exception as _e:
                    print(f'[MM3DBG] L{_li} err {_e}', flush=True)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ---------------------------------------------------------------------------
# MixtureOfExperts metadata mixin (mirrors deepseek_v2/v4)
# ---------------------------------------------------------------------------
class MiniMaxM3MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[MiniMaxM3MoE]

    def extract_moe_parameters(self, example_moe: MiniMaxM3MoE | None):
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning("MiniMaxM3: no MoE layer found in model.layers.")
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


# ---------------------------------------------------------------------------
# ForCausalLM
# ---------------------------------------------------------------------------
class AscendMiniMaxM3ForCausalLM(
    nn.Module,
    SupportsPP,
    MiniMaxM3MixtureOfExperts,
    SupportsLoRA,
):
    # Fused gate_up_proj is the only packed module for dense MLP / shared
    # experts. Attention q/k/v are kept SEPARATE (see MiniMaxM3Attention note),
    # so no qkv_proj packing here. The routed experts (w1/w2/w3) are handled
    # by FusedMoE.make_expert_params_mapping, not packed_modules_mapping.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    model_cls = MiniMaxM3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        config = _text_config(hf_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        # tie_word_embeddings is False for M3, but guard anyway.
        if bool(_cfg(config, "tie_word_embeddings", default=False)):
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_expert_groups = 1  # M3 has no expert groups.
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, MiniMaxM3DecoderLayer)
            if isinstance(layer.mlp, MiniMaxM3MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id)
        # Checkpoint stores experts as ...experts.E.{gate_proj,down_proj,up_proj}
        # AFTER our w1->gate_proj / w2->down_proj / w3->up_proj rename in
        # load_weights, so we map by the post-rename names.
        return FusedMoE.make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_local_experts
            if hasattr(self.config, "num_local_experts")
            else self.config.n_routed_experts,
            num_redundant_experts=self.model.num_redundant_experts,
        )

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Full name remap + stacked (gate_up) + FusedMoE expert mapping +
        index_* direct load + bf16/quant split.

        Checkpoint -> model name transformations (applied per weight):
          * strip `language_model.` VL prefix:
                language_model.model.*  -> model.*
                language_model.lm_head.* -> lm_head.*
          * expert weight rename:
                .block_sparse_moe.experts.E.w1. -> .mlp.experts.E.gate_proj.
                .block_sparse_moe.experts.E.w2. -> .mlp.experts.E.down_proj.
                .block_sparse_moe.experts.E.w3. -> .mlp.experts.E.up_proj.
          * shared expert + gate rename:
                .block_sparse_moe.shared_experts. -> .mlp.shared_experts.
                .block_sparse_moe.gate.            -> .mlp.gate.
                .block_sparse_moe.e_score_correction_bias
                    -> .mlp.gate.e_score_correction_bias
          * int8 scale/offset siblings (.weight_scale / .weight_offset) already
            match the modelslim quant-method param names, so they flow through
            the normal weight loaders to the int8 linear/MoE params.

        index_q/k_proj/norm (bf16) load directly into the ReplicatedLinear /
        GemmaRMSNorm placeholder modules created in MiniMaxM3Attention.
        """
        # gate_up fusion for dense MLP + shared experts.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # ---- 1. strip VL wrapper prefix -------------------------------
            if name.startswith("language_model.model."):
                name = name[len("language_model.") :]  # -> model.*
            elif name.startswith("language_model.lm_head."):
                name = name[len("language_model.") :]  # -> lm_head.*
            elif name.startswith("language_model."):
                # Any other language_model.* (defensive).
                name = name[len("language_model.") :]
            # Vision tower / non-text weights (Phase-1 text-only): skip.
            if name.startswith("vision_") or name.startswith("visual.") or (
                name.startswith("model.vision")
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            # ---- 2. MoE module name: block_sparse_moe -> mlp --------------
            # We build MoE under `.mlp.` (deepseek idiom); checkpoint uses
            # `.block_sparse_moe.`. Rename so params match.
            if ".block_sparse_moe." in name:
                name = name.replace(".block_sparse_moe.", ".mlp.")

            # ---- 3. expert weight short names w1/w2/w3 -> proj names -------
            if ".w1." in name:
                name = name.replace(".w1.", ".gate_proj.")
            if ".w2." in name:
                name = name.replace(".w2.", ".down_proj.")
            if ".w3." in name:
                name = name.replace(".w3.", ".up_proj.")

            # ---- 4. router correction bias --------------------------------
            if name.endswith(".mlp.e_score_correction_bias"):
                name = name.replace(
                    ".mlp.e_score_correction_bias",
                    ".mlp.gate.e_score_correction_bias",
                )

            # ---- 5. stacked gate_up (dense MLP + shared experts) ----------
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts (mlp.experts.N.gate_proj) are handled by the
                # expert mapping below; skip them here.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                break
            else:
                # ---- 6. routed experts via expert mapping -----------------
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        typing.Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                else:
                    if is_expert_weight:
                        # Expert weight not mapped to this rank — skip.
                        continue
                    # ---- 7. everything else (direct load) -----------------
                    # Covers: q/k/v/o_proj (+scale/offset), q/k_norm,
                    # index_q/k_proj (+norm), gate.weight, down_proj (dense &
                    # shared & expert handled above), input/post layernorms,
                    # embed_tokens, lm_head, final norm.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        # Unknown / not-yet-supported tensor (e.g. MTP, vision).
                        logger.debug("Skipping unmapped weight: %s", name)
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params
