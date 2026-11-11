# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Pure-torch math for DSpine (``Draft in Parallel, Condition Through Depth``).
#
# DSpine serves a DFlash-style parallel draft backbone plus a "code bridge": after
# every draft layer a small learned residual write passes a quantized feature of
# each position to the next layer. The refinement tail then redoes the *last*
# write with the exact candidate token's code for the predecessor ("exact
# sender") and walks a K-lane chain to produce the block proposals.
#
# The reference implementation expresses the tail with fused CUDA/Triton kernels
# (``sglang_dspine/kernels``) and a vocabulary-wide table pair (TG/TSV). These
# helpers are the same math in plain torch so it runs on Ascend NPUs, and they
# stay free of vLLM imports so they can be unit-tested on CPU.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

# ``torch.finfo(torch.float32).tiny``: the clamp floor of the bridge read RMS,
# identical to the reference implementation.
_F32_TINY = 1.1754943508222875e-38
BRIDGE_PREFIX = "selected_prefix_bridge."


def get_dflash_config(config: Any) -> dict[str, Any]:
    """Return the ``dflash_config`` mapping of a draft HF config (possibly EAGLEConfig)."""
    cfg = getattr(config, "dflash_config", None)
    if isinstance(cfg, dict):
        return cfg
    inner = getattr(config, "model", None)
    cfg = getattr(inner, "dflash_config", None)
    return cfg if isinstance(cfg, dict) else {}


def is_dspine_hf_config(hf_config: Any) -> bool:
    """True for DSpine draft checkpoints (a code bridge in ``dflash_config``)."""
    if hf_config is None:
        return False
    dflash = get_dflash_config(hf_config)
    return isinstance(dflash.get("selected_prefix_code_bridge"), dict)


def resolve_head_topk(config: Any, num_layers: int) -> int:
    """The refinement width K.

    The checkpoint states it either as ``code_bridge_second_pass.head_topk`` or as
    ``dflash_config.selected_prefix_last_write_refine.head_topk`` (the released
    checkpoints use the latter). Both must agree when present.
    """
    dflash_cfg = get_dflash_config(config)
    found: dict[str, int] = {}
    second = getattr(config, "code_bridge_second_pass", None)
    if second is not None:
        second = dict(second)
        unknown = set(second) - {"mode", "first_exact", "head_topk"}
        if unknown:
            raise ValueError(f"Unknown code_bridge_second_pass keys: {sorted(unknown)}")
        if second.get("mode") != "serial_table":
            raise ValueError(
                f"DSpine serves the serial last-write refinement only; the checkpoint config says "
                f"code_bridge_second_pass.mode={second.get('mode')!r}"
            )
        if int(second.get("first_exact", -1)) != num_layers - 1:
            raise ValueError(
                f"serial_table redoes the LAST write only: code_bridge_second_pass.first_exact must be "
                f"{num_layers - 1}, got {second.get('first_exact')!r}"
            )
        found["code_bridge_second_pass.head_topk"] = int(second["head_topk"])
    refine = dflash_cfg.get("selected_prefix_last_write_refine")
    if refine is not None:
        refine = dict(refine)
        if (
            set(refine) - {"kind", "version", "head_topk", "weight"}
            or not {"kind", "head_topk"} <= set(refine)
            or refine["kind"] != "exact_sender"
            or refine.get("version", 1) != 1
            or (not refine.get("weight", 1.0) >= 0)
        ):
            raise ValueError(
                f"DSpine expects selected_prefix_last_write_refine = "
                f"{{kind: 'exact_sender', version: 1, head_topk, weight >= 0}}, got {refine!r}"
            )
        found["dflash_config.selected_prefix_last_write_refine.head_topk"] = int(refine["head_topk"])
    if not found:
        raise ValueError(
            "DSpine checkpoint config names no refinement width: add code_bridge_second_pass "
            "{mode: serial_table, first_exact, head_topk} or "
            "dflash_config.selected_prefix_last_write_refine.head_topk"
        )
    values = set(found.values())
    if len(values) != 1:
        raise ValueError(f"DSpine refinement widths disagree: {found}")
    k = values.pop()
    if k & (k - 1) or not 2 <= k <= 64:
        raise ValueError(f"DSpine head_topk must be a power of two in [2, 64], got {k}")
    return k


@dataclass(frozen=True)
class DSpineBridgeSpec:
    """Validated geometry of a DSpine code bridge."""

    hidden_size: int
    num_layers: int
    vocab_size: int
    code_size: int
    rank: int
    eps: float
    block_size: int
    active_layers: tuple[int, ...]
    terminal_residual_rank: int
    head_topk: int
    # The 27B drafts are trained with a sliding window on their draft attention;
    # the port currently runs the draft with full attention (see the design doc).
    sliding_window: int | None = None

    @property
    def sqrt_code_size(self) -> float:
        return math.sqrt(self.code_size)


def resolve_dspine_spec(config: Any) -> DSpineBridgeSpec:
    """Parse and validate the DSpine-specific part of a draft config."""
    cfg = get_dflash_config(config)
    if not cfg:
        raise ValueError("DSpine draft config lacks dflash_config")
    bridge = cfg.get("selected_prefix_code_bridge")
    if not isinstance(bridge, dict):
        raise ValueError("DSpine draft config lacks dflash_config.selected_prefix_code_bridge")
    if cfg.get("selected_prefix_terminal", True):
        raise ValueError("DSpine supports terminal-disabled code bridges only")
    if cfg.get("selected_prefix_decode_mode") != "serial":
        raise ValueError(
            f"DSpine requires selected_prefix_decode_mode = 'serial', got {cfg.get('selected_prefix_decode_mode')!r}"
        )
    if cfg.get("selected_prefix_feature_write_mode") != "independent":
        raise ValueError("DSpine requires independent learned writes")
    if cfg.get("selected_prefix_feature_writes") is not True:
        raise ValueError("DSpine requires selected_prefix_feature_writes = true")
    if not cfg.get("draft_block_causal"):
        raise ValueError("DSpine requires causal draft attention (draft_block_causal)")
    if cfg.get("selected_prefix_feature_reliability") is not None:
        raise ValueError("Reliability probes are not part of the DSpine path")
    if cfg.get("selected_prefix_feature_next_token_rows", False):
        raise ValueError("DSpine's serial refinement is defined for anchor-row (DFlash) semantics")
    if cfg.get("conv_kernel_size") or cfg.get("conv_group_size"):
        raise ValueError("DSpine checkpoints do not use grouped draft convolutions")
    for key in ("selector_rank", "selector_top_k", "projector_type", "attention_sink_bias"):
        if cfg.get(key):
            raise ValueError(f"DSpine drafts carry no {key} (DFlash2/Domino/DSpark head)")
    if cfg.get("attention_value_scale") is not None:
        raise ValueError("DSpine drafts do not use attention_value_scale")

    num_layers = int(config.num_hidden_layers)
    layers_raw = cfg.get("selected_prefix_feature_layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ValueError("DSpine draft config lacks dflash_config.selected_prefix_feature_layers")
    active = tuple(int(i) - 1 for i in layers_raw)
    if not all(0 <= i < num_layers for i in active):
        raise ValueError(f"selected_prefix_feature_layers out of range: {sorted(active)}")
    if len(set(active)) != len(active):
        raise ValueError(f"duplicate selected_prefix_feature_layers: {sorted(active)}")
    if num_layers - 1 not in active:
        raise ValueError("DSpine refines the last layer's write, so the last layer must be active")

    code_size = int(bridge["code_size"])
    rank = int(bridge["rank"])
    eps = float(bridge["eps"])
    if code_size <= 0 or rank <= 0 or eps <= 0:
        raise ValueError(f"invalid code bridge geometry: code_size={code_size} rank={rank} eps={eps}")

    block_size = int(getattr(config, "block_size", 0) or 0)
    if block_size < 2:
        raise ValueError(f"DSpine draft block_size must be >= 2, got {block_size}")

    terminal_rank = cfg.get("selected_prefix_terminal_residual_rank", 0)
    if type(terminal_rank) is not int or not 0 <= terminal_rank <= rank:
        raise ValueError(f"selected_prefix_terminal_residual_rank must be an int in [0, rank], got {terminal_rank!r}")
    if terminal_rank and cfg.get("selected_prefix_last_write_refine") is None:
        raise ValueError("The terminal token residual needs the exact-code last-write refinement")

    return DSpineBridgeSpec(
        hidden_size=int(config.hidden_size),
        num_layers=num_layers,
        vocab_size=int(config.vocab_size),
        code_size=code_size,
        rank=rank,
        eps=eps,
        block_size=block_size,
        active_layers=active,
        terminal_residual_rank=terminal_rank,
        head_topk=resolve_head_topk(config, num_layers),
        sliding_window=(
            int(getattr(config, "sliding_window", 0) or 0) if getattr(config, "use_sliding_window", False) else None
        ),
    )


def layer_read_weight(bridge: Any, layer: int) -> torch.Tensor:
    """The code-read projection of ``layer`` (the last layer owns ``read_weight``)."""
    if layer == bridge.num_layers - 1:
        return bridge.read_weight
    return bridge.early_read_weights[layer]


@torch.no_grad()
def bridge_layer_write(
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    input_ids: torch.Tensor,
    layer: int,
    bridge: Any,
    next_norm_weight: torch.Tensor,
    next_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """One DSpine code-bridge write after draft ``layer``.

    ``hidden`` is the layer's MLP output [m, H] and ``residual`` the pre-MLP stream
    [m, H] (or None when the caller already summed the layer). Rows are packed as
    whole request blocks of ``bridge.block_size`` rows; the first row of a block is
    the anchor (the bonus token) and receives no write.

    Returns ``(next_norm(out), out, stash)`` where ``out`` is the updated residual
    stream and ``stash`` holds the pre-write state the refinement tail reads:
    ``raw`` [m, H] bf16, ``rms`` [m] fp32, ``packed`` [m, 3R] bf16.
    """
    if hidden.dtype != torch.bfloat16:
        raise ValueError(f"The DSpine code bridge requires BF16 activations, got {hidden.dtype}")
    m, h = hidden.shape
    block = int(bridge.block_size)
    if block < 2 or m % block:
        raise ValueError(f"DSpine code-bridge input must contain whole blocks of {block} rows, got {m}")
    if m == 0:
        empty = hidden.new_empty(0, h)
        return empty, empty, {"raw": empty, "rms": hidden.new_empty(0, dtype=torch.float32), "packed": empty}

    raw = hidden if residual is None else hidden + residual
    raw = raw.to(torch.bfloat16)
    f = raw.float()
    rms = torch.sqrt(torch.clamp(f.pow(2).mean(dim=-1), min=_F32_TINY))

    eps = float(bridge.eps)
    normalized = (f / (rms.unsqueeze(-1) + eps)).to(torch.bfloat16)

    code_size = int(bridge.code_size)
    sqrt_c = math.sqrt(code_size)
    projected = torch.matmul(normalized, layer_read_weight(bridge, layer).t()).float()
    proj_norm = torch.sqrt(torch.sum(projected * projected, dim=-1, keepdim=True))
    codes = (sqrt_c * projected / (proj_norm + eps)).to(torch.bfloat16)

    batch = m // block
    anchors = input_ids.reshape(batch, block)[:, 0]
    anchor_codes = (sqrt_c * bridge.codebook[anchors].float()).to(torch.bfloat16)
    codes = torch.cat([anchor_codes.unsqueeze(1), codes.view(batch, block, code_size)[:, 1:]], dim=1)
    codes = codes.reshape(m, code_size)

    packed = torch.matmul(codes, bridge.packed_weight.t())
    rank = int(bridge.rank)
    q, pred_gate, pred_value = packed.float().split(rank, dim=-1)
    q = q.view(batch, block, rank)
    pred_gate = pred_gate.view(batch, block, rank)
    pred_value = pred_value.view(batch, block, rank)

    # Row 0 of a block is the anchor: it carries no predecessor message. For every
    # other row the predecessor is the previous row of the same block.
    message = torch.zeros_like(q)
    message[:, 1:] = torch.sigmoid(q[:, 1:] + pred_gate[:, :-1] + bridge.gate.bias.float()) * pred_value[:, :-1]
    delta = torch.matmul(message.reshape(m, rank).to(torch.bfloat16), bridge.feature_write_weights[layer].t()).float()
    delta = delta.view(batch, block, h)

    not_anchor = (torch.arange(block, device=hidden.device) != 0).to(f.dtype).view(1, block, 1)
    out = (f.view(batch, block, h) + rms.view(batch, block, 1) * delta * not_anchor).to(torch.bfloat16)

    x = out.float()
    y = (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(next_norm_eps))).to(torch.bfloat16)
    normed = (y.float() * next_norm_weight.float()).to(torch.bfloat16)

    stash = {"raw": raw, "rms": rms, "packed": packed}
    return normed.reshape(m, h), out.reshape(m, h), stash


@torch.no_grad()
def build_serial_tables(
    lm_head_weight: torch.Tensor,
    norm_w: torch.Tensor,
    write_w: torch.Tensor,
    pv_w: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chunk: int = 8192,
    cast_bf16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vocabulary-wide greedy refinement tables (TG [V, R], TSV [V, 2R]).

    ``tg[v] = (head_v * norm_w) @ write_w`` and
    ``tsv[v] = (sqrt(C) * codebook_v) @ [predecessor_gate; prev]^T``.
    ``cast_bf16`` reproduces the reference kernels' rounding points; tests pass
    ``False`` to keep the caller's dtype (e.g. float64) and check the algebra.
    """
    vocab, hidden = lm_head_weight.shape
    rank = write_w.shape[1]
    code_size = codebook.shape[1]
    if write_w.shape != (hidden, rank):
        raise ValueError(f"write_w must be [H, R] = [{hidden}, {rank}], got {tuple(write_w.shape)}")
    if pv_w.shape != (2 * rank, code_size):
        raise ValueError(f"pv_w must be [2R, C] = [{2 * rank}, {code_size}], got {tuple(pv_w.shape)}")
    if codebook.shape != (vocab, code_size):
        raise ValueError(f"codebook must be [V, C] = [{vocab}, {code_size}], got {tuple(codebook.shape)}")

    sqrt_c = math.sqrt(code_size)
    if cast_bf16:
        write_w = write_w.to(torch.bfloat16)
        pv_w = pv_w.to(torch.bfloat16)
        norm_scale = norm_w.float()

    tg_parts: list[torch.Tensor] = []
    tsv_parts: list[torch.Tensor] = []
    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        if cast_bf16:
            rows = (lm_head_weight[start:end].float() * norm_scale).to(torch.bfloat16)
            codes = (codebook[start:end].float() * sqrt_c).to(torch.bfloat16)
        else:
            rows = lm_head_weight[start:end] * norm_w.to(lm_head_weight.dtype)
            codes = codebook[start:end] * sqrt_c
        tg_parts.append(torch.matmul(rows, write_w))
        tsv_parts.append(torch.matmul(codes, pv_w.t()))

    tg = tg_parts[0] if len(tg_parts) == 1 else torch.cat(tg_parts, dim=0)
    tsv = tsv_parts[0] if len(tsv_parts) == 1 else torch.cat(tsv_parts, dim=0)
    return tg, tsv


@torch.no_grad()
def serial_refine_greedy(
    rows: torch.Tensor,
    anchors: torch.Tensor,
    stash_raw: torch.Tensor,
    stash_rms: torch.Tensor,
    stash_packed: torch.Tensor,
    lm_head_weight: torch.Tensor,
    tg: torch.Tensor,
    tsv: torch.Tensor,
    gate_bias: torch.Tensor,
    norm_w: torch.Tensor,
    topk: int,
    block_size: int,
    *,
    return_candidates: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Greedy serial last-write refinement over one draft block per request.

    ``rows`` [bs, T-1, H] are the draft forward's output rows 1..T-1 of each block,
    ``anchors`` [bs] the block's bonus token and the stash tensors the last bridge
    write's pre-write state [bs*T, ...].

    For each proposal row the K candidate tokens come from the draft head; the
    chain walks the KxK lane table (predecessor candidate -> best candidate) to
    pick one token per row. Returns token ids [bs, T-1] in the target vocabulary,
    plus the candidates [bs, T-1, K] when requested.
    """
    batch, steps, hidden = rows.shape
    block = int(block_size)
    if steps != block - 1:
        raise ValueError(f"DSpine refinement expects {block - 1} proposal rows per block, got {steps}")
    rank = int(tg.shape[1])

    raw = stash_raw.reshape(batch, block, hidden)[:, 1:].float()
    rms = stash_rms.reshape(batch, block)[:, 1:].float()
    q = stash_packed.reshape(batch, block, -1)[:, 1:, :rank].float()

    logits = torch.matmul(rows.reshape(batch * steps, hidden), lm_head_weight.t())
    # The reference sorts the BF16 logits in float32; do the same so the tie-break
    # and the selected candidates match the served SGLang path.
    cand = torch.topk(logits.float(), int(topk), dim=-1).indices
    cand = torch.sort(cand, dim=-1).values.view(batch, steps, int(topk))

    head_rows = lm_head_weight[cand].float()  # [bs, T-1, K, H]
    raw_scaled = raw * norm_w.float()
    base = torch.sum(raw_scaled.unsqueeze(2) * head_rows, dim=-1)  # [bs, T-1, K]

    anchor_lanes = anchors.view(batch, 1, 1).expand(batch, 1, int(topk))
    sid = torch.cat([anchor_lanes, cand[:, :-1]], dim=1) if steps > 1 else anchor_lanes
    send = tsv[sid].float()
    pred_gate, pred_value = send[..., :rank], send[..., rank:]
    message = torch.sigmoid(q.unsqueeze(2) + pred_gate + gate_bias.float().view(1, 1, 1, rank)) * pred_value

    tg_cand = tg[cand].float()
    lane_scores = torch.matmul(message, tg_cand.transpose(-1, -2))  # [bs, T-1, K(lane), K(candidate)]
    scores = base.unsqueeze(2) + rms.view(batch, steps, 1, 1) * lane_scores
    lane_best = torch.argmax(scores, dim=-1)  # [bs, T-1, K]

    lane = lane_best[:, 0, 0]
    tokens: list[torch.Tensor] = []
    for i in range(steps):
        tokens.append(torch.gather(cand[:, i, :], 1, lane.unsqueeze(1)).squeeze(1))
        if i + 1 < steps:
            lane = torch.gather(lane_best[:, i + 1, :], 1, lane.unsqueeze(1)).squeeze(1)
    out = torch.stack(tokens, dim=1)
    if return_candidates:
        return out, cand
    return out
