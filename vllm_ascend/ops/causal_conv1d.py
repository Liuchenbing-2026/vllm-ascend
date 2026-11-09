import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    PyTorch reference implementation of causal_conv1d.

    Args:
        x: (batch, dim, seqlen)
        weight: (dim, width)
        bias: (dim,)
        initial_states: (batch, dim, width - 1)
        final_states_out: (batch, dim, width - 1)
        return_final_states: bool
        activation: str

    Returns:
        out: (batch, dim, seqlen)
        final_states_out: (batch, dim, width - 1) if return_final_states
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    PyTorch implementation of causal_conv1d_fn.

    Args:
        x: (dim, cu_seq_len) for varlen
        weight: (dim, width)
        bias: (dim,)
        activation: str
        conv_states: (..., dim, width - 1)
        has_initial_state: (batch) bool
        cache_indices: (batch) int32
        query_start_loc: (batch + 1) int32
        pad_slot_id: int

    Returns:
        out: (batch, dim, seqlen)
    """

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    if query_start_loc is None:
        raise RuntimeError("causal_conv1d_fn requires query_start_loc for varlen inputs.")
    if cache_indices is None:
        raise RuntimeError("causal_conv1d_fn requires cache_indices.")
    if has_initial_state is None:
        raise RuntimeError("causal_conv1d_fn requires has_initial_state.")
    if conv_states is None:
        raise RuntimeError("causal_conv1d_fn requires conv_states.")

    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    # Normalize x to [dim, total_tokens]
    if x.dim() == 3:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.shape[1] == 1:
            x = x.squeeze(1).transpose(0, 1)
        else:
            raise RuntimeError(f"Unsupported x shape for causal_conv1d_fn: {tuple(x.shape)}")
    if x.dim() != 2:
        raise RuntimeError(f"Unsupported x ndim for causal_conv1d_fn: {x.dim()}")

    feature_dim = x.shape[0]
    if weight.shape[0] != feature_dim and weight.shape[1] == feature_dim:
        weight = weight.transpose(0, 1)
    weight = weight.contiguous()
    dim, width = weight.shape
    if dim != feature_dim:
        raise RuntimeError(
            f"causal_conv1d_fn: weight dim mismatch, x dim={feature_dim}, weight.shape={tuple(weight.shape)}"
        )

    state_len = width - 1
    if conv_states.shape[-2] != dim and conv_states.shape[-1] == dim:
        conv_states = conv_states.transpose(-1, -2)
    if conv_states.shape[-2] != dim:
        raise RuntimeError(
            f"causal_conv1d_fn: conv_states dim mismatch, "
            f"expected dim={dim}, conv_states.shape={tuple(conv_states.shape)}"
        )
    if conv_states.shape[-1] < state_len:
        raise RuntimeError(f"causal_conv1d_fn: conv_states too short, need >= {state_len}, got {conv_states.shape[-1]}")

    seqlens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
    splits = torch.split(x, seqlens, dim=-1)

    out_chunks = []
    for i, x_s in enumerate(splits):
        cache_idx = int(cache_indices[i].item())
        if cache_idx == pad_slot_id:
            continue

        state = conv_states[cache_idx]
        init_state = state[..., :state_len].unsqueeze(0) if bool(has_initial_state[i].item()) else None
        out_ref, final_state = causal_conv1d_ref(
            x_s.unsqueeze(0),
            weight,
            bias,
            activation=activation,
            return_final_states=True,
            initial_states=init_state,
        )
        state[..., :state_len].copy_(final_state.squeeze(0))
        out_chunks.append(out_ref.squeeze(0))

    if not out_chunks:
        return x.new_zeros((dim, 0))
    return torch.cat(out_chunks, dim=-1)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """Batched causal depthwise conv update (Ascend).

    Replaces the previous per-request Python loop (which performed up to four
    device-to-host ``.item()`` synchronisations per sequence per layer) with a
    fully vectorised implementation:

    1. gather the owning conv state for every sequence in one ``index_select``;
    2. rebuild the ``[B, width - 1 + L, dim]`` sliding window with a single
       gather (no per-sequence slicing);
    3. run one grouped ``conv1d`` for the whole batch;
    4. roll the recurrent state forward with a single ``gather`` + one
       ``index_copy_``, using ``num_accepted_tokens`` as a row offset.

    Args:
        x: ``[num_tokens, dim]`` for the varlen path (``query_start_loc`` given),
            ``[B, dim]`` or ``[B, L, dim]`` otherwise.
        conv_state: ``[num_slots, dim, state_len]`` (``state_len`` may be padded).
        weight: ``[dim, width]`` (or ``[width, dim]``).
        conv_state_indices: ``[B]`` slot ids for the state cache.
        num_accepted_tokens: ``[B]`` number of tokens to commit per sequence.
        query_start_loc: ``[B + 1]`` varlen offsets (exclusive prefix sum).
        max_query_len: upper bound of the per-sequence token count. When not
            given it is derived from the input shape / offsets.
        pad_slot_id: slot id meaning "no state"; such rows are read-only.

    Returns:
        Tensor with the same shape/dtype as ``x``.
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x_work = x.to(conv_state.dtype)

    if weight.shape[0] != x_work.shape[-1] and weight.shape[1] == x_work.shape[-1]:
        weight = weight.transpose(0, 1)
    weight = weight.contiguous().to(conv_state.dtype)
    dim, width = weight.shape
    state_len = width - 1

    if conv_state.shape[-2] != dim and conv_state.shape[-1] == dim:
        # Accept both (..., dim, state_len) and (..., state_len, dim) inputs.
        conv_state = conv_state.transpose(-1, -2)
    if conv_state.shape[-2] != dim:
        raise RuntimeError(
            "causal_conv1d_update: conv_state dim mismatch, "
            f"expected dim={dim}, conv_state.shape={tuple(conv_state.shape)}"
        )
    if conv_state.shape[-1] < state_len:
        raise RuntimeError(
            "causal_conv1d_update: conv_state too short, "
            f"need >= {state_len}, got {conv_state.shape[-1]}"
        )

    device = x_work.device

    # ---- 1. normalise x to [B, Lp, dim] plus a flat index map -----------------
    flat_index = None
    valid_mask = None
    if query_start_loc is not None:
        assert conv_state_indices is not None
        batch = conv_state_indices.numel()
        starts = query_start_loc[:batch].to(device=device, dtype=torch.int64)
        ends = query_start_loc[1 : batch + 1].to(device=device, dtype=torch.int64)
        lens = ends - starts
        if max_query_len is None:
            max_query_len = int(lens.max().item())
        lp = int(max_query_len)
        offs = torch.arange(lp, device=device, dtype=torch.long).unsqueeze(0)
        flat_index = starts.unsqueeze(1) + offs
        valid_mask = flat_index < ends.unsqueeze(1)
        gather_index = torch.where(valid_mask, flat_index, torch.zeros_like(flat_index))
        x_win = x_work.index_select(0, gather_index.reshape(-1)).reshape(batch, lp, dim)
    elif x_work.dim() == 3:
        batch, lp, _ = x_work.shape
        x_win = x_work
    else:
        batch = x_work.shape[0]
        lp = 1
        x_win = x_work.unsqueeze(1)

    # ---- 2. gather the per-sequence conv state -------------------------------
    if conv_state_indices is not None:
        slot_ids = conv_state_indices[:batch].to(device=device, dtype=torch.long)
        is_pad = slot_ids == pad_slot_id
        safe_slots = torch.where(is_pad, torch.zeros_like(slot_ids), slot_ids)
    else:
        slot_ids = None
        is_pad = None
        safe_slots = None
    init_state = (
        conv_state.index_select(0, safe_slots)
        if safe_slots is not None
        else conv_state[:batch]
    )  # [B, dim, state_len]
    init_state = init_state[..., :state_len].transpose(1, 2).contiguous()  # [B, state_len, dim]

    # ---- 3. one grouped conv1d over the whole batch --------------------------
    window = torch.cat([init_state, x_win], dim=1)  # [B, state_len + Lp, dim]
    conv_out = F.conv1d(
        window.transpose(1, 2),
        weight.unsqueeze(1),
        bias.to(conv_state.dtype) if bias is not None else None,
        groups=dim,
    ).transpose(1, 2)  # [B, Lp, dim]
    if activation is not None:
        conv_out = F.silu(conv_out)

    # ---- 4. roll the state forward by num_accepted_tokens --------------------
    if num_accepted_tokens is not None:
        committed = (
            num_accepted_tokens[:batch]
            .to(device=device, dtype=torch.long)
            .clamp_(0, lp)
        )
    else:
        committed = torch.full((batch,), lp, device=device, dtype=torch.long)
    # The recurrent state is the last ``state_len`` tokens consumed, i.e.
    # ``window[committed : committed + state_len]`` once ``committed`` tokens of
    # this step have been accepted.
    base = committed.unsqueeze(1)
    ar = torch.arange(state_len, device=device, dtype=torch.long).unsqueeze(0)
    state_index = (
        (base + ar).unsqueeze(-1).expand(batch, state_len, dim)
    )  # [B, state_len, dim]
    new_state = window.gather(1, state_index)  # [B, state_len, dim]
    # Padded state caches expose more than ``state_len`` columns; only the
    # leading ``state_len`` columns are meaningful (same as the reference
    # implementation, which writes ``state[..., :state_len]``).
    state_view = conv_state[..., :state_len]
    if slot_ids is not None:
        # Pad rows must not touch their slot. Sending every pad row to a real
        # row's slot *with that row's value* keeps the scatter duplicate-free
        # in value, so the (unspecified) write order cannot corrupt a slot.
        ref = torch.argmax((~is_pad).to(torch.int64))
        ref_slot = safe_slots[ref]
        ref_value = torch.where(is_pad[ref], init_state[ref], new_state[ref])
        write_slots = torch.where(is_pad, ref_slot, safe_slots)
        write_state = torch.where(
            is_pad.reshape(batch, 1, 1), ref_value.expand_as(new_state), new_state
        )
        state_view.index_copy_(
            0, write_slots, write_state.transpose(1, 2).to(conv_state.dtype)
        )
    else:
        state_view[:batch] = new_state.transpose(1, 2).to(conv_state.dtype)

    # ---- 5. scatter the outputs back into the caller's layout ----------------
    # Only accepted tokens carry a real conv result; everything else keeps the
    # caller's own values, exactly like the per-request loop that skipped them.
    keep = torch.arange(lp, device=device).unsqueeze(0) < committed.unsqueeze(1)
    if is_pad is not None:
        keep = keep & ~is_pad.unsqueeze(1)
    if flat_index is not None:
        keep = keep & valid_mask
        out = x_work.clone()
        out[flat_index[keep]] = conv_out.reshape(-1, dim)[keep.reshape(-1)]
        return out.to(original_x_dtype)
    conv_out = torch.where(keep.unsqueeze(-1), conv_out, x_win)
    if x.dim() == 3:
        return conv_out.to(original_x_dtype)
    return conv_out.reshape(batch, dim).to(original_x_dtype)
