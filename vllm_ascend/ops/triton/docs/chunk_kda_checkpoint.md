# Chunk KDA Prefill Checkpoint (`chunk_kda` / `chunk_gated_delta_rule_fwd_h_kda`)

## Description

- **Function**: Compute chunkwise KDA prefill output and final recurrent state on
  Ascend NPUs while optionally snapshotting an intermediate recurrent state
  checkpoint at a per-sequence chunk boundary (`STORE_CHECKPOINT=True`).
- **Formula**: For each sequence `i` and value/key tile of head `h`, initialize
  float32 accumulator `H = h0[i, h]` (or zeros) and iterate over 64-token chunks
  `t = 0 .. NT - 1`:

  ```text
  if STORE_CHECKPOINT and checkpoint_offset[i] > 0 and t * BT == checkpoint_offset[i]:
      ckpt = H.to(checkpoint_state.dtype)
      checkpoint_state[i, h] = ckpt
      H = ckpt.to(float32)  # match restored cache dtype between cold and warm resume
  v_new = v_t - w_t @ H.T
  H = H * exp2(gk_last) + k_t.T @ v_new_decayed
  ```

  Cold execution rounds `H` through `checkpoint_state.dtype` at the exported
  boundary so resumed prefix-cache runs match cold prefill bit-for-bit from that
  boundary onward.
- **Algorithm flow** (parallel across `(cdiv(V, BV), N * H)` programs):
  1. `chunk_kda` / `chunk_kda_with_fused_gate` expands optional gate fusion,
     normalizes queries/keys in kernel when requested, computes intra-chunk decay
     and WY/recomputation intermediates (`A`, `Aqk`, `w`, `u`, `kg`), and calls
     `chunk_gated_delta_rule_fwd_h_kda`.
  2. `chunk_gated_delta_rule_fwd_h_kda` validates NPU device placement, integer
     index dtypes (`int32`/`int64`), contiguous index/checkpoint buffers, and
     accumulates chunk recurrent states while storing boundary snapshots when
     `checkpoint_offset > 0` matches `t * BT`.
  3. `chunk_gla_fwd_o_gk` combines `q`, `v_new`, `g`, `Aqk`, and `h` into the
     output tensor `o` (updating `v` buffer in place on the chunk KDA path).
- **Supported modes**: Atlas A2 and Atlas A3 inference, eager mode and piecewise
  NPU graphs when the model attention `_forward` keeps `@eager_break_during_capture`
  outermost. 950PR&950DT: N/A (not validated).

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q`, `k` | Input | Queries and keys `[B, T, Hg, K]` (`B=1` for varlen); normalized in kernel when `use_qk_l2norm_in_kernel=True` | FP16 / BF16 / FP32 | ND |
| `v` | Input / In-place Output | Values `[B, T, H, V]`; mutated in place as output buffer `o` by `chunk_kda` | FP16 / BF16 / FP32 | ND |
| `g` / `raw_g` | Input | Log decay gate (`g`) or raw gate logits (`raw_g`) of shape `[B, T, H, K]` | FP16 / BF16 / FP32 | ND |
| `beta` / `raw_beta` | Input | Update gate `[B, T, H]` (or `[B, T, H]` logits when `raw_beta` is passed) | FP16 / BF16 / FP32 | ND |
| `A_log`, `g_bias` | Input | Per-head log decay scale `[H]` (or `[1, 1, H, 1]`) and optional gate bias | FP32 / activation dtype | ND |
| `scale` | Attribute | Query scaling factor (`K ** -0.5` when `None`) | Python float / `None` | Scalar / N/A |
| `initial_state` | Input | Initial recurrent state `[N, H, V, K]` or `None` | FP16 / BF16 / FP32 | Contiguous ND |
| `output_final_state` | Attribute | Whether to materialize final FP32 recurrent state `[N, H, V, K]` | Python bool | Scalar |
| `use_qk_l2norm_in_kernel` | Attribute | Apply fused RMS/L2 normalization to `q` and `k` | Python bool | Scalar |
| `cu_seqlens` | Input | Cumulative sequence token offsets `[N + 1]` on NPU, or `None` for fixed-length batches | INT32 / INT64 | Contiguous 1D ND |
| `prebuilt_meta` | Attribute | Optional precomputed device chunk metadata (`chunk_indices_chunk64`, `chunk_offsets_chunk64`) or `None` | Object / `None` | N/A |
| `safe_gate`, `lower_bound` | Attribute | Gate activation selector and bounded-sigmoid lower bound (`None` keeps softplus when `safe_gate` is not set) | Python bool / float / `None` | Scalar / N/A |
| `checkpoint_state` | Output | Pre-allocated recurrent checkpoint buffer `[N, H, V, K]` on NPU, paired with `checkpoint_offsets`, or `None` | FP16 / BF16 / FP32 | Contiguous 4D ND |
| `checkpoint_offsets` | Input | Per-sequence token offset `[N]` at which to snapshot the chunk-start state (`0` disables snapshot for that row), or `None` | INT32 / INT64 | Contiguous 1D ND |
| `o` | Output | Attention output `[B, T, H, V]` | Same as `v` | ND |
| `final_state` | Output | Final recurrent state `[N, H, V, K]` or `None` | FP32 / `None` | Contiguous ND |

## Constraints

- `k`, `w`, `u`, `cu_seqlens`, `checkpoint_state`, and `checkpoint_offsets` must
  reside on the same NPU device (`device.type == "npu"`).
- `cu_seqlens` and `checkpoint_offsets` require `torch.int32` or `torch.int64`
  and are normalized with `.contiguous()` before kernel launch.
- `checkpoint_state` and `checkpoint_offsets` must be provided together (`both`
  or `neither`). When provided, `checkpoint_state` must be contiguous with shape
  `(N, H, V, K)` and dtype in `(torch.float16, torch.bfloat16, torch.float32)`,
  and `checkpoint_offsets` must have shape `(N,)`.
- Head dimension `K <= 256` (`BV = 64`), chunk size `BT = 64`. Positive
  `checkpoint_offsets[i]` must align to a chunk boundary (`64 * k`) before
  sequence end to be exported during recurrence.
- Enabled in model serving only when `additional_config.kda_internal_prefill_checkpoints`
  is `True` (opt-in default `False`). Graph capture requires `@eager_break_during_capture`
  outermost around stateful metadata extraction/export.

## Origin and Differences

- **Origin**: Flash-linear-attention (FLA) KDA chunkwise prefill kernels adapted
  for vLLM (`vllm/models/kimi_k3/nvidia/ops/third_party/kda.py`) and integrated
  with upstream internal prefill checkpoint support in `MambaSpec`.
- **Differences**:
  - Added `STORE_CHECKPOINT`, `checkpoint_state`, `checkpoint_offsets`, and
    `int64` offset arithmetic in `chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda`,
    rounding exported checkpoints through the cache buffer dtype at export time.
  - Added strict NPU device, dtype (`int32`/`int64` indices, `fp16`/`bf16`/`fp32`
    checkpoints), shape, and contiguity checks on kernel entry.
  - Extended `chunk_kda_with_fused_gate` to support optional `A_log`,
    `lower_bound=None`, `safe_gate`, both `beta` and `raw_beta`, and checkpoint
    arguments without changing non-checkpoint callers.

## Test Cases

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_kda_npu.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/ops/test_kda_checkpoint.py
```
