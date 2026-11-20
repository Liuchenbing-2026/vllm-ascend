# dspark_swa_indices

## Description

- **Function**: Unified Triton-Ascend implementation of DSpark (DeepSeek V4 / V4.1) non-causal SWA index and length construction (`build_dspark_swa_indices_triton` and shared entry point `vllm_ascend.attention.dsa_v1.build_dspark_swa_indices`). Through `output_mode`, it supports both **physical KV slots** (`"physical"`, used by `AscendDSAMetadataBuilder`) and **request-local logical sequence positions** (`"logical"`, used by `AscendDSAV41MetadataBuilder` / `SparseFlashMLA`).
- **Formula** (per request `r`, all integer arithmetic):
    - `q_len = query_start_loc[r + 1] - query_start_loc[r]`
    - `prefix_len = seq_lens[r] - q_len`
    - `start_pos = max(prefix_len - window_size, 0)`
    - `visible_len = seq_lens[r] - start_pos`
    - for column `c` in `[0, index_width)`: `pos = start_pos + c`:
        - in `"logical"` mode: `slot[c] = pos` if `c < visible_len`, else `-1`
        - in `"physical"` mode: `block_num = pos // block_size`, `block_off = pos - block_num * block_size`, `slot[c] = block_table[r, block_num] * block_size + block_off` if `c < visible_len`, else `-1`
    - every active output row `[q0, q0 + q_len)` receives the same `slot` vector and `lens[row] = visible_len`; padded rows `[active_rows, num_rows_padded)` are reset to `slots = -1`, `lens = 0` in the same launch
- **Algorithm flow** (processed per request-column-block item, independently):
    1. Compute grid: `grid_size = min(get_vectorcore_num(), num_slots * NUM_CB)` where `NUM_CB = ceil(index_width / BLOCK_W)` and `num_slots = max(R_alloc, ceil(num_rows_padded / num_query_per_req))`; each program claims work items `w = pid, pid + grid_size, ...` grid-stride (`r = w // NUM_CB`, `cb = w % NUM_CB`).
    2. Load `active_rows = query_start_loc[num_reqs]` on device. Distributed pad-row cleanup: each item resets the slice of `[active_rows, num_rows_padded)` inside its row band `[r * num_query_per_req, (r + 1) * num_query_per_req)`, so a captured ACL graph never replays stale rows.
    3. For active items (`r < num_reqs` and `q_len > 0`):
        - `"logical"` mode sets `slot = pos`.
        - `"physical"` mode computes `blk_base = clamp((start_pos + cb * BLOCK_W) // BLOCK_SIZE, 0, num_blocks - 1)` and loads only the bounded block-table slice `[blk_base, blk_base + BT_WIN_POW2)` into UB (`BT_WIN_POW2 = max(2, next_pow2((BLOCK_W + BLOCK_SIZE - 2) // BLOCK_SIZE + 1))`), then gathers physical block IDs via `tl.gather`. This eliminates full-row UB preloading and supports arbitrary `block_table` widths and any positive `block_size` (including non-power-of-two block sizes).
    4. Loop over `q_off = 0, Q_POW2, ...` (`Q_POW2 = min(16, next_pow2(num_speculative_tokens + 1))`) using 2D stores to write `[q_len, index_width]` output rows and (for `cb == 0`) store `visible_len` into `lens`.
- **Supported modes**: Atlas A2 (910B), A3 (910C), and A5 NPU paths for both `"physical"` and `"logical"` modes under eager execution and ACL graph capture/replay. Triton is required on NPU; non-NPU (CPU) unit-test harnesses use the CPU reference helper in `vllm_ascend.attention.dsa_v1`.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `block_table` | Input | SWA KV cache block table `[num_reqs, num_blocks]` (required in `"physical"` mode; ignored/`None` in `"logical"` mode); row stride passed separately via `stride(0)` | int32 / int64 | ND |
| `query_start_loc` | Input | Cumulative query offsets `[num_reqs + 1]` | int32 / int64 | 1D |
| `seq_lens` | Input | Per-request context lengths `[num_reqs]` | int32 / int64 | 1D |
| `num_speculative_tokens` | Input (attribute) | Number of draft tokens per request; bounds default `Q_POW2` and index alignment | int32 | scalar |
| `window_size` | Input (attribute) | Sliding-window size of the SWA cache (model constant) | int32 | scalar |
| `block_size` | Input (attribute) | DSA KV-cache block size in `"physical"` mode (any positive integer; normalized to `1` in `"logical"` mode) | int32 | scalar |
| `index_width` | Input (attribute) | Aligned index width (`ceil((window_size + num_speculative_tokens) / 128) * 128` when omitted) | int32 | scalar |
| `num_decode_tokens` | Input (attribute) | Fixed output row count supplied by the caller to avoid D2H sync; graph replay may have fewer active rows (`query_start_loc[num_reqs]` on device) | int32 | scalar |
| `max_num_reqs` | Input (attribute) | Capture-time request-slot capacity sizing the fixed grid and pad cleanup; optional | int32 | scalar |
| `num_query_per_req` | Input (attribute) | Row-expansion factor per request slot; defaults to `num_speculative_tokens + 1` | int32 | scalar |
| `output_mode` | Input (attribute) | `"physical"` (KV slot IDs) or `"logical"` (request-local positions) | str | scalar |
| `indices_output` | Output | Contiguous output buffer `[>=num_rows, 1, index_width]`, `-1` for invisible/padded lanes; persistent buffer or freshly allocated when omitted | int32 | ND |
| `lens_output` | Output | Contiguous output buffer `[>=num_rows]` or `[>=num_rows, 1]`, `0` on padded rows; default allocation is `[alloc_rows]` int64 in `"physical"` mode and `[alloc_rows, 1]` int32 in `"logical"` mode | int32 / int64 | 1D / 2D |

## Constraints

- All tensor inputs and caller-supplied outputs used in a Triton launch must reside on the same accelerator device (`npu`/`cuda`); host/CPU or cross-device tensors raise `ValueError`.
- In `"physical"` mode, `block_size` must be `> 0` (power-of-two is not required) and `block_table` must have unit column stride (`stride(1) == 1`) with `shape[1] >= 1`. There is no 8192-column limit on `block_table_width`.
- `indices_output` must be contiguous `int32` of shape `[>=num_rows, 1, index_width]`. When `max_num_reqs` is passed for graph capture, supply the FULL persistent buffer (`indices_output=buffer`, not `buffer[:num_rows]`) so trailing padded rows `[active_rows, num_rows_padded)` are cleared in the same launch.
- `lens_output` must be contiguous `int32` or `int64` with shape `[>=num_rows]` or `[>=num_rows, 1]`.
- Padded rows `[active_rows, num_rows_padded)` are reset to `slots = -1`, `lens = 0` in the same launch using device-side `active_rows = query_start_loc[num_reqs]`.
- All runtime scalars (`num_blocks`, `num_reqs`, `num_rows_padded`, `num_rows`, `num_query_per_req`, `num_slots`) are `do_not_specialize`; only the compile key (`WINDOW_SIZE`, `BLOCK_SIZE`, `INDEX_W`, `BLOCK_W`, `BT_WIN_POW2`, `Q_POW2`, `bt_stride`, `USE_LOGICAL_INDICES`, `lens_dtype`) triggers JIT compilation and is warmed up by `model_executor/warmup/kernel_warmup.py` for both DSA (`"physical"`) and DSA V4.1 (`"logical"`).
- `get_vectorcore_num()` is initialized idempotently via `init_device_properties_triton()` before launch.

## Origin and Differences

- **Origin**: Consolidated from the previous physical-slot kernel, the retired separate logical-position Triton kernel, and the production PyTorch `repeat_interleave` path in `vllm_ascend/attention/dsa_v1.py`.
- **Differences**:
    - Unifies physical and logical output modes into one Triton kernel and shared entry point (`output_mode`), preserving DSA (`[T]` int64 default lengths) and DSA V4.1 (`[T, 1]` int32 default lengths) contracts.
    - Replaces full-row block-table UB preloading with bounded per-tile window loading (`BT_WIN_POW2`), removing the `next_pow2(block_table_width) <= 8192` cap and power-of-two `block_size` restriction.
    - Supports arbitrary `q_len` via bounded `Q_POW2` tiling loops and retires the duplicate logical Triton file and production NPU PyTorch fallback.

## Test Cases

Accuracy tests cover both `"physical"` and `"logical"` modes against the eager PyTorch reference with bit-exact comparison (`rtol=0, atol=0`), non-power-of-two block sizes (`12, 48, 96, 100`), wide block tables (`10000`, `12000`, `16384` columns > 8192) with small SWA windows and block boundaries, `num_reqs == 0` / `num_reqs > max_num_reqs` / `q_len > Q_POW2`, persistent-buffer direct writes via `build_dspark_swa_indices`, and shrinking reused buffers (`90 -> 30 -> 0` active rows) under direct launch and ACL graph replay.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dspark_swa_indices.py
```
