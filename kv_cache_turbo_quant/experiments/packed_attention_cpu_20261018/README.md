# Store4 packed attention CPU candidate

| Item | Status |
|---|---|
| Scope | One-query GQA decode mathematical prototype; no serving dispatch changes |
| Base | 87b101ba2690c55e4314798092d93af095d8c38f |
| Task branch | perf/tq-packed-attention-cpu-20261018 |
| CPU result | 10 new shape/boundary cases and 13 invalid-contract cases passed; final BF16 output compared with zero tolerance |
| Device result | No NPU kernel implemented, compiled, or tested; no device performance claim |
| Delivery goal | Existing full-model speed failure and quality/stability/reproduction gaps remain open |

The old 16:14 read-only notice governed that evidence-recheck round. The current
user instruction restores local development and CPU verification. It does not
authorize an NPU job, container, service, lease, PR, or associated-branch push.

## Frozen layout and interface

The prototype reuses `code/integration/kvtq_store.py`; it does not implement a new
codec, centroid table, quantizer, or rotation generator. Source provenance is the
base above and that file's existing SHA256
`9e80293023796907c35d369c1b7a578d33a781c98c9dd1eec1307fd3e02bf031`.

| Field | Contract |
|---|---|
| Query | CPU BF16 `[batch, query_heads, 128]`, one query per sequence |
| K and V | Separate contiguous uint8 `[physical_blocks, page_size, kv_heads, 66]` |
| Bytes 0..63 | Two nonuniform-centroid indices per byte: dimension 2j in the low nibble and 2j+1 in the high nibble |
| Bytes 64..65 | Stored BF16 norm, little endian; no QJL residual or gamma in this layout |
| Codec rounding | Existing BF16 centroid times stored BF16 norm, with BF16 output; no uniform INT4 reinterpretation |
| Block table | int32/int64 `[batch, max_pages]`; active physical pages must be valid; unused entries are ignored |
| Lengths | One positive integer per nonempty sequence, bounded by the table; empty batch supported |
| GQA | Query head count divisible by KV head count; primary target Hq16/Hkv2/D128 |
| Rotations | Existing FP32 Q times R-transpose, rounded to BF16 before attention; attention output rounded to BF16 before FP32 inverse rotation and final BF16 write |
| Output | BF16 query shape; optional existing CPU output view is written in place |

`packed_decode_cpu` gathers only the active logical tile and delegates its
dequantization to the unchanged store reference through a temporary tile view.
It updates per-head maximum, denominator, and numerator with online softmax.
It does not materialize full-length dense K/V within the candidate. This CPU
allocation structure is not an NPU UB/L1, DMA, synchronization, or tiling proof.

The independent attention oracle materializes K/V with the same existing codec
and uses PyTorch dense SDPA. Both attention calculations intentionally use FP64
to check the recurrence and layout without claiming FIA-equivalent device
reductions. Their final BF16 outputs must be bitwise equal. Passing this new
CPU gate does not relax or satisfy the existing NPU zero-tolerance gate, packed
kernel precision, model quality, stability, or full-model speed requirements.

## New cases and reproducible CPU entry

The new cases cover B16/L20, B60/L2000, B1/L8192, lengths
1/63/64/65/127/128/129/1999/2000/2001, tile sizes 31/64/127/128,
permuted physical pages, unused negative table entries, zero and finite large
norms, alternate GQA ratios, page size64, noncontiguous query/table/output, empty
batch, and unchanged inputs. Invalid lengths, active pages, dtype, GQA, tile
size, and scale must be rejected. These are new candidate checks, not reruns of
the historical model matrix, token comparison, or NPU verification scripts.

Actual local environment: macOS, Python3.9.6, torch2.8.0, numpy2.0.2; two CPU
threads. No container or base image was used for this CPU phase, and no packages
were installed. A future device phase must separately fix its image and native
dependencies; the old device image is not represented as this run's environment.

From the task worktree, with a fresh private report path:

```bash
/usr/bin/python3 kv_cache_turbo_quant/experiments/packed_attention_cpu_20261018/verify_cpu.py \
  --report /absolute/private/output/cpu_math_unique_run.json
```

The verifier records source hashes, cases, failures and the CPU-only scope.
Reports are created exclusively. Any failed input is preserved in a separate
fixture rather than overwritten. To run only a new or previously failed case:

```bash
/usr/bin/python3 kv_cache_turbo_quant/experiments/packed_attention_cpu_20261018/verify_cpu.py \
  --case ragged_tile_page_tails --report /absolute/private/output/new_case_unique.json
```

Full successful new checks are not rerun without a changed implementation or
new unresolved concern. Results and execution commands remain in the private
task evidence directory, not embedded as historical NPU or full-model evidence.

## Remaining device work and closure

The next implementation needs a real device kernel that consumes the same 66B
pages, preserves the codec rounding, and combines QK, softmax and PV without a
full-length BF16 K/V buffer. NPU FP32 reduction behavior, boundaries, nondefault
streams, memory safety and the existing precision gate remain unverified.
CPU FP64 arithmetic cannot establish any of those device properties.

Only after a concrete kernel/build plan is reviewable should a bounded device
window be requested from the designated resource coordinator. No device window
or lease is scheduled here. Real-model quality and stability and the original
short no-regression / long positive-gain speed targets still require their own
evidence. No old failure or precision threshold is removed or weakened.
