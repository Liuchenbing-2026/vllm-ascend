# Reproducing the arch22 chunk_gated_delta_rule results

Everything below was measured on 910B4-1 with the kernel source in this commit.
The harness in this directory is the harness that produced the numbers; nothing
was re-typed for the writeup.

Raw logs are in `logs/`. What they cover, and what they do not:

| Numbers | Log |
|---|---|
| section 1c: packing gate, host-side padding | `logs/a2_packing_and_padding.log` |
| section 1c: real captured vectors | `logs/a2_realvectors.log` |
| section 1 end-to-end, section 8b, the `actual_seq_lengths` probe | `logs/m18_all.log` |
| section 1b second machine, and its `PASS all bit-exact` | `logs/a2_secondmachine.log` |
| **section 1 operator table (the seven-row one)** | **not in this bundle** -- measured 2026-09-04 on A2 |
| **the W8A8 production row, and the 71.06 s in 8b** | **not in this bundle** -- measured 2026-09-05 on m18 |
| **the `COMPARED 36 tensors / PASS` in section 1** | **not in this bundle**; the same check on the second machine is in `logs/a2_secondmachine.log` |
| the anecdotes with numbers in section 9 | not in this bundle |

The unsourced rows are internally consistent -- every delta recomputes from the
cells printed next to it -- which is exactly what internal consistency is worth.
Where a logged measurement overlaps one of them it agrees to within 0.4pp: the
gate log's `even` mode gives -53.9% and -84.3% against the table's -54.3% and
-85.2%.

## 1. What is being claimed

Bit-exactness, first: 36/36 output tensors compare equal under `torch.equal`
across the 18 shapes in `ab_gdr.py`. Both operator outputs are checked, `o` and
the final state `fs`. If this does not pass, nothing else matters.

Operator level. **Read section 1c before quoting the multi-request rows**: they
were measured on 64-aligned synthetic batches, which is a regime a live vLLM
server never produces. The number for real traffic is -14.0% to -15.4%, and it is
in section 1c. The four B=1 rows are not affected -- the gate needs `b > 1`.

One process per shape, one card, one session, round-level ABBA, median of 30, at
Qwen3.6-35B-A3B GDN shapes (Dk=Dv=128, global Nk=16 / Nv=32), with
`asl = [T//B]*B`:

| Config | Shape | Base (us) | Patch (us) | Delta | Within-arm spread |
|---|---|---|---|---|---|
| TP8 (Nk=2, Nv=4) | T=8192 B=1 | 1457.5 | 1225.6 | -15.9% | 0.5% / 0.6% |
| TP8 | T=8192 B=16 | 2162.4 | 988.6 | -54.3% | 1.6% / 1.1% |
| TP8 | T=2560 B=40 | 2511.4 | 371.6 | -85.2% | 0.5% / 2.4% |
| TP4 (Nk=4, Nv=8) | T=8192 B=1 | 1922.9 | 1661.8 | -13.6% | 0.3% / 2.0% |
| TP4 | T=8192 B=16 | 2743.6 | 1494.6 | -45.5% | 2.1% / 0.1% |
| TP2 (Nk=8, Nv=16) | T=8192 B=1 | 2618.3 | 2362.6 | -9.8% | 0.5% / 0.5% |
| TP1 (Nk=16, Nv=32) | T=8192 B=1 | 4740.9 | 4332.5 | -8.6% | 0.4% / 0.5% |

The single-batch rows carry roughly +-1pp. The gain grows with how many
sequences prefill in the same step, which is what changes 4 and 5 in the commit
message address -- **but only when those sequences are 64-aligned**; see 1c.

End to end, `vllm bench serve`, TP4, four-arm ABBA. Prefill-dominated shape
(random 1024-token prompts, `--random-range-ratio 0`, one output token,
concurrency 32, 256 requests), bf16. **This shape is a diagnostic, chosen to
maximise the operator's share of the step; it is not a deployment workload.** For
the deployment workloads see the production table below.

| Metric | Base | Patch | Delta |
|---|---|---|---|
| Median TTFT | 1640.15 ms | 1604.84 ms | -2.15% |
| Mean TTFT | 1654.85 ms | 1607.66 ms | -2.85% |
| P99 TTFT | 2148.29 ms | 2126.19 ms | -1.03% |
| Request throughput | 18.36 req/s | 18.83 req/s | +2.56% |
| Benchmark duration | 13.950 s | 13.595 s | -2.54% |

Four non-overlapping cells on duration, median TTFT, mean TTFT and P99.
Within-arm spread is not uniform: 0.02-0.08% on P99, 0.55-0.95% on median TTFT,
0.96-1.72% on duration and throughput, and 1.03-2.56% on mean TTFT. The mean-TTFT
base arm (1676.02 / 1633.67 ms, 2.56%) is as wide as that metric's own -2.85%
delta, so mean TTFT rests on the non-overlap alone; prefer median TTFT and
duration. The same four arms at `--random-range-ratio 0.5` give -2.21% duration
but **+0.38%** median TTFT with interleaved cells (base 1535.26 / 1613.65, patch
1582.41 / 1578.56), so the range ratio is not a free parameter here.

> An earlier revision of this file reported +5.6% throughput and -3.7% median
> TTFT for this shape. That run used `--num-prompts 128`, a 6.24-second
> benchmark, where start-up and tail are a large fraction of the total. Re-run
> at `--num-prompts 256` (13.9 s) the same configuration gives the +2.56% above.
> Take +2.56%. (The re-run also carried the deployment's `sysctl` settings and
> jemalloc preload, so the environments are not identical; the sample size is
> the likelier explanation but it is not isolated.)

Production-shaped runs (4096- and 6144-token prompts, 256 output tokens, 160
requests, concurrency 32, `--max-num-batched-tokens 8192`) gain about 1%. Four
independent four-arm ABBA rounds:

| Run | Throughput | Duration | Median TTFT |
|---|---|---|---|
| w8a8, prefix+input shapes | +0.67% / +1.08% | -0.55% / -0.98% | -1.73% (load 1) |
| bf16, prefix+input shapes | +1.26% / +0.91% | -0.98% / -0.84% | -2.27% / -1.75% |
| bf16, plain 4096 / 6144, `--random-range-ratio 0` | n/s / +0.62% | n/s / -0.64% | -1.58% (6144) |
| bf16, plain 4096 / 6144, `--random-range-ratio 0.5` | +1.03% / +1.12% | -1.02% / -1.25% | -5.48% (6144) |

The 4096 cells of the `--random-range-ratio 0` round are marked n/s. Its four
passes interleave -- base 82.31 / 82.76 s against patch 82.57 / 81.69 s, with
patch_2 slower than base_1 -- and the patch arm's 0.88 s within-arm spread is
2.2x the 0.405 s between-arm delta, so by the rule in section 7 that round says
nothing at 4096. The 6144 half of the same round is clean (patch 98.78 / 98.34 s
against base 99.01 / 99.38 s). The -5.48% median TTFT has a 3.75% base-arm spread
(812.15 / 782.23 ms against a 43.71 ms delta); treat it as directional.

Throughput is `N/duration` with N fixed, quoted from vLLM's two-decimal req/s
field, which quantises at 0.5-0.75% here. Where it disagrees with the duration
column the duration figure is the one to take: the varlen-4096 cell reads +1.47%
from the req/s field and +1.03% from the durations, and the table now carries the
latter. Three of the throughput cells (aligned 4096, aligned 6144, varlen 6144)
have overlapping arms while their duration cells do not.

The first two rows use the deployment's own bench script (prefix 1229 + input
2867 and prefix 4301 + input 1843); the last two use the deployment's other
script (plain 4096 and 6144). The `--random-range-ratio 0.5` variant is not from
either script -- it was introduced here to make the packing gate fail, see 1c.

Two structural reasons the number is ~1% and not more, both outside the
operator: 256 output tokens put 87.5-88.2% of mean end-to-end latency in decode
on the fixed-length shapes and 92.5-93.2% on the range-ratio-0.5 ones, and decode runs
the untouched recurrent operator; and prompts that size against an 8192-token
budget admit only one or two sequences per prefill step, the operator's weakest
regime.

## 1c. What the operator gets from a live server, and what that costs

Everything in section 1 above uses `asl = [T//B]*B`. `probe2.py` builds it that
way and every shape in `bench_op_abba.sh` divides evenly (T/B = 8192, 512, 64,
8192, 512, 8192, 8192), so the alignment half of the gate is always satisfied.
The gate also needs `b > 1`, so only the three multi-request shapes (TP8 B=16,
TP8 B=40, TP4 B=16) take the cross-sequence packing path; the four B=1 rows never
do -- `packed = 0` there whatever the lengths -- and are unaffected by everything
below. Production does not produce the aligned batches those three need.

`patch_gdn.py` instruments `gdn.py` and logs what the operator actually
receives. Across 1950 calls on a live server (TP4, `--max-num-batched-tokens
8192`, one output token, concurrency 32, three prompt distributions):

* `b` takes the values 1, 4, 5, 8, 9, 10, 12, 14 and 18. The operator does see
  real multi-sequence batches -- this is not a `b == 1` situation.
* `packed = 0` in **100%** of those calls, **including** with
  `--random-range-ratio 0`.
* The cause is the chat template. `--random-input-len 1024` arrives at the
  operator as **1035** tokens and `512` arrives as **522**, so nothing is ever
  64-aligned. A sweep of `--random-input-len` over 1008..1024 found nothing
  aligned (remainders 59, 60, 61, 63, 63, 1, 2, 11); probing lower, 501 -> 512 for
  a single request -- but at that setting a batch of 7 arrives as T=3579 rather
  than 7x512=3584 and still reports `packed=0`, so a whole batch cannot be
  aligned from the client side either. (The probe prints `lens=` only for b=1, so
  "each prompt drifts by a token or two" is the inference; the b=7 total is the
  measurement.)

So the packed path is unreachable through this serving stack, and the -54.3%,
-85.2% and -45.5% rows describe a regime production never enters.

Measured on the captured vectors themselves (`real_seq_lengths.txt`,
`bench_real_vectors.sh`), three arms, four-arm ABBA over base/patch:

| Vector | b | Base (us) | Patch (us) | Delta | Patch, lengths padded to 64 | Delta |
|---|---|---|---|---|---|---|
| in=1024 r=0 | 9 | 2198.4 | 1883.2 | **-14.3%** | 1730.0 | -21.3% |
| in=1024 r=0 | 8 | 2135.9 | 1807.0 | **-15.4%** | 1740.2 | -18.5% |
| in=1024 r=0.5 | 9 | 2197.2 | 1889.8 | **-14.0%** | 1690.7 | -23.1% |
| in=512 r=0.5 | 14 | 2303.3 | 1973.4 | **-14.3%** | 1409.3 | -38.8% |
| in=512 r=0.5 | 18 | 2824.9 | 2402.3 | **-14.9%** | 1668.4 | -40.9% |

**-14.0% to -15.4% is the operator number for real traffic** (all five captured
vectors are nk=4 / nv=8, i.e. TP4 only; b=1 accounts for 300 of the 1950 logged
calls and is not in this table), and it is flat from b=8 to b=18 to within the
measurement's own within-arm spread, which reaches 2.9% on one patch arm.
`Base` here is the unmodified-source package on the same vendor path, not the
CANN built-in; section 1b measures those two 0.1-1.5% apart.

The padded column satisfies the gate and so bounds what fixing it would buy; it
is a lower bound, because it pays for 3.9-7.4% extra pad tokens that a
kernel-side fix would not. It is operator time only, and `probe5.py`'s padding is
a cost model rather than a valid transform, since its pad rows do not carry
beta = 0 / g = 0. Host-side padding is not a shortcut to this gain: `probe4.py`
measures the real thing on hardware -- bit-exact on both outputs in all ten rows
-- and net of the scatter/gather it costs +26.6% / +20.9% / +64.8% / +32.2% on
the 8192- and 4096-token shapes, with only T=2560 B=40 a win at -50.4%. The bound
is on a kernel-side fix. The value of the gate depends
strongly on `b`: padding removes a further 3.7-10.5% of the patched operator
time at b=8..9, but 28.6-30.6% at b=14..18 (3.1-9.1pp and 24.5-26.0pp against the
base arm, i.e. the difference of the two Delta columns above).

Folding that back, on the assumption that the end-to-end gain is linear in the
operator gain, and taking the operator's share of this prefill-dominated workload
as -2.54% / -14.3% ~= 18% -- an inference from two ABBA deltas, not a measured
share -- fixing the gate would take the 1024-token end-to-end gain from 2.5% to
roughly 3.8%. The base arm's own two passes (14.07 s, 13.83 s) put that
projection anywhere between 2.5% and 5.0%, so treat 3.8% as an order of
magnitude. Measuring the share rather than inferring it needs a profiler run
giving the operator's call count and per-call time against step time. At the
512-token, b=14..18 end it is worth considerably more, but the end-to-end run at
that shape had 21% within-arm spread and no number from it is quoted here.

The gate itself, isolated (`bench_packing_gate.sh`, four-arm ABBA; only the
length vector moves, the shape and the work are held fixed):

| Shape | Mode | packed | Base | Patch | Delta |
|---|---|---|---|---|---|
| TP4 B16 | even | 1 | 2747.9 | 1523.3 | -44.6% |
| TP4 B16 | lastoff | 1 | 2746.5 | 1521.8 | -44.6% |
| TP4 B16 | off1 | 0 | 2760.9 | 2389.6 | -13.4% |
| TP4 B16 | jit | 0 | 2731.7 | 2326.8 | -14.8% |
| TP8 B16 | even | 1 | 2208.8 | 1017.3 | -53.9% |
| TP8 B16 | lastoff | 1 | 2209.4 | 1019.9 | -53.8% |
| TP8 B16 | off1 | 0 | 2226.5 | 1887.7 | -15.2% |
| TP8 B16 | jit | 0 | 2221.7 | 1811.2 | -18.5% |
| TP8 B40 | even | 1 | 2531.1 | 398.1 | -84.3% |
| TP8 B40 | lastoff | 1 | 2549.4 | 397.3 | -84.4% |
| TP8 B40 | off1 | 0 | 2532.1 | 2020.4 | -20.2% |
| TP8 B40 | jit | 0 | 2586.4 | 2085.3 | -19.4% |
| TP4 B1 (control) | all four | 0 | 1959-1971 | 1684-1708 | -13.4 to -14.0% |

`off1` moves exactly one token from sequence 0 to sequence 1 and `lastoff` takes
one token off the final sequence. `lastoff` introduces one partial chunk, on the
final sequence; `off1` introduces two, on sequences 0 and 1, and one extra chunk
of work (129 against 128 at B=16, 41 against 40 at B=40). Only `off1` trips the
gate -- `packed` drops to 0 -- because the gate ignores the last sequence
(`bid + 1 < b` in `chunk_gated_delta_rule.h`), so `lastoff` stays `packed = 1`.
`lastoff` keeps the full -84.4% while `off1` collapses to -20.2%, and the base
arm moves at most 1.1% across the modes shown, 2.2% including TP8 B40 `jit`
(2531.1 to 2586.4) -- which cannot explain a 64pp swing, so the difference is the
code path, not the shape. The B=1 control is `packed = 0` in every mode and lands
within 0.7pp of itself, which is the noise floor.

Two things follow. Most of the multi-request gain in section 1 is the packing,
but not all of it. Strip the gate and TP4 B16 lands on the B=1 control (-13.4%
`off1`, -14.8% `jit`, against a -13.4 to -14.0% control), but TP8 B16 lands at
-15.2% / -18.5% and TP8 B40 at -20.2% / -19.4% -- 1 to 6pp better than the
control, so batching still buys something with `packed = 0`. That comparison
crosses a TP degree, because the only B=1 control in this experiment is TP4. And
`bench_op_abba.sh` alone cannot see any of this, which is why
`bench_packing_gate.sh` and `bench_real_vectors.sh` exist.

## 1b. Independent reproduction on a second machine

Run on 2026-09-06 on a different 910B4-1, in a container created for the purpose
from a stock image, against a fresh `ops-transformer` clone, with the kernel
source taken from this branch. Three arms: the CANN built-in, a package built
from unmodified source, and a package built from this branch.

Correctness reproduced: `COMPARED 36 tensors / RESULT: PASS all bit-exact`.

| Shape | built-in | base pkg | patch pkg | patch vs built-in | reported |
|---|---|---|---|---|---|
| TP8 T=8192 B=1 | 1480.4 | 1502.7 | 1272.8 | -14.0% | -15.9% |
| TP8 T=8192 B=16 | 2174.4 | 2207.9 | 1030.9 | -52.6% | -54.3% |
| TP8 T=2560 B=40 | 2521.1 | 2535.7 | 408.7 | -83.8% | -85.2% |
| TP4 T=8192 B=1 | 1946.9 | 1953.9 | 1709.6 | -12.2% | -13.6% |
| TP4 T=8192 B=16 | 2768.1 | 2765.1 | 1517.2 | -45.2% | -45.5% |
| TP2 T=8192 B=1 | 2612.4 | 2632.5 | 2419.2 | -7.4% | -9.8% |
| TP1 T=8192 B=1 | 4780.3 | 4776.9 | 4363.7 | -8.7% | -8.6% |

All seven land within 0.1-2.4pp of the numbers in section 1, slightly smaller
because the built-in arm ran last and picked up the machine's warmup. The base
package sits within -0.1% to +1.5% of the built-in, which is the point of having
it: the vendor-packaging path contributes nothing, so the difference is the
kernel.

## 2. Environment

| | |
|---|---|
| Device | 910B4-1 (20 AI Cube / 40 AI Vector cores, 192KB UB, 512KB L1) |
| CANN | 9.1.0 |
| torch_npu | 2.10.0.post4 (2.9.0.post4 does **not** have `npu_chunk_gated_delta_rule`) |
| vllm | 0.27.1, installed from a source distribution -- **not a git checkout, so there is no commit id** |
| vllm-ascend (runtime) | `5debbe58d1f0ed09621c0d427781de806ec04013`, plus 13 uncommitted local modifications |
| Kernel source | `ops-transformer`, branch `9.1.0`, `0684247` |

The operator is new in CANN 9.1.0. It is absent from 8.0.0, 8.3.RC1, 8.5.1,
9.0.0 and 9.0.1, verified three independent ways (`libopapi.so` symbols, the
`aclnnop` headers, and the opp kernel directory).

Two caveats on that table, both of which matter if you are trying to match the
numbers exactly rather than reproduce the direction:

The runtime vllm-ascend tree had 13 uncommitted modifications, so
`5debbe5` alone will not reconstruct it. One of them is a local switch added to
`vllm_ascend/ops/gdn.py` to force the Triton fallback for comparison; the rest
predate this work (megamoe, dsa_v1, sampler, cv_linear, model_runner_v1,
ascend_config, utils).

And the measurements did not go through this branch's build. See next section.

## 3. Two delivery paths, one kernel

The kernel reaches the device two different ways:

- **This branch** vendors the source into `csrc/attention/chunk_gated_delta_rule/`
  and builds it as part of vllm-ascend, which is what PR #12607 set up.
- **The measurements** built the same source into a CANN custom opp vendor
  package and installed it under `$ASCEND_OPP_PATH/vendors/`, where
  `torch_npu.npu_chunk_gated_delta_rule` picks it up through `load_priority`.

The second path was used because it toggles in about a second, which is what
makes a four-arm ABBA with package swaps practical at all. It also means the
runtime vllm-ascend version is irrelevant to the kernel under test: `5debbe5`
does not contain PR #12607's vendored kernel, it calls into CANN.

The two paths hand the compiler byte-identical source. That is the claim the
checksum chain below exists to support, and it is worth re-deriving rather than
taking on trust:

```
git show HEAD:csrc/attention/chunk_gated_delta_rule/op_kernel/arch22/chunk_gated_delta_rule_matmul_basic.h | md5sum
  -> 806d1f38...

pkg.run --noexec --extract=/tmp/x
find /tmp/x -name chunk_gated_delta_rule_matmul_basic.h | xargs md5sum
  -> 806d1f38...        # same bytes, and this is the package that was measured
```

Do this check. An earlier round of this work uploaded a patch but not a
newly-added file, so a stale copy survived on the build host and four
consecutive builds silently carried a variant that had supposedly been reverted
-- it compiled, it was bit-exact, the numbers looked good, and nothing anywhere
reported an error. Source-tree state is not evidence; extract the artifact and
hash it.

While you are extracting, `diff -rq` the base and patch packages. Here they
differed in exactly three files -- the header, its `.o`, and its `.json` --
which is what makes the A/B a single-variable experiment. That by-product is
worth more than the checksum.

## 3b. Preconditions — check these first

Three things decide whether the patched kernel is reached at all. Each one fails
silently, and each one yields a clean, correct-looking zero.

**The torch_npu binding.** The operator exists in CANN 9.1.0, but not every
torch_npu build exposes it to PyTorch:

```bash
python3 -c "import torch_npu; print(torch_npu.__version__, hasattr(torch_npu,'npu_chunk_gated_delta_rule'))"
```

`2.10.0.post2` prints `False`; `2.10.0.post4` prints `True`. Verified in one
container by changing nothing but the torch_npu version. On `False`,
vllm-ascend's `_probe_fused_chunk()` disables the fused path and everything runs
through Triton, so the two arms execute identical code.

**Which operator the call site uses.** There are two, and they are not the same
one:

| gdn.py calls | kernel comes from |
|---|---|
| `torch.ops._C_ascend.npu_chunk_gated_delta_rule` | vllm-ascend's own build of `csrc/` (what PR #12607 set up) |
| `torch_npu.npu_chunk_gated_delta_rule` | CANN, overridable by an opp vendor package |

`grep -n "chunk_gated_delta_rule" vllm_ascend/ops/gdn.py` says which. Building
this branch changes the first; installing the opp package changes the second.
Doing one while the runtime uses the other leaves the patch as dead code.

**Whether the vendor package is actually active.** On a CANN install with a
single custom vendor, `load_priority` has no trailing comma, so the obvious
`sed 's/^load_priority=gdrcust_transformer,//'` matches nothing and the "base"
arm silently keeps running the patched kernel. Clear the whole line and echo it
back:

```bash
sed -i 's/^load_priority=.*/load_priority=/' $ASCEND_OPP_PATH/vendors/config.ini
cat $ASCEND_OPP_PATH/vendors/config.ini
```

## 4. Build and install the custom opp package

```bash
git clone -b 9.1.0 https://gitcode.com/cann/ops-transformer.git
cd ops-transformer
K=attention/chunk_gated_delta_rule/op_kernel

# Copy the six headers from csrc/attention/chunk_gated_delta_rule/op_kernel/arch22/
# in this branch into $K. This repo is flat: no csrc/ prefix and no arch22/ level.
cp /path/to/branch/csrc/attention/chunk_gated_delta_rule/op_kernel/arch22/*.h $K/

# Because the layout is flat, the include of the tiling header must lose its
# "../". Without this the build fails with
#   error: '../chunk_gated_delta_rule_tiling_data.h' file not found
sed -i 's|#include "\.\./chunk_gated_delta_rule_tiling_data.h"|#include "chunk_gated_delta_rule_tiling_data.h"|' \
    $K/chunk_gated_delta_rule.h $K/chunk_gated_delta_rule_stage1.h \
    $K/chunk_gated_delta_rule_stage2.h $K/chunk_gated_delta_rule_stage3.h

bash build.sh --pkg --soc=ascend910b --vendor_name=gdrcust --ops=chunk_gated_delta_rule -j64
echo "build exit=$?"     # check it: a failed build leaves the previous .run in place
./cann-ops-transformer-gdrcust_linux-aarch64.run --quiet
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/gdrcust_transformer/op_api/lib/:$LD_LIBRARY_PATH
```

The build takes about 2.5 minutes. Installing puts `gdrcust_transformer` first
in `$ASCEND_OPP_PATH/vendors/config.ini`'s `load_priority`.

Check `build.sh`'s exit code rather than looking for a `.run` file. A failed
build leaves an earlier one on disk, and picking that up produces two packages
that are byte-identical -- which reads as "the change had no effect" rather than
"the change was never compiled".

The strongest baseline is a second package built from the *unmodified* source
rather than the CANN built-in: both arms then travel the same vendor path and
the only variable is the kernel. Measured here, that base package sits within
-0.1% to +1.5% of the built-in across all seven shapes, so the packaging path
itself contributes nothing.

Check that no other vendor in that list also provides this operator, or your
"base" arm is not the CANN built-in:

```bash
for d in $ASCEND_OPP_PATH/vendors/*/; do
  echo "$d: $(ls $d/op_impl/ai_core/tbe/kernel/ascend910b/ 2>/dev/null | tr '\n' ' ')"
done
```

## 5. Confirm the routing before trusting any number

```bash
GDR_OUT=/tmp/gdr GDR_DEV=0 SKIP_PERF=1 python ab_gdr.py check
```

`MAPS_PROBE` in the output lists what the process actually mapped. With the
package active it contains the vendor's `libcust_opapi.so`. Without that line
you are timing the built-in operator and will measure a difference of zero,
correctly.

## 6. Correctness

```bash
# NOT s/^load_priority=gdrcust_transformer,// -- on a single-vendor install there
# is no trailing comma, that matches nothing, and the "base" arm then runs the
# patched kernel against itself and prints PASS (section 3b).
sed -i 's/^load_priority=.*/load_priority=/' $ASCEND_OPP_PATH/vendors/config.ini
cat $ASCEND_OPP_PATH/vendors/config.ini   # must print load_priority= and nothing after
GDR_OUT=/tmp/gdr GDR_DEV=0 SKIP_PERF=1 python ab_gdr.py base

./pkg.run --quiet
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/gdrcust_transformer/op_api/lib/:$LD_LIBRARY_PATH
GDR_OUT=/tmp/gdr GDR_DEV=0 SKIP_PERF=1 python ab_gdr.py patch

GDR_OUT=/tmp/gdr python cmp_gdr.py base patch
# expected: COMPARED 36 tensors / RESULT: PASS all bit-exact
```

Bit-exactness over a fixed shape set is a strong check on the arithmetic and a
weak one on ordering. It cannot rule out a timing-dependent race, which is why
change 3 keeps an explicit Fixpipe->MTE2 handshake rather than relying on the
Fixpipe unit flag: the variant that leans on the unit flag is also bit-exact on
all 18 shapes and about 1-2% faster, and it was still rejected.

## 7. Operator level

```bash
GDR_DEV=0 bash bench_op_abba.sh | tee op_abba.log
```

Report base as `mean(A1, A2)` and patch as `mean(B1, B2)` per shape, and report
the within-arm spread alongside. A delta smaller than the spread is not a
result -- on the T=2560 B=40 shape the two arms of one sweep came out +4.6% and
-2.7%, opposite signs, which is the correct way to discover you have no signal.

This sweep only measures 64-aligned batches. For what the operator does on real
traffic, and for the packing gate that separates the two, run:

```bash
# needs BOTH packages: one built from unmodified source, one from this branch
GDR_DEV=0 BASE_RUN=/work/pkg_base.run PATCH_RUN=/work/pkg_patch.run \
  bash bench_packing_gate.sh  | tee packing_gate.log
GDR_DEV=0 BASE_RUN=/work/pkg_base.run PATCH_RUN=/work/pkg_patch.run \
  bash bench_real_vectors.sh  | tee real_vectors.log
```

To re-capture the length vectors on your own deployment rather than trusting the
ones in `real_seq_lengths.txt`:

```bash
python3 patch_gdn.py /path/to/vllm_ascend/ops/gdn.py apply
# the budget is per log file and is reset by deleting it, so point each bench at
# its own path; one 64-prompt bench logged 1080 calls, so 400 would truncate it
GDR_ASL_PROBE=2000 GDR_ASL_LOG=/tmp/asl_<bench>.log vllm serve ...   # then run a bench
python3 patch_gdn.py /path/to/vllm_ascend/ops/gdn.py revert
```

`revert` restores byte-for-byte from the `.gdrbak` it wrote; run it from a shell
`trap` so an interrupted probe cannot leave a shared container patched.

## 8. End to end

```bash
MODEL=/path/to/Qwen3.6-35B-A3B-w8a8 CARDS=4,5,6,7 bash bench_e2e_abba.sh | tee e2e_abba.log
```

`bench_e2e_abba.sh` as shipped hardcodes `--num-prompts 128` and benches twice
per arm. That is the 6.2-second configuration retracted in section 1, not the run
that produced the table there. The section 1 numbers come from 256 prompts, four
shape / range-ratio / seed combinations, and one bench per arm per shape. Set
`NUM_PROMPTS=256` and run one bench per arm before comparing against it.

Within one server instance the two passes of the old two-pass form differed by a
consistent ~3.5%, present in both arms; that figure predates this round and no
log in `logs/` contains a pass1/pass2 pair.

## 8b. Production-shaped scenario

`bench_scenario_prod.sh` carries the deployment configuration verbatim -- the
server flags and both bench workloads (prefix 1229 + input 2867, and prefix 4301
+ input 1843, 256 output tokens, 160 requests, concurrency 32) -- wrapped in the
same four-arm ABBA.

```bash
MODEL=/path/to/model CARDS=4,5,6,7 bash bench_scenario_prod.sh | tee prod.log
```

Its header lists the four deviations from the deployment script, of which one
matters for interpreting the result: the first runs used the W8A8 checkpoint,
while the deployment uses the unquantized bf16 one. Set `QUANT=""` to run bf16.

> An earlier revision predicted that the ~1% seen on W8A8 was an **upper bound**
> for bf16, reasoning that the operator is bf16 either way so its absolute time
> is fixed, while bf16 makes everything around it slower and therefore shrinks
> its share. That prediction was measured and did not hold: bf16 gave -0.98% /
> -0.84% duration and +1.26% / +0.91% throughput against W8A8's -0.55% / -0.98%
> and +0.67% / +1.08% -- the same order, if anything slightly larger. Total
> duration rose 13.6% (71.06 s -> 80.70 s) while the absolute saving roughly
> doubled, which contradicts "the operator's absolute saving is constant". No
> verified explanation; the way to get one is a profiler run on both checkpoints
> comparing the operator's call count and per-call time. Treat the W8A8 and bf16
> numbers as two measurements of the same ~1%, not as a bound and its interior.

## 8c. The kernel alone is ~0 end to end. What converts it. (2026-09-12/13)

Everything above section 8b measures the operator, or measures the server with
only the operator changed. That second thing is worth stating plainly, because
the first honest answer it gave was zero:

| Date | Arms | Median TTFT | Verdict |
|---|---|---|---|
| 2026-09-08 | stock kernel vs this kernel, six-arm ABBA, production harness | **-0.29%** | indistinguishable from zero; 95% CI on duration [-3.06%, +2.36%] |
| 2026-09-12 | same two arms, **both with the GDN host-dispatch patch** | **-2.12%** (p=0.029, arms separated) | real |

Same kernel, same machine, same workload, same harness. The only thing that
changed between the two rows is a host-side patch to `vllm_ascend/ops/gdn.py`
that is not in this branch -- it is on `gdn-host-dispatch-opt` (`4c57a8c1`) in
the same fork.

The mechanism: the device time this kernel saves was landing in "device finished,
now waiting for the host to dispatch the next op". Removing the host overhead in
the GDN layer is what lets it show up on the clock.

| | Ceiling from the operator's share | Before the host patch | After the host patch |
|---|---|---|---|
| Burst duration | -1.93% | -0.20% (10% converted) | **-1.78%** (92%) |
| Median TTFT | -2.90% | -0.29% (10%) | **-2.12%** (73%) |

Stacked against the unpatched base, both changes on: burst 6.116 -> 5.888 s
(**-3.7%**), median TTFT 3042 -> 2907 ms. Quote that TTFT number as
**-4.2% ~ -4.4%**, not -4.4%: the endpoint ratio is -4.44% but composing the two
measured halves, (1-2.10%)(1-2.12%), gives -4.18%; the 0.26pp gap is the shared
arm measuring 2978 in one round and 2970 in the other.

**Independently reproduced 2026-09-13** by someone else on the same machine, six
metrics, all same-direction. The kernel half nearly landed on the claim; the host
half came in lighter:

| Half | Claimed here | Reproduced |
|---|---|---|
| Kernel: duration / median TTFT / TPOT | -1.78% / -2.12% / -1.61% | -1.49% / **-2.04%** / -1.36% |
| Host patch: duration / median TTFT / throughput | -2.01% / -2.12% / +2.04% | -1.30~-1.47% / -1.21~-1.45% / +1.40~+1.50% |

That asymmetry is expected rather than troubling: the host patch pays off only at
the moments the device is actually waiting on the host (it removes ~750 ms of host
work per burst and ~123 ms of wall clock, a ~1/6 conversion), so its size moves
with machine load, while the kernel half does not. Composed from the reproduced
halves the stacked TTFT gain is -3.2% ~ -3.5%. **Give -3% ~ -4.5% as the
expectation to anyone who has not measured their own box.**

### Operator level on production shapes

Section 7's table is measured on 64-aligned synthetic batches. Section 1c already
warns that a live server never produces those. Measured directly on shapes taken
from a production trace (TP4, nk=4/nv=8, one process per shape, round-level ABBA,
per-row `/proc/self/maps` routing evidence, 32/32 rows consistent):

| Shape | stock (us) | this kernel (us) | Delta |
|---|---|---|---|
| T=8189 B=3 lens 2730,2730,2729 | 2021.7 | 1722.6 | -14.79% |
| T=8189 B=3 lens 4096,3000,1093 | 1962.9 | 1694.3 | -13.68% |
| T=8190 B=2 | 1887.8 | 1642.6 | -12.99% |
| T=8186 B=1 | 1967.2 | 1682.5 | -14.47% |
| T=6154 B=1 | 1503.4 | 1312.5 | -12.70% |
| T=4106 B=1 | 1062.4 | 948.9 | -10.68% |
| **production total** | **10405.4** | **9003.4** | **-13.47%** |
| control T=8192 B=1 / B=16 (aligned) | 1957.8 / 2753.7 | 1709.1 / 1499.5 | -12.70% / **-45.55%** |

16/16 output tensors bit-exact (bf16 compared as int16 -- comparing as float lets
`0.0 == -0.0` hide a bit difference).

Contamination on a shared machine is one-sided: a neighbour can only make you
slower. So the table reports the fastest round per arm, not the ABBA mean. It
mattered: one stock round ran 19.6% slower than its own arm's first round when a
neighbour started a service mid-run, which inflated the mean-based total to
-15.04%. The other seven cells agree between the two estimators within 0.4pp.

### Where the time went, and what is left

The operator emits **one fused kernel**
(`aclnnChunkGatedDeltaRule_ChunkGatedDeltaRule_C`), so Stage1/2/3 cannot be
attributed separately from `kernel_details.csv`; only per-pipe shares are
available. At T=8189 B=3, absolute pipe times in us (stock -> this kernel):

```
AIC  scalar 568.5 -> 311.4   fixpipe 443.9 -> 282.2   mte2 304.1 -> 277.3
AIV  scalar 566.6 -> 604.9   vector  425.0 -> 428.1   mte2 307.9 -> 309.7   mte3 130.3 -> 134.6
```

Cube side is down 432 us net, 419 us of it from scalar and fixpipe alone; all four
AIV pipes are unchanged. That matches the change list -- every one of the five
changes touches the cube pipeline or the core partition, none touches the vector
chain. AIC pipe sum 0.769 -> 0.628 (cube idle 23.1% -> 37.2%); AIV sum 0.757 ->
0.911 with `aiv_time` ~ kernel duration.

**The bottleneck has changed sides: the vector side is now the constraint.**
`aic_mac_ratio` is 0.034 -> 0.042, so the MAC array is ~96% idle and this operator
is not compute-bound in either state; "do less arithmetic" is not a direction.

The top remaining candidate is therefore **Stage2/Stage3 using only half the AIV**
(`GetSubBlockIdx() == 1` returns immediately) -- which is now supported by
measurement rather than by reading the code, and is the same order of magnitude as
the 605 us AIV scalar term. It is a candidate, not a verified gain. Section 9's
first entry is the reason for that distinction.

Profiling was used for attribution only. Any faster/slower conclusion comes from
runs without collection: the same shape reads -14.14% with the profiler attached
and -14.79% without.

### Workload caveat

All of the end-to-end numbers above are on a prefill-heavy burst (32 output
tokens). On a decode-dominated run (160 prompts x 4096 in / 256 out) both halves
fall below the detection floor and measure as zero -- 256 output tokens put
87.5-88.2% of mean end-to-end latency in decode, and the six-arm floor on that
workload is 2.71%. A null there is a statement about the instrument, not about the
kernel.

## 9. Things that produced wrong numbers here

**One shape per process.** Timing several shapes in one process inflated a
later small shape by 2.6x (626us read as 1631us). The inflation is not uniform:
the patched build, with smaller workspaces, was affected less, so a
same-process sweep exaggerates the win in the flattering direction.

**Cross-session absolute times are not comparable.** In two sessions the same
package read 1186.7us and 1196.0us for the same shape; base timings for TP4/TP2/
TP1 sat 3-4% apart between sessions. Only same-card same-session pairs mean
anything. A gain that "shrank" between writeups turned out to be a baseline
that moved, not a patch that regressed.

**TPOT is not a control under chunked prefill.** It is tempting to treat TPOT
as untouched -- decode runs a different operator -- and subtract it as a noise
floor. With `--enable-chunked-prefill` a scheduler step mixes prefill chunks
with decode tokens, so a decoding request's inter-token latency includes prefill
work done in the same step. TPOT improves genuinely when prefill gets faster.
Normalising by it deletes the result. There is no clean control in these runs;
the four-arm ABBA plus non-overlapping cells is the evidence.

**Killing the server does not free the cards.** Covered in the header of
`bench_e2e_abba.sh`. The criterion is `npu-smi`, not the process table.

**An orchestration script launched through `docker exec` outlives its parent.**
Killing the host-side waiter and the vllm processes left the inner loop running;
starting a second run then had two instances toggling the same global
`config.ini` and racing on the same port, each one's teardown killing the
other's server. None of that reports an error, it just corrupts the data. Count
instances (`pgrep -fc`), do not assume.

**`--additional-config` key placement is version-dependent.** `fuse_muls_add`
and `enable_npugraph_ex` belong inside `ascend_compilation_config`. At the top
level, an `AscendConfig` built with pydantic `extra="forbid"` refuses to start;
older versions without that setting accept the flat form and silently ignore
the keys, so the options are off and nothing says so.

**A synthetic length vector is a benchmark decision, not a neutral default.**
`asl = [T//B]*B` selects the kernel's packed path for every shape in
`bench_op_abba.sh`. Nothing in the harness said so and no output field showed
which path ran, so two headline rows described a regime production never
reaches. `probe3.py` prints `packed=` and `probe5.py` takes an explicit vector;
if a benchmark can silently pick a code path, make the path an output field.

**A retrying remote-exec wrapper runs the whole command twice.** The A2 helper
retries on a dropped connection. The retry re-ran a launcher that truncated the
log and started a second experiment instance; the two then installed different
packages under each other and interleaved into one file. The result looked
plausible. Guard launchers with an atomic `mkdir` lock -- both the remote script
and the launcher -- which is what `bench_packing_gate.sh` and
`bench_real_vectors.sh` now do.

**The `.run` installer does not activate the package it just installed.** It
writes `load_priority` only when `config.ini` does not already exist. Install
over an existing file -- for instance one a previous built-in arm emptied -- and
the files land but the vendor stays inactive, so the arm measures the CANN
built-in while claiming to measure the patch. Always write `load_priority`
explicitly after installing, and echo it.

**A short benchmark is a noisy benchmark.** `--num-prompts 128` at this
throughput is a 6.2-second run; the same configuration at 256 prompts (13.9 s)
moved the headline from +5.6% to +2.56%. Size the run so start-up and tail are
small, and quote the within-arm spread next to every number.

## 10. Rollback

Remove `gdrcust_transformer` from `load_priority` in
`$ASCEND_OPP_PATH/vendors/config.ini`. The built-in operator is used again on
the next process start; nothing else is modified.
