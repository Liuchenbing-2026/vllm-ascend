# SPDX-License-Identifier: Apache-2.0
"""PyTorch HyperConnection operations for the NPU bootstrap path."""

import os

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers the NPU op namespace)

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# ``QWEN38_HC_FUSED=1`` swaps the elementwise HyperConnection glue for fused
# NPU primitives (``npu_rms_norm`` + ``addcmul``). The reference path stays the
# default so the two can be A/B compared from the same source tree.
# ``QWEN38_HC_FUSED=2`` selects the single-kernel Triton port of the NVIDIA
# HyperConnection glue (combine + grouped Gemma RMSNorm in one launch).
# ``QWEN38_HC_FUSED=3`` keeps the fused NPU primitives but removes the fp32
# round-trips: the shared Gemma affine is handed to ``npu_rms_norm`` as a 1-D
# gamma (instead of a fp32 up-cast + broadcast mul + down-cast) and the HC
# combine/gate glue runs in the activation dtype. This drops the
# ``aclnnInplaceCopy_Cast`` traffic that dominated the HC glue in decode.
# ``QWEN38_HC_FUSED=4`` is the conservative variant: it keeps the fp32 combine
# and only removes the RMSNorm/gate_mix casts.
# ``QWEN38_HC_FUSED=5`` keeps the combine *and* gate glue in fp32 and only
# swaps the RMSNorm (bit-exact, no extra rounding).
# ``QWEN38_HC_FUSED=6`` composes the two earlier wins: the single-kernel Triton
# combine+grouped-GemmaRMSNorm of mode 2 *and* the cast-free gate/silu glue of
# modes 3/4. Decode is launch-overhead bound (the elementwise HC ops on
# ``[M, HC*HS]`` are ~4.6us each but ~0.6us of real work), so collapsing the
# combine+norm prologue from ~8 launches to 1 is the point.
_MODE = os.environ.get("QWEN38_HC_FUSED", "0")
_USE_FUSED = _MODE in ("1", "2", "3", "4", "5", "6", "7")
_USE_TRITON = _MODE in ("2", "6")
_USE_LEAN = _MODE in ("3", "4", "5", "6", "7")
_LEAN_COMBINE = _MODE in ("3", "6", "7")
_LEAN_GATE = _MODE in ("3", "4", "6", "7")
_LEAN_SILU = _MODE in ("3", "4", "6", "7")

# ``QWEN38_HC_FUSED=7`` is mode 3 plus ``npu_gemma_rms_norm`` for the
# per-branch affine, which folds the trailing ``aclnnMul`` of the HC RMSNorm
# prologue into the normalisation kernel (measured 62.5us -> 41.4us per call on
# ``[64, 4, 2560]`` while the box was loaded).  The op applies ``1 + gamma``
# itself, so the raw checkpoint weight is the operand.  Its normalisation runs
# at activation precision, so its tensor-level deviation from the fp32
# reference is ~4x the eager ``npu_rms_norm`` + ``Mul`` path
# (mean|d| 0.0072 vs 0.0019 on rms 1.0).  Mode 7 therefore only becomes a
# default after ``acc_tf.py`` clears it.
_USE_GEMMA_NORM = _MODE in ("7",)

# ``QWEN38_HC_GATE_TRITON=1`` is an orthogonal switch (it composes with any
# ``QWEN38_HC_FUSED`` mode that keeps the elementwise gate glue, i.e. modes
# 3/4/6). The gate mix is ``sigmoid(gate) * x`` reduced over the HC axis, which
# the eager path emits as three kernels on ``[M, HC*HS]``:
# ``aclnnSigmoid`` (45.7ms), ``aclnnMul`` (67.2ms) and ``aclnnMean`` (75.1ms)
# in the decode profile -- 3.9% of device time for ~3MB of traffic, so it is
# pure launch overhead. One Triton launch does all three.
_GATE_TRITON_ENV = os.environ.get("QWEN38_HC_GATE_TRITON", "0")
_GATE_TRITON = _GATE_TRITON_ENV in ("1", "2")

# ``QWEN38_HC_GATE_TRITON=2`` is the same fused gate mix with the three row
# strides declared ``tl.constexpr``.  Produced by the ``triton-latency-optimizer``
# skill's ordered scan: optimisation point 1 ("入参静态化") fires on
# ``references/constexpr_parameters.md``'s trigger 2 ("固定的 STRIDE") because
# ``_hc_gate_mix`` only takes this path when ``x``/``gate`` are 2-D with
# ``stride(1) == 1``, so ``stride(0) == HC * HS`` is a function of the model
# width rather than of the batch.  Launch-level specialisation folds the row
# address arithmetic into the (row, tile) decomposition and removes three
# runtime operands from the launch record.  The arithmetic inside the kernel is
# byte-identical to level 1, so the two are A/B comparable row for row.
_GATE_TRITON_STATIC = _GATE_TRITON_ENV == "2"

# ``QWEN38_HC_GATE2D`` is the round-211 rewrite of the same gate mix.  The
# level-1 kernel above walks the HC axis with ``tl.static_range(HC)``, i.e. it
# issues ``2 * HC`` strided 1-D loads of ``BLOCK`` lanes per program and builds
# ``2 * HC`` ``BLOCK``-lane address vectors for them.  In the k204dp1 decode
# window that shape costs 1.719 ms of ``aiv_time`` per step (2.031 ms of stream
# wall) over 106 launches -- 16.2 us per launch for 7.38 MB of traffic, ~456
# GB/s -- and the pipe split says the time is not all memory: ``aiv_vec_time``
# 0.968 ms and ``aiv_scalar_time`` 0.649 ms against ``aiv_mte2_time`` 1.037 ms.
# The rewrite loads one ``(HC, BLOCK)`` tile instead, so the per-program load
# count drops from ``2 * HC`` to 2 and the reduction over HC becomes a
# ``tl.sum(axis=0)``.
#
# Levels are a (BLOCK, ROWS) table so that one build can price several shapes
# from the same source tree -- level 5/6 additionally amortise the per-program
# prologue over 4/8 rows.  Rows only divide the grid when ``n`` divides, so a
# row count that does not divide the token count silently falls back to one row
# per program: the microbenchmark runs at a fixed ``n`` and would not have
# caught a ragged grid.
_GATE2D_ENV = os.environ.get("QWEN38_HC_GATE2D", "0")
_GATE2D = _GATE2D_ENV in ("1", "2", "3", "4", "5", "6")

# ``QWEN38_HC_CNROW=1`` folds the HC combine *and* the grouped GemmaRMSNorm it
# feeds into one row-tile launch, i.e. the round-218 geometry (``=3``, one
# program per row with HC as a 2-D tile axis) applied to the round-206 fused
# combine+norm kernel that mode 6 still ships.
#
# Why the shipped fusion is not the same thing: ``_hc_combine_norm_kernel``
# (modes 2/6) gives one program one ``(row, stream)`` pair and walks the row in
# two masked ``BLOCK_SIZE = 512`` passes -- the first stores the rounded
# combine result and accumulates the sum of squares, the second re-reads it
# from global memory to apply the affine.  That is ``n * HC`` programs of two
# passes over ``HS``.  The delivery line instead runs *two* row-tile launches:
# ``_hc_combine_mix_kernel_2d`` writes ``[n, HC, HS]`` and
# ``_grouped_gemma_rmsnorm_row_kernel`` immediately reads the whole block back.
# Folding them keeps the round-218 row geometry (``n`` programs, unmasked
# 2048 + 512 legs, ``tl.sum(axis=1)``) and keeps the combined value in
# registers between the two halves, so the extra cost against the pure norm
# kernel is the residual/block/injection read and the ``combined`` write --
# not a second pass of the whole block.
#
# Promoted to the code default in round 219b.  Three same-seed brackets against
# the two-launch chain, on dev 8-15 at the delivery point (8K/1K, C40, dnum 160,
# champion env, the switch the only difference):
#
#   seed 4910   k219c1 939.89  ->  k219n1 960.58   +2.20 %
#   seed 4912   k219c3 932.10  ->  k219n3 959.55   +2.95 %   (candidate row first)
#   seed 4911   k219c2r 942.29 ->  k219n2r 972.86  +3.25 %   (both rows re-run)
#
# The first pass at seed 4911 read the control at 891.68, 4 % under every other
# control on this box, and priced the pair at +8.95 %; the re-run put the control
# back at 942.29 while the candidate reproduced (971.56 -> 972.86, 0.13 %), so
# that pair is reported at its replicated value.  ``_hc_combine_norm_row_kernel``
# is ``torch.equal`` to the two-launch chain on both outputs (hc_cnrow_check.py,
# per-branch and shared affine, two seeds plus a saturation-edge row), so this is
# a pure price measurement and there is no accuracy gate to clear.
_CNROW_ENV = os.environ.get("QWEN38_HC_CNROW", "1")
_CNROW = _CNROW_ENV in ("1", "2")
_GATE2D_TABLE = {
    "1": (512, 1),
    "2": (640, 1),
    "3": (1280, 1),
    "4": (256, 1),
    "5": (512, 4),
    "6": (512, 8),
}

# ``QWEN38_HC_COMBINE_TRITON=1`` applies the round-85 lesson to the next glue
# in the block: the HC combine is ``2 * sigmoid(logits / HC)`` (Divs + Sigmoid
# + two [*, 4] casts) followed by ``addcmul(residual, block, injection)`` on
# ``[rows, HC, HS]``.  In the DP1 decode window that is 2652 launches of
# ``aclnnAddcmul`` (50.0 ms) plus 2652 of ``aclnnDivs`` and the same number of
# small Sigmoids and casts -- ~68 ms / 3.3 % of decode busy, and the gate
# measurement showed these glue launches return ~1.65x their busy share in
# tok/s.  One launch replaces the whole chain.  The reduction in launch count
# is the point, not the traffic: this kernel reads and writes the same bytes
# as the eager ``addcmul``.
# ``=2`` is the same fold with the round-211 access-pattern fix.  The 1-D form
# above walks the HC axis as a python-level loop of separate strided 1-D loads;
# the graph-replay probe (``hc_gate_probe3.py``, round 212) prices the two at
# the delivery shape -- n=160, inside an npu graph so the launcher is not part
# of the number:
#
#     variant        n=160      n=40
#     eager addcmul  22.06 us   13.86 us
#     combine 1-D    167 ns/row @16000 (2802 us there)  -- the -4.87 % of r117
#     combine 2-D    19.71 us    9.51 us   bit-exact with eager
#
# so ``=2`` keeps the single launch and the bit-exactness of ``=1`` while
# reaching the (HC, BLOCK) tile that the gate rewrite needed to hit its
# ceiling.  ``=1`` is retained because its verdict is part of the ledger.
_COMBINE_MODE = os.environ.get("QWEN38_HC_COMBINE_TRITON", "0")
_COMBINE_TRITON = _COMBINE_MODE in ("1", "2")
_COMBINE_2D = _COMBINE_MODE == "2"

# ``QWEN38_HC_SILU_TRITON=1`` folds the last un-fused pair of launches in the
# HC module into one.  ``hc_silu`` is ``F.silu(lora / hc_count)`` on
# ``[rows, lora_rank]``; with ``QWEN38_HC_FUSED=3`` that is ``aclnnDivs_RealDiv``
# (7.55 us) followed by ``aclnnSilu`` (6.66 us), 97 times per decode step in the
# round-75 window -- 0.866 ms/step of a 69.71 ms step for ~200 KB of traffic.
# Orthogonal to the gate and combine switches; it only composes with modes that
# keep the eager ``hc_silu`` (3/4/6/7), which is what the delivery line uses.
_SILU_TRITON_LEVEL = os.environ.get("QWEN38_HC_SILU_TRITON", "0")
_SILU_TRITON = _SILU_TRITON_LEVEL in ("1", "3")
# Level 3 is the division-free 2-D kernel.  Level 1 is the flattened kernel
# whose ``offs // n_col`` cost 505 us per call on the vector unit (see the
# ledger, 16.89): it stays wired to what it measured, so an old row keeps its
# meaning.  The two levels are separate code paths, selected here and nowhere
# else.
_SILU_TRITON_V3 = _SILU_TRITON_LEVEL == "3"
_SILU_BLOCK = 4096
_SILU_BLOCK_R = 16
_SILU_BLOCK_C = 64

# ``QWEN38_HC_NORM_TRITON=1`` folds the last un-fused *pair* of launches in the
# HC prologue -- ``npu_rms_norm(rows, ones, eps)`` followed by the ``(1 + w)``
# affine multiply -- into the one grouped-GemmaRMSNorm Triton kernel that the
# ``QWEN38_HC_FUSED=2/6`` modes already carry (``y = x * rsqrt(mean(x^2) + eps)``
# then ``y += y * w``).  In the k204dp1 delivery window that pair is 97
# incidences of ``RmsNorm`` (2123.5 us of aiv time per decode step) plus 97 of
# ``aclnnMul`` (1480.7 us) of a 61.9 ms step, i.e. ~3.6 ms for ~4.1 MB of
# traffic that a single launch moves in 1.64 MB.
#
# It is orthogonal to ``QWEN38_HC_GATE_TRITON`` / ``QWEN38_HC_SILU_TRITON`` and
# composes with every mode that keeps the LEAN prologue (3/4/5/6/7); the
# combine glue is deliberately *not* touched, because the fused combine is the
# one fold this campaign already priced and lost (``HC_COMBINE_TRITON``, -4.87 %).
#
# Precision: the eager pair normalises in fp32 and then rounds to the
# activation dtype twice -- once at the ``npu_rms_norm`` output and once at the
# multiply -- while the Triton kernel rounds once, at the store, so the fold
# carries *fewer* activation-precision roundings than the line it replaces.
# The rounding sites move, so it is still A/B tested through ``acc_tf.py``.
# ``QWEN38_HC_NORM_TRITON=2`` and ``=3`` keep the same fold but re-lay the
# launch geometry, because k218prof showed the *geometry* is what this kernel
# pays for.  The op is ``[M, HC*HS]``, the reduction axis is ``HS = 2560`` and
# ``=1`` derives ``BLOCK_SIZE = next_power_of_2(2560) = 4096``, so 37.5 % of
# every vector repeat is masked away.  ``2560 = 2048 + 512`` is two exact
# powers of two, so the same arithmetic runs as two *unmasked* legs.
# ``hc_norm_probe.py`` priced all three shapes on an idle A3 die, us per launch
# for the same 6.56 MB of unique traffic:
#   M=640   =1 52.1   =2 43.5 (1.20x)   =3 34.8 (1.50x)   no-reduce ceiling 33.0
#   M=1600  =1 122.5  =2 100.7 (1.22x)  =3 74.2 (1.65x)   no-reduce ceiling 72.9
# ... and the box agreed, but not with the probe's arithmetic.  sweep218 (dev
# 8-15, 8192 in / 1024 out / C40, seeds 4900/4901, same-box same-seed bracket)
# read =1 883.06/907.34 -> 895.20, =2 926.78/926.11 -> 926.44 (+3.49 %), and
# =3 939.59/945.79 -> 942.69 (+5.31 % over =1, +1.76 % over =2, both seeds
# moving the same way on both steps).  +5.31 % is far more than the ~1.5 % of
# step wall that the 1998 us/step kernel's vector time can explain, which is
# the useful part of the reading: at M = 160 the =1 geometry launches
# 160 x HC = 640 programs per call for 6.6 MB of traffic, so what it was
# paying for was *program dispatch*, not vector work -- and that cost is
# visible in Duration but not in aiv_time.  =3 launches 160.
#
# ``=3`` is therefore the shipping default from round 218 on.  The gate is
# hc_norm_check.py (kernel level: =1 4/1.64 M elements differ from the fp32
# torch reference, max |d| = 7.812e-03 = one bf16 ulp; =3 8/1.64 M, same
# one-ulp bound) plus the teacher-forced logprob probe on one box,
# back to back: =3 vs =1 gives mean_abs 0.098393 / p99 0.66057, and the
# *same-code self-repeat* of the control gives 0.096409 / 0.63288 -- i.e. the
# change sits inside the probe's own run-to-run spread.  Tail, stated honestly:
# 2 of 2802 tokens move by more than 2.0 nats under =3 (1.31 max in the
# self-repeat), which is what a one-ulp change on 1e-5 of activations can do to
# a near-tie; mean and p50 are unchanged.
# ``=2`` is that split on the shipped grid; ``=3`` additionally makes one
# program own a whole *row*, so the HC streams are a 2-D tile and the group
# reduction collapses to one axis-1 ``tl.sum``.  Both reproduce ``=1`` to
# within one bf16 ulp on fewer than 1e-5 of elements.  Below ~M=500 the probe's
# ~29 us per-launch floor hides every difference -- a bare ``Tensor.copy_`` of
# the same traffic costs the same 29 us -- so the small-M (decode) ranking is
# decided on the box, not in the probe.
_NORM_MODE = os.environ.get("QWEN38_HC_NORM_TRITON", "3")
_NORM_TRITON = _NORM_MODE in ("1", "2", "3")
_NORM_GEOM = {"1": 1, "2": 2, "3": 3}.get(_NORM_MODE, 0)
# 2560 = 2048 + 512, the two exact legs the split kernels are written for.
_NORM_SPLIT_A = 2048
_NORM_SPLIT_B = 512

# ``QWEN38_HC_LEAN_INJECT=1`` keeps the HC injection glue
# (``2 * sigmoid(logits / hc_count)``) in the activation dtype instead of
# up-casting the ``[M, HC]`` logits to fp32 and casting the gate back. The two
# ``aclnnInplaceCopy_Cast`` launches per HC block are ~12.6ms each in the
# decode window on a ``[64, 4]`` tensor, i.e. pure launch overhead. The
# division by ``hc_count`` stays exact (``HC = 4`` is a power of two) and only
# the ``sigmoid`` rounding moves from fp32 to bf16, so this is A/B tested
# through ``acc_tf.py`` before it is allowed to become the default.
_LEAN_INJECT = os.environ.get("QWEN38_HC_LEAN_INJECT", "0") == "1"

# ``QWEN38_HC_INJECT_TRITON=1`` is the *narrow* half of the fold that
# ``QWEN38_HC_COMBINE_TRITON`` priced and lost (-4.87 %, sweep111).  The eager
# injection gate is ``(2 * sigmoid(logits.float() / HC)).to(dtype)`` on the
# ``[rows, HC]`` logits -- at our batch that is a **640-element** tensor, so
# every one of the five launches it emits is pure launch overhead:
#
#   ``injection_logits.float()``   -> aclnnInplaceCopy_Cast
#   ``/ hc_count``                 -> aclnnDivs_RealDiv
#   ``torch.sigmoid``              -> aclnnSigmoid
#   ``2.0 *``                      -> aclnnMuls
#   ``.to(dtype)``                 -> aclnnInplaceCopy_Cast
#
# 97 incidences per decode step (one per HC block, 48 layers + the MTP block),
# i.e. ~485 launches/step for 1.3 KB of data.  In ``prof/k204dp1``'s rank0
# census those five families report 168 ``aclnnSigmoid`` calls/step (377 us),
# 108 ``aclnnMuls`` (164 us), ~97 of the 208 ``aclnnDivs`` (490 us) and ~194 of
# the 274 ``aclnnInplaceCopy_Cast`` (332 us) -- ~1.4 ms/step of a 62 ms step
# spent launching, not computing.
#
# ``HC_COMBINE_TRITON`` bundled this gate with the ``addcmul`` that consumes it
# and lost; the ledger's traffic arithmetic (§16.105.3) says the combine itself
# is why (it reads 3.28 MB + 0.82 MB and writes 3.28 MB in one Triton launch
# that the aclnn ``Addcmul`` does at a better rate).  This switch therefore
# leaves ``addcmul`` exactly as it is and folds only the gate.
#
# Precision: the kernel is the default eager chain with the up-cast moved
# inside it -- load the activation dtype, compute ``x / HC``, ``sigmoid`` and
# ``* 2`` in fp32, and round **once**, at the store, exactly where the eager
# ``.to(dtype)`` rounds.  Same operands, same fp32 arithmetic, same single
# activation-precision rounding site, so it stays A/B comparable and any
# accept still faces the ``acc_tf.py`` gate.  With ``QWEN38_HC_LEAN_INJECT=1``
# this switch is deliberately inert: that line rounds the sigmoid in bf16 and
# this kernel does not, so the two must not be mixed.
_INJECT_TRITON = os.environ.get("QWEN38_HC_INJECT_TRITON", "0") == "1"
_INJECT_BLOCK = 1024

# ``QWEN38_HC_NORM_W=1`` folds the HC RMSNorm affine (``1 + weight``) into the
# ``npu_rms_norm`` call itself, as that op's ``gamma`` operand, instead of
# normalising with a ``ones`` gamma and multiplying the result afterwards.
#
# ``rms_norm(x, ones) * w  ==  rms_norm(x, w)`` exactly -- the same epsilon,
# the same normalisation axis -- so the following disappears from every HC
# block (per call, and the shapes below are the live ones):
#
#   ``aclnnMul``  "[n, 4, 2560]"      -- 3294 calls/window, the third-largest
#                                       elementwise op in the prefill phase
#   ``aclnnInplaceCopy_Cast``         -- the per-call gamma cast, already
#                                       hoisted by ``_affine_view_dtype``
#
# This is the substitute for the ``MulsAddFusionPass``: that pass matches
# ``x * routed_scaling_factor + y``, and the Qwen3.8 checkpoint has no
# ``routed_scaling_factor`` (defaults to 1.0) *and* its ``Muls`` is followed by
# a ``Cast`` rather than an ``Add``, so the pass matches zero times.  Folding
# the gamma here removes a real kernel with no change to the arithmetic.
#
# Scope guard: only the *shared* affine (``weight.numel() == hidden_size``,
# which is what ``hc_per_branch_norm = false`` gives) can be handed to the op
# as a 1-D gamma.  The per-branch ``[HC, HS]`` layout would need the weight
# tiled to ``[n*HC, HS]`` to match the operand's leading axis, which is a
# materialisation this fold is meant to avoid -- that case keeps the Mul.
_HC_NORM_W = os.environ.get("QWEN38_HC_NORM_W", "0") == "1"

# ``QWEN38_HC_AFFINE_RMS=1`` folds the trailing ``aclnnMul`` of the HC prologue
# into the normalisation itself, for the *per-branch* affine that this
# checkpoint actually carries (``hc_per_branch_norm = True`` -> ``[HC, HS]``).
#
# ``rms_norm(rows, ones) * gamma`` and ``rms_norm(x3, gamma)`` are the same
# expression: the op normalises the trailing axis of its operand and broadcasts
# ``gamma`` over the leading axes, so an operand shaped ``[n, HC, HS]`` with a
# ``[HC, HS]`` gamma applies branch ``c``'s affine to branch ``c``'s stream --
# which is what the rank-2 broadcast of the eager multiply does.  One launch
# replaces two on ``[640, 2560]`` + ``[160, 4, 2560]``: 97 incidences each per
# decode step, 1.282 ms of the 69.71 ms step is the multiply alone.
#
# The fp32 normalisation is unchanged (same operand, same ``eps``, same
# reduction axis as the shipped ``npu_rms_norm(rows, ones, eps)``); only the
# multiply moves inside the op.  That is the difference between this switch and
# ``QWEN38_HC_FUSED=7``, which normalises at activation precision through
# ``npu_gemma_rms_norm``.
_AFFINE_RMS = os.environ.get("QWEN38_HC_AFFINE_RMS", "0") == "1"


@triton.jit
def _injection_gate_kernel(
    x_ptr,
    out_ptr,
    n,
    hc_f,
    BLOCK: tl.constexpr,
) -> None:
    """``(2 * sigmoid(x.float() / hc_count)).to(out.dtype)`` in one launch.

    Flat 1-D over the ``[rows, HC]`` logits (``HC`` is 4, so the block is one
    row per lane group and the row/column decomposition that
    ``_hc_silu_div_kernel_2d`` needed buys nothing here).  The mask is against
    a runtime ``n`` because the row count is the batch.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, 2.0 * tl.sigmoid(x / hc_f), mask=mask)


def _inject_triton_ok(
    injection_logits: torch.Tensor, dtype: torch.dtype
) -> bool:
    """Preconditions for the one-launch injection gate.

    The kernel indexes the operand flat, so it needs a contiguous ``[*, HC]``
    block; ``HC`` itself is passed at runtime, so no width constraint is
    needed.  Returns False whenever the switch is off, which is what keeps the
    default line's arithmetic untouched.
    """
    return (
        _INJECT_TRITON
        and not _LEAN_INJECT
        and injection_logits.is_contiguous()
        and injection_logits.numel() > 0
    )


def _injection_gate_triton(
    injection_logits: torch.Tensor, hc_count: int, dtype: torch.dtype
) -> torch.Tensor:
    """Launch ``_injection_gate_kernel`` and return the ``[*, HC]`` gate."""
    out = torch.empty(
        injection_logits.shape, dtype=dtype, device=injection_logits.device
    )
    n = injection_logits.numel()
    _injection_gate_kernel[(triton.cdiv(n, _INJECT_BLOCK),)](
        injection_logits,
        out,
        n,
        float(hc_count),
        _INJECT_BLOCK,
    )
    return out


def _injection_gate(
    injection_logits: torch.Tensor, hc_count: int, dtype: torch.dtype
) -> torch.Tensor:
    """``2 * sigmoid(logits / hc_count)`` for the HC injection glue.

    The eager path up-casts the ``[M, HC]`` logits to fp32 and casts the gate
    back down, which is two extra ``aclnnInplaceCopy_Cast`` launches per HC
    block for a tensor that holds ``M * HC`` elements. With
    ``QWEN38_HC_LEAN_INJECT=1`` the whole expression stays in the activation
    dtype: ``HC`` is a power of two so the division is exact either way and
    only the ``sigmoid`` rounding moves from fp32 to bf16.
    """

    if _inject_triton_ok(injection_logits, dtype):
        return _injection_gate_triton(injection_logits, hc_count, dtype)
    if _LEAN_INJECT:
        return (2.0 * torch.sigmoid(injection_logits / hc_count)).to(dtype)
    return (2.0 * torch.sigmoid(injection_logits.float() / hc_count)).to(dtype)


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    GROUP_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    # ``BLOCK_SIZE``/``GROUP_DIM`` are plain host-side ints: deriving them with
    # ``DIM // NUM_GROUPS`` inside the kernel yields a ``constexpr`` object and
    # ``triton.next_power_of_2`` then dies with
    # ``'constexpr' object has no attribute 'bit_length'`` once the kernel is
    # reached through the compiled graph path.

    pid = tl.program_id(0)
    group_id = pid % NUM_GROUPS
    row = pid // NUM_GROUPS

    offs_g = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * GROUP_DIM + offs_g
    mask = offs_g < GROUP_DIM
    # A [GROUP_DIM] affine is shared; a [DIM] affine follows the grouped
    # checkpoint layout.
    w_offs = offs_g if W_SHARED else offsets

    x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + w_offs, mask, other=0.0)

    rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    y = x * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offsets, y, mask)


@triton.jit
def _grouped_gemma_rmsnorm_split_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    GROUP_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SPLIT_A: tl.constexpr,
    SPLIT_B: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    """``QWEN38_HC_NORM_TRITON=2``: the shipped grid, unmasked legs.

    Identical operand mapping, arithmetic and rounding sites as
    ``_grouped_gemma_rmsnorm_kernel``; only the load geometry changes.  The one
    masked 4096-lane ``tl.arange`` becomes ``SPLIT_A + SPLIT_B`` (2048 + 512 at
    ``HS = 2560``), both of which are exact powers of two and therefore need no
    mask at all.  The two partial sums are added in fp32, so the result differs
    from the shipped kernel only by the fp32 summation order -- one bf16 ulp on
    a handful of elements out of 1.6 M in the probe.
    """
    pid = tl.program_id(0)
    row = pid // NUM_GROUPS
    # checklist: no ``%`` -- ``NUM_GROUPS`` is a power of two constexpr, but
    # the explicit form is the one the Ascend lowering is written against.
    group_id = pid - row * NUM_GROUPS
    base = group_id * GROUP_DIM
    row_x = row * stride_x
    row_y = row * stride_y

    oa = tl.arange(0, SPLIT_A)
    ob = SPLIT_A + tl.arange(0, SPLIT_B)
    wbase = 0 if W_SHARED else base

    xa = tl.load(x_ptr + row_x + base + oa).to(tl.float32)
    xb = tl.load(x_ptr + row_x + base + ob).to(tl.float32)
    wa = tl.load(w_ptr + wbase + oa)
    wb = tl.load(w_ptr + wbase + ob)

    rrms = tl.rsqrt((tl.sum(xa * xa) + tl.sum(xb * xb)) / GROUP_DIM + EPS)

    ya = xa * rrms
    ya += ya * wa.to(tl.float32)
    yb = xb * rrms
    yb += yb * wb.to(tl.float32)
    tl.store(y_ptr + row_y + base + oa, ya)
    tl.store(y_ptr + row_y + base + ob, yb)


@triton.jit
def _grouped_gemma_rmsnorm_row_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    GROUP_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SPLIT_A: tl.constexpr,
    SPLIT_B: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    """``QWEN38_HC_NORM_TRITON=3``: one program per row, HC as a tile axis.

    The split legs of ``=2`` are kept, but the ``NUM_GROUPS`` streams of one
    row are fetched as a single ``(NUM_GROUPS, SPLIT_*)`` tile.  The affine is
    read once per row instead of once per (row, group) -- it is only 20 KB and
    L2-resident, but at ``M = 1600`` the shipped geometry re-reads it 6400
    times, i.e. 32.8 MB of L2 traffic against 65.6 MB of real x/y traffic --
    and the group reduction becomes one axis-1 ``tl.sum`` rather than
    ``NUM_GROUPS`` scalar reductions.  ``NUM_GROUPS`` must be a power of two
    (it is ``tl.arange``'s bound); the caller checks that, and ``HC = 4``.
    """
    row = tl.program_id(0)
    oa = tl.arange(0, SPLIT_A)
    ob = SPLIT_A + tl.arange(0, SPLIT_B)
    hoff = tl.arange(0, NUM_GROUPS)[:, None] * GROUP_DIM
    ha = hoff + oa[None, :]
    hb = hoff + ob[None, :]
    row_x = row * stride_x
    row_y = row * stride_y

    if W_SHARED:
        wa = tl.load(w_ptr + oa)
        wb = tl.load(w_ptr + ob)
    else:
        wa = tl.load(w_ptr + ha)
        wb = tl.load(w_ptr + hb)
    xa = tl.load(x_ptr + row_x + ha).to(tl.float32)
    xb = tl.load(x_ptr + row_x + hb).to(tl.float32)

    ss = tl.sum(xa * xa, axis=1) + tl.sum(xb * xb, axis=1)
    rrms = tl.rsqrt(ss / GROUP_DIM + EPS)[:, None]

    ya = xa * rrms
    ya += ya * wa.to(tl.float32)
    yb = xb * rrms
    yb += yb * wb.to(tl.float32)
    tl.store(y_ptr + row_y + ha, ya)
    tl.store(y_ptr + row_y + hb, yb)


def _norm_triton_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Preconditions for the one-launch grouped GemmaRMSNorm fold.

    ``QWEN38_HC_NORM_TRITON`` stays inert for anything the kernel was not
    written for: it indexes the operand as ``[rows, num_groups * group_dim]``
    with a unit last stride, needs the affine to be one contiguous
    ``[num_groups, group_dim]`` (or ``[group_dim]``) block, and only the
    activation dtypes appear on the delivery line.
    """
    return (
        x.is_contiguous()
        and x.dim() >= 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and weight.is_contiguous()
        and x.shape[-1] % 16 == 0
    )


def _grouped_gemma_rmsnorm_triton(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    """``_grouped_gemma_rmsnorm_kernel`` as a drop-in for the eager pair.

    Evaluates ``y = x * rsqrt(mean(x^2) + eps) * (1 + w)`` with the affine
    grouped along the last axis -- the same expression (and the same
    per-group reduction axis) as ``npu_rms_norm(rows, ones, eps)`` followed by
    ``_affine_view_dtype``, in one launch instead of two.  ``x`` may carry any
    leading shape whose last axis is ``num_groups * group_dim``; the operand is
    flattened to ``[rows, num_groups * group_dim]`` exactly like
    ``_rms_norm_groups`` does, so callers holding an already-split
    ``[n, num_groups, group_dim]`` tensor pass ``combined.reshape(n, -1)``.
    """
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    group_dim = flat.shape[-1] // num_groups
    out = torch.empty_like(flat)
    w_shared = weight.numel() == group_dim
    rows = flat.shape[0]
    # ``=2``/``=3`` are written for the shipping ``HS = 2560`` split; anything
    # else keeps the general masked kernel rather than silently re-deriving it.
    split_ok = group_dim == _NORM_SPLIT_A + _NORM_SPLIT_B
    if _NORM_GEOM >= 3 and split_ok and (num_groups & (num_groups - 1)) == 0:
        _grouped_gemma_rmsnorm_row_kernel[(rows,)](
            flat,
            weight,
            out,
            flat.stride(0),
            out.stride(0),
            group_dim,
            num_groups,
            _NORM_SPLIT_A,
            _NORM_SPLIT_B,
            W_SHARED=w_shared,
            EPS=eps,
        )
        return out.reshape(shape)
    if _NORM_GEOM >= 2 and split_ok:
        _grouped_gemma_rmsnorm_split_kernel[(rows * num_groups,)](
            flat,
            weight,
            out,
            flat.stride(0),
            out.stride(0),
            group_dim,
            num_groups,
            _NORM_SPLIT_A,
            _NORM_SPLIT_B,
            W_SHARED=w_shared,
            EPS=eps,
        )
        return out.reshape(shape)
    _grouped_gemma_rmsnorm_kernel[(flat.shape[0] * num_groups,)](
        flat,
        weight,
        out,
        flat.stride(0),
        out.stride(0),
        group_dim,
        num_groups,
        triton.next_power_of_2(group_dim),
        W_SHARED=weight.numel() == group_dim,
        EPS=eps,
    )
    return out.reshape(shape)


@triton.jit
def _hc_combine_norm_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
) -> None:
    # Derived on the host for the same reason as in
    # ``_grouped_gemma_rmsnorm_kernel``: ``triton.next_power_of_2`` cannot be
    # handed a ``constexpr`` that was produced inside the kernel body.
    #
    # One program owns one ``(row, stream)`` pair and walks the row in
    # ``BLOCK_SIZE`` tiles rather than materializing a
    # ``(NUM_TILES_PAD, BLOCK_SIZE)`` 2-D accumulator.  The 2-D form kept
    # 8 x 512 fp32 accumulators plus three 8 x 512 operand tiles live, which is
    # the UB-overflow shape the in-tree RoPE kernel guards against on A2/A3
    # (``ops/triton/rope.py``: "Large head_dim RoPE can overflow UB with the
    # default tile"); on ``QWEN38_HC_FUSED=6`` it reproduced as an op-level
    # failure during startup.  Two tiled passes replace it: the first also
    # stores the rounded combine result and accumulates the sum of squares, the
    # second re-reads it and applies the affine.  The arithmetic - including
    # the bf16 rounding of ``out`` before the norm - is unchanged.
    pid = tl.program_id(0)
    row = pid // HC
    stream = pid % HC
    base = stream * HC_DIM
    inj = tl.load(inj_ptr + row * stride_inj + stream)
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)

    sum_sq = 0.0
    for tile in tl.static_range(NUM_TILES):
        offs_inner = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_inner = offs_inner < HC_DIM
        offs = base + offs_inner
        res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
        block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
        # Round the materialized combine result before normalization. This
        # matches the unfused combine -> RMSNorm boundary.
        out = (res.to(tl.float32) + block.to(tl.float32) * inj).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)
        out_f = out.to(tl.float32)
        sum_sq += tl.sum(out_f * out_f)

    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    for tile in tl.static_range(NUM_TILES):
        offs_inner = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_inner = offs_inner < HC_DIM
        offs = base + offs_inner
        # Shared norm weights repeat across streams; per-branch weights use the
        # same flattened HC layout as the residual.
        w_offs = offs_inner if W_SHARED else offs
        out = tl.load(out_ptr + row * stride_out + offs, mask_inner, other=0.0)
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
        y = out.to(tl.float32) * rrms
        y += y * w.to(tl.float32)
        tl.store(y_ptr + row * stride_y + offs, y, mask_inner)


@triton.jit
def _hc_combine_norm_row_kernel(
    res_ptr,
    block_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_res,
    stride_block,
    stride_inj,
    stride_out,
    stride_y,
    HC: tl.constexpr,
    GROUP_DIM: tl.constexpr,
    SPLIT_A: tl.constexpr,
    SPLIT_B: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    """``QWEN38_HC_CNROW``: round-218 row geometry for the fused combine+norm.

    One program owns one row and keeps all ``HC`` streams of it in a
    ``(HC, SPLIT)`` tile, exactly like ``_grouped_gemma_rmsnorm_row_kernel``;
    the combine that used to be a separate launch is computed in registers in
    front of the norm instead of being read back out of ``out_ptr``.

    Both rounding sites of the two-launch chain are reproduced:
      * the gate is rounded to the activation dtype before the Mac
        (``_hc_combine_mix_kernel_2d``'s contract against the eager
        ``addcmul``), and
      * the combine result is rounded to the activation dtype *before*
        ``x * x`` enters the sum of squares, which is what the unfused
        combine -> RMSNorm boundary does.
    ``GROUP_DIM`` must equal ``SPLIT_A + SPLIT_B`` (``HS = 2560`` -> 2048 +
    512), so neither leg needs a mask and the reduction is one axis-1
    ``tl.sum`` per leg.
    """
    row = tl.program_id(0)
    oa = tl.arange(0, SPLIT_A)
    ob = SPLIT_A + tl.arange(0, SPLIT_B)
    hoff = tl.arange(0, HC)[:, None] * GROUP_DIM
    ha = hoff + oa[None, :]
    hb = hoff + ob[None, :]
    row_r = row * stride_res
    row_o = row * stride_out
    row_y = row * stride_y

    j = tl.load(inj_ptr + row * stride_inj + tl.arange(0, HC)).to(tl.float32)
    g = (2.0 * tl.sigmoid(j / HC)).to(out_ptr.dtype.element_ty).to(tl.float32)

    ba = tl.load(block_ptr + row * stride_block + oa).to(tl.float32)
    bb = tl.load(block_ptr + row * stride_block + ob).to(tl.float32)
    xa = (
        tl.load(res_ptr + row_r + ha).to(tl.float32) + ba[None, :] * g[:, None]
    ).to(out_ptr.dtype.element_ty)
    xb = (
        tl.load(res_ptr + row_r + hb).to(tl.float32) + bb[None, :] * g[:, None]
    ).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row_o + ha, xa)
    tl.store(out_ptr + row_o + hb, xb)

    fa = xa.to(tl.float32)
    fb = xb.to(tl.float32)
    ss = tl.sum(fa * fa, axis=1) + tl.sum(fb * fb, axis=1)
    rrms = tl.rsqrt(ss / GROUP_DIM + EPS)[:, None]

    if W_SHARED:
        wa = tl.load(w_ptr + oa).to(tl.float32)
        wb = tl.load(w_ptr + ob).to(tl.float32)
    else:
        wa = tl.load(w_ptr + ha).to(tl.float32)
        wb = tl.load(w_ptr + hb).to(tl.float32)
    ya = fa * rrms
    ya += ya * wa
    yb = fb * rrms
    yb += yb * wb
    tl.store(y_ptr + row_y + ha, ya)
    tl.store(y_ptr + row_y + hb, yb)


@triton.jit
def _hc_gate_mix_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x,
    stride_g,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
) -> None:
    """``mean_hc(sigmoid(gate) * x)`` in one pass.

    ``x`` and ``gate`` are ``[M, HC * HC_DIM]`` with the HC axis in the middle,
    so the reduce is strided and the eager path has to materialise both the
    sigmoid and the product before reducing.

    One program owns exactly one ``BLOCK_SIZE`` tile of one row, so the live
    unified-buffer footprint is ``3 * BLOCK_SIZE`` fp32 values (~6 KB at
    ``BLOCK_SIZE=512``).  The earlier one-program-per-row form accumulated a
    ``(NUM_TILES_PAD, BLOCK_SIZE)`` 2-D tile (8 x 512 fp32 = 16 KB of
    accumulators plus two 16 KB operand tiles, both live across the unrolled
    ``HC`` loop).  That is the UB-overflow shape the in-tree RoPE kernel
    explicitly guards against on A2/A3
    (``ops/triton/rope.py``: "Large head_dim RoPE can overflow UB with the
    default tile"), and on ``QWEN38_HC_GATE_TRITON=1`` it reproduced as a
    device-side ``vector core timeout`` during startup.  Tiling the row across
    programs keeps the arithmetic byte-identical.
    """
    pid = tl.program_id(0)
    row = pid // NUM_TILES
    tile = pid % NUM_TILES
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < HC_DIM
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in tl.static_range(HC):
        x_k = tl.load(
            x_ptr + row * stride_x + k * HC_DIM + offs, mask, other=0.0
        ).to(tl.float32)
        g_k = tl.load(
            g_ptr + row * stride_g + k * HC_DIM + offs, mask, other=0.0
        ).to(tl.float32)
        acc += x_k * tl.sigmoid(g_k)
    acc = acc * (1.0 / HC)
    tl.store(
        y_ptr + row * stride_y + offs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _hc_gate_mix_kernel_static(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x: tl.constexpr,
    stride_g: tl.constexpr,
    stride_y: tl.constexpr,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
) -> None:
    """``_hc_gate_mix_kernel`` with the launch-invariant strides as constants.

    Body is copied verbatim; only the signature changed, so any numerical
    difference would be a compiler bug rather than a change of semantics.
    """
    pid = tl.program_id(0)
    row = pid // NUM_TILES
    tile = pid % NUM_TILES
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < HC_DIM
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in tl.static_range(HC):
        x_k = tl.load(
            x_ptr + row * stride_x + k * HC_DIM + offs, mask, other=0.0
        ).to(tl.float32)
        g_k = tl.load(
            g_ptr + row * stride_g + k * HC_DIM + offs, mask, other=0.0
        ).to(tl.float32)
        acc += x_k * tl.sigmoid(g_k)
    acc = acc * (1.0 / HC)
    tl.store(
        y_ptr + row * stride_y + offs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _hc_gate_mix_2d_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x,
    stride_g,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_TILES: tl.constexpr,
    ROWS: tl.constexpr,
) -> None:
    """``mean_hc(sigmoid(gate) * x)`` with one ``(HC, BLOCK)`` tile per load.

    Same arithmetic as ``_hc_gate_mix_kernel``, byte for byte -- ``acc`` is
    still an fp32 sum over the HC axis scaled by ``1 / HC`` -- but the HC walk
    becomes part of the block shape instead of a python-level loop over
    separate 1-D loads.  ``ROWS`` rows share the program prologue; the rows are
    independent, so the loop cannot reorder any single row's arithmetic.

    Registered only when ``QWEN38_HC_GATE2D`` is set, and it is only ever
    reached when ``x``/``gate`` are 2-D with unit inner stride, exactly like
    ``_hc_gate_mix_kernel``.  Because the tile spans the full HC axis it also
    requires ``HC_DIM`` to be a multiple of ``BLOCK``; the caller checks that
    before it picks the level.
    """
    pid = tl.program_id(0)
    row0 = (pid // NUM_TILES) * ROWS
    tile = pid % NUM_TILES
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    hoff = tl.arange(0, HC)[:, None] * HC_DIM + offs[None, :]
    for r in tl.static_range(ROWS):
        row = row0 + r
        x2 = tl.load(x_ptr + row * stride_x + hoff).to(tl.float32)
        g2 = tl.load(g_ptr + row * stride_g + hoff).to(tl.float32)
        acc = tl.sum(x2 * tl.sigmoid(g2), axis=0) * (1.0 / HC)
        tl.store(
            y_ptr + row * stride_y + offs,
            acc.to(y_ptr.dtype.element_ty),
        )


# Lazily materialized constants (fp32 affine / ones) keyed by the weight tensor
# identity and version so a reload invalidates them.
_CONST_CACHE: dict = {}


@triton.jit
def _hc_combine_mix_kernel(
    residual_ptr,
    block_ptr,
    inj_ptr,
    out_ptr,
    stride_r,
    stride_b,
    stride_i,
    stride_o,
    HC: tl.constexpr,
    HC_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
) -> None:
    """``addcmul(residual, block[:, None, :], 2*sigmoid(logits/HC))``, one launch.

    The layout is the same one ``_hc_gate_mix_kernel`` reads: ``residual`` is
    ``[M, HC * HC_DIM]`` with the HC axis in the middle, ``block`` is
    ``[M, HC_DIM]`` broadcast over HC, and ``inj`` is the ``[M, HC]`` logits.
    One program owns a ``BLOCK_SIZE`` tile of one row so the live unified-buffer
    footprint stays at two fp32 tiles -- the one-program-per-row shape is what
    died with ``vector core timeout`` on the gate kernel before it was tiled
    (see ``_hc_gate_mix_kernel``'s docstring).
    """
    pid = tl.program_id(0)
    row = pid // NUM_TILES
    tile = pid % NUM_TILES
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < HC_DIM
    bo = tl.load(block_ptr + row * stride_b + offs, mask, other=0.0).to(tl.float32)
    for k in tl.static_range(HC):
        j = tl.load(inj_ptr + row * stride_i + k).to(tl.float32)
        # The eager path materialises ``2*sigmoid(logits/HC)`` in the
        # activation dtype and hands *that* to addcmul, so the single rounding
        # point is reproduced here and the Mac stays in fp32.
        g = (2.0 * tl.sigmoid(j / HC)).to(out_ptr.dtype.element_ty).to(tl.float32)
        r = tl.load(
            residual_ptr + row * stride_r + k * HC_DIM + offs, mask, other=0.0
        ).to(tl.float32)
        tl.store(
            out_ptr + row * stride_o + k * HC_DIM + offs,
            (r + bo * g).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _hc_combine_mix_kernel_2d(
    residual_ptr,
    block_ptr,
    inj_ptr,
    out_ptr,
    stride_r,
    stride_b,
    stride_i,
    stride_o,
    HC: tl.constexpr,
    HC_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
) -> None:
    """``_hc_combine_mix_kernel`` with one ``(HC, BLOCK)`` tile per program.

    Same arithmetic, byte for byte: the gate is still rounded to the activation
    dtype before the Mac (that single rounding point is what ``torch.equal``
    checks against the eager ``addcmul``) and the Mac still happens in fp32
    against ``block`` broadcast over the HC axis.  What changes is only how the
    residual is read: one 2-D load covers all ``HC`` slices of a block instead
    of ``HC`` separate strided 1-D loads, which is the same edit that moved the
    gate kernel from 61.4 to 40.0 ns/row against a 40.3 ns/row traffic ceiling.

    ``HC_DIM`` must be a multiple of ``BLOCK_SIZE`` (``_combine_triton_ok``
    requires exactly that), so no mask is needed.
    """
    pid = tl.program_id(0)
    row = pid // NUM_TILES
    tile = pid % NUM_TILES
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hoff = tl.arange(0, HC)[:, None] * HC_DIM + offs[None, :]
    bo = tl.load(block_ptr + row * stride_b + offs).to(tl.float32)
    j = tl.load(inj_ptr + row * stride_i + tl.arange(0, HC)).to(tl.float32)
    g = (2.0 * tl.sigmoid(j / HC)).to(out_ptr.dtype.element_ty).to(tl.float32)
    r = tl.load(residual_ptr + row * stride_r + hoff).to(tl.float32)
    tl.store(
        out_ptr + row * stride_o + hoff,
        (r + bo[None, :] * g[:, None]).to(out_ptr.dtype.element_ty),
    )


def _combine_triton_ok(residual, block_output, injection_logits, hc_count) -> bool:
    """Every precondition the fused combine kernel needs, in one place.

    Returns False whenever the switch is off, which is what keeps the delivered
    arithmetic identical to the eager path on the default configuration.
    """
    if not _COMBINE_TRITON:
        return False
    if residual.dim() != 2 or residual.shape[-1] % hc_count:
        return False
    hidden = residual.shape[-1] // hc_count
    if hidden % 512:
        return False
    if block_output.shape[-1] != hidden:
        return False
    if injection_logits.shape[-1] != hc_count:
        return False
    if (
        residual.stride(1) != 1
        or block_output.stride(1) != 1
        or injection_logits.stride(1) != 1
    ):
        return False
    return residual.shape[0] == block_output.shape[0] == injection_logits.shape[0]


def _cn_row_ok(residual, block_output, injection_logits, norm_weight, hc_count):
    """Preconditions for the fused row-tile combine + grouped RMSNorm.

    Everything ``_combine_triton_ok`` already demands, plus the round-218
    geometry's own requirements: the per-branch stream must be exactly
    ``2048 + 512`` (the unmasked split legs), the stream count must be a power
    of two (``tl.arange``'s bound), and the affine must be one contiguous
    ``[HC, HS]`` or ``[HS]`` block.
    """
    if not _CNROW:
        return False
    if not _combine_triton_ok(
        residual, block_output, injection_logits, hc_count
    ):
        return False
    if hc_count & (hc_count - 1):
        return False
    hidden = residual.shape[-1] // hc_count
    if hidden != _NORM_SPLIT_A + _NORM_SPLIT_B:
        return False
    if not norm_weight.is_contiguous():
        return False
    if norm_weight.numel() not in (hidden, hc_count * hidden):
        return False
    return residual.dtype in (torch.bfloat16, torch.float16)


def _hc_combine_norm_row(
    residual, block_output, injection_logits, norm_weight, eps, hc_count
):
    """Launch the fused combine + grouped GemmaRMSNorm row kernel."""
    n, dim = residual.shape
    hidden = dim // hc_count
    combined = residual.new_empty(residual.shape)
    normed = residual.new_empty(residual.shape)
    _hc_combine_norm_row_kernel[(n,)](
        residual,
        block_output,
        injection_logits,
        norm_weight,
        combined,
        normed,
        residual.stride(0),
        block_output.stride(0),
        injection_logits.stride(0),
        combined.stride(0),
        normed.stride(0),
        hc_count,
        hidden,
        _NORM_SPLIT_A,
        _NORM_SPLIT_B,
        W_SHARED=norm_weight.numel() == hidden,
        EPS=eps,
    )
    return combined, normed


def _hc_combine_triton(residual, block_output, injection_logits, hc_count):
    """Launch the fused combine kernel and return the combined tensor.

    ``QWEN38_HC_COMBINE_TRITON=2`` selects the ``(HC, BLOCK)`` tile form, which
    the round-212 graph-replay probe prices at 19.71 us against a 22.06 us
    eager ``addcmul`` chain at the delivery shape (n=160, one tile set per row);
    ``=1`` keeps the original 1-D kernel because the ledger's -4.87 % verdict
    belongs to a kernel that is still shipped behind a switch.
    """
    n = residual.shape[0]
    hidden = residual.shape[-1] // hc_count
    num_tiles = hidden // 512
    out = residual.new_empty(residual.shape)
    kern = _hc_combine_mix_kernel_2d if _COMBINE_2D else _hc_combine_mix_kernel
    kern[(n * num_tiles,)](
        residual,
        block_output,
        injection_logits,
        out,
        residual.stride(0),
        block_output.stride(0),
        injection_logits.stride(0),
        out.stride(0),
        hc_count,
        hidden,
        512,
        num_tiles,
    )
    return out


def _one_plus_weight_f32(weight: torch.Tensor) -> torch.Tensor:
    """Cache ``1 + weight`` in fp32 (the Gemma affine)."""
    key = (id(weight), weight._version)
    entry = _CONST_CACHE.get(key)
    if entry is None or entry[0] is not weight:
        entry = (weight, (1.0 + weight.float()).contiguous())
        _CONST_CACHE[key] = entry
        if len(_CONST_CACHE) > 512:
            _CONST_CACHE.clear()
            _CONST_CACHE[key] = entry
    return entry[1]


def _one_plus_weight_dtype(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cache ``1 + weight`` *already materialised in* ``dtype``.

    ``_affine_view(...).to(dtype)`` allocates and launches an
    ``aclnnInplaceCopy_Cast`` on the ``[HC, HS]`` affine every decode step; the
    values are identical every call, so the cast is hoisted into the weight
    cache and the per-call cast disappears (bit-exact, no rounding change).
    """
    if dtype == torch.float32:
        return _one_plus_weight_f32(weight)
    key = (id(weight), weight._version, dtype)
    entry = _CONST_CACHE.get(key)
    if entry is None or entry[0] is not weight:
        entry = (weight, (1.0 + weight.float()).to(dtype).contiguous())
        _CONST_CACHE[key] = entry
        if len(_CONST_CACHE) > 512:
            _CONST_CACHE.clear()
            _CONST_CACHE[key] = entry
    return entry[1]


def _ones_last(dim: int, dtype: torch.dtype) -> torch.Tensor:
    key = ("ones", dim, dtype)
    entry = _CONST_CACHE.get(key)
    if entry is None:
        entry = torch.ones(dim, dtype=dtype, device="npu")
        _CONST_CACHE[key] = entry
    return entry


def _affine_view(weight: torch.Tensor, num_groups: int,
                 group_dim: int) -> torch.Tensor:
    gamma = _one_plus_weight_f32(weight)
    if weight.numel() == group_dim:
        return gamma.reshape(1, 1, group_dim)
    return gamma.reshape(num_groups, group_dim)


def _affine_view_dtype(weight: torch.Tensor, num_groups: int, group_dim: int,
                       dtype: torch.dtype) -> torch.Tensor:
    """``_affine_view`` without the per-call ``Cast`` into ``dtype``."""
    gamma = _one_plus_weight_dtype(weight, dtype)
    if weight.numel() == group_dim:
        return gamma.reshape(1, 1, group_dim)
    return gamma.reshape(num_groups, group_dim)


def _rms_norm_groups(x, weight, eps, num_groups):
    """Group RMSNorm for a *flattened* ``[..., num_groups * group_dim]`` x.

    The group axis is the last one, so ``group_dim == x.shape[-1] // num_groups``
    and a ``[num_groups, group_dim]`` affine applies. Callers holding an
    already-split ``[n, num_groups, group_dim]`` tensor must reduce it with
    ``npu_rms_norm(rows, ones, eps)`` + ``_affine_view`` themselves (see the
    LEAN branch of ``_hc_combine_norm``); passing such a tensor here would
    divide the *stream* length a second time and explode the reshape.
    """
    shape = x.shape
    group_dim = shape[-1] // num_groups
    flat = x.reshape(-1, shape[-1])
    if _USE_LEAN:
        if _NORM_TRITON and _norm_triton_ok(x, weight):
            # One launch replaces ``npu_rms_norm(rows, ones, eps)`` plus the
            # ``(1 + w)`` affine multiply -- same reduction axis, same fp32
            # accumulation, one activation-precision rounding instead of two.
            return _grouped_gemma_rmsnorm_triton(x, weight, eps, num_groups)
        # Rows are the per-stream vectors (length ``group_dim``). When the
        # checkpoint shares one affine across the HC streams it can be passed
        # straight to the fused op, which is bit-level equivalent to the fp32
        # ``normalized * (1 + weight)`` path but without any Cast kernel.
        rows = flat.reshape(-1, group_dim)
        if weight.numel() == group_dim:
            return torch_npu.npu_rms_norm(
                rows, _one_plus_weight_f32(weight), eps
            )[0].reshape(shape)
        if _USE_GEMMA_NORM:
            return torch_npu.npu_gemma_rms_norm(
                flat.reshape(-1, num_groups, group_dim),
                weight.reshape(num_groups, group_dim),
                eps,
            )[0].reshape(shape)
        normed = torch_npu.npu_rms_norm(
            rows, _ones_last(group_dim, torch.float32), eps
        )[0].reshape(-1, num_groups, group_dim)
        affine = _affine_view_dtype(weight, num_groups, group_dim, normed.dtype)
        return (normed * affine).reshape(shape)
    rows = flat.float().reshape(-1, num_groups, group_dim)
    normed = torch_npu.npu_rms_norm(
        rows.reshape(-1, group_dim), _ones_last(group_dim, torch.float32), eps
    )[0].reshape(rows.shape)
    return (normed * _affine_view(weight, num_groups, group_dim)).reshape(
        shape
    ).to(x.dtype)


def _grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    if x.shape[-1] % num_groups:
        raise ValueError("Grouped RMSNorm input is not divisible by num_groups")
    group_dim = x.shape[-1] // num_groups
    if weight.numel() not in (group_dim, x.shape[-1]):
        raise ValueError("Grouped RMSNorm weight has an invalid size")
    if (
        _USE_TRITON
        and x.dim() == 2
        and x.stride(1) == 1
        and weight.is_contiguous()
    ):
        n, dim = x.shape
        y = x.new_empty(x.shape)
        _grouped_gemma_rmsnorm_kernel[(n * num_groups,)](
            x,
            weight,
            y,
            x.stride(0),
            y.stride(0),
            group_dim,
            num_groups,
            triton.next_power_of_2(group_dim),
            W_SHARED=weight.numel() == group_dim,
            EPS=eps,
        )
        return y
    if _USE_FUSED:
        return _rms_norm_groups(x, weight, eps, num_groups)
    grouped = x.reshape(-1, num_groups, group_dim).float()
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    affine = weight.float().reshape(1, -1, group_dim)
    if weight.numel() == group_dim:
        affine = affine[:, :1]
    return (normalized * (1.0 + affine)).reshape_as(x).to(x.dtype)


@triton.jit
def _hc_silu_div_kernel(
    x_ptr,
    out_ptr,
    n_elem,
    n_col,
    stride_x,
    stride_o,
    hc_f,
    BLOCK: tl.constexpr,
) -> None:
    """``silu(x[:, :n_col] / hc_count)`` for a row-strided ``x``, one launch.

    Row-strided rather than flat because the call site passes ``lora`` as a
    view of the merged down-projection ``[rows, 336]``: element (r, c) lives at
    ``r * stride_x + c``.  Walking that stride absorbs the separate
    ``aclnnDivs_Slice`` launch eager pays before the division, and the same
    kernel still serves the plain contiguous call site (there
    ``stride_x == n_col``).

    The division rounds to the activation dtype *before* the swish, because
    that is where eager rounds it (``x / hc_count`` materialises a bf16 tensor
    that ``aclnnSilu`` then reads).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    row = offs // n_col
    col = offs % n_col
    x = tl.load(x_ptr + row * stride_x + col, mask=mask, other=0.0).to(tl.float32)
    t = (x / hc_f).to(out_ptr.dtype.element_ty)
    t32 = t.to(tl.float32)
    tl.store(
        out_ptr + row * stride_o + col,
        (t32 * tl.sigmoid(t32)).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _silu_triton_ok(x: torch.Tensor, hc_count: int) -> bool:
    """Every precondition the fused silu kernel needs, in one place.

    False whenever the switch is off, which is what keeps the default
    configuration bit-identical to the eager path.

    ``stride(1) == 1`` rather than ``is_contiguous()``: the call site hands in
    a slice view of the merged down-projection (row stride 336, width 320) and
    the kernel walks that stride.  Demanding full contiguity here is what made
    the first version of this patch a silent no-op on 96 % of the call sites.
    """
    if not _SILU_TRITON or hc_count <= 0:
        return False
    if x.dim() != 2 or x.numel() == 0:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    return x.stride(1) == 1


def _hc_silu_triton(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Launch ``_hc_silu_div_kernel`` and return ``silu(x / hc_count)``."""
    out = x.new_empty(x.shape)
    n = x.numel()
    n_col = x.shape[1]
    _hc_silu_div_kernel[(triton.cdiv(n, _SILU_BLOCK),)](
        x,
        out,
        n,
        n_col,
        x.stride(0),
        out.stride(0),
        float(hc_count),
        _SILU_BLOCK,
    )
    return out


@triton.jit
def _hc_silu_div_kernel_2d(
    x_ptr,
    out_ptr,
    n_row,
    n_col,
    stride_x,
    stride_o,
    hc_f,
    N_COL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    """``silu(x[:, :N_COL] / hc_count)`` with no index arithmetic.

    The v2 kernel walked a flat offset and recovered the row as ``offs //
    n_col``.  ``n_col`` is 320 -- a runtime scalar, not a power of two -- so
    the vector unit evaluated an integer divide and a modulo for every element
    and the kernel cost 616 us/call where eager costs 10.8 (measured, see the
    module docstring).  Here ``program_id(0)`` *is* the row block and
    ``program_id(1)`` *is* the column block, so both indices are affine in the
    lane index and nothing divides.

    The width is ``constexpr`` because it is a property of the model
    (``lora_rank``), and the mask against it is then a compare against a
    constant.  Rows stay a runtime value: the call site's row count is the
    batch, and it changes every step.

    The arithmetic is unchanged from v2 and from eager: divide in fp32, round
    to the activation dtype (that is where eager rounds it -- ``x / hc_count``
    materialises a bf16 tensor that ``aclnnSilu`` then reads), then swish in
    fp32 and round once.  Bit-identical to ``F.silu(x / hc_count)`` on every
    case ``hc_silu_probe3.py`` tests.
    """
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    rows = pr * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    m = (rows[:, None] < n_row) & (cols[None, :] < n_col)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x + cols[None, :], mask=m, other=0.0
    ).to(tl.float32)
    t = (x / hc_f).to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(
        out_ptr + rows[:, None] * stride_o + cols[None, :],
        (t * tl.sigmoid(t)).to(out_ptr.dtype.element_ty),
        mask=m,
    )


def _hc_silu_triton_v3(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Launch ``_hc_silu_div_kernel_2d`` and return ``silu(x / hc_count)``."""
    out = x.new_empty(x.shape)
    n_row, n_col = x.shape
    grid = (triton.cdiv(n_row, _SILU_BLOCK_R), triton.cdiv(n_col, _SILU_BLOCK_C))
    _hc_silu_div_kernel_2d[grid](
        x,
        out,
        n_row,
        n_col,
        x.stride(0),
        out.stride(0),
        float(hc_count),
        n_col,
        _SILU_BLOCK_R,
        _SILU_BLOCK_C,
    )
    return out


def _hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    if hc_count <= 0:
        raise ValueError("hc_count must be positive")
    if _silu_triton_ok(x, hc_count):
        if _SILU_TRITON_V3:
            return _hc_silu_triton_v3(x, hc_count)
        return _hc_silu_triton(x, hc_count)
    if _LEAN_SILU:
        return F.silu(x / hc_count)
    return F.silu(x.float() / hc_count).to(x.dtype)


def _hc_gate_mix(
    x: torch.Tensor, gate: torch.Tensor, hc_count: int
) -> torch.Tensor:
    if x.shape != gate.shape or x.shape[-1] % hc_count:
        raise ValueError("HC gate and input shapes are incompatible")
    hidden_size = x.shape[-1] // hc_count
    if _GATE2D:
        block, rows = _GATE2D_TABLE[_GATE2D_ENV]
        if (
            _LEAN_GATE
            and x.dim() == 2
            and x.stride(1) == 1
            and gate.stride(1) == 1
            and hidden_size % block == 0
        ):
            n = x.shape[0]
            if rows > 1 and n % rows:
                rows = 1
            num_tiles = hidden_size // block
            y = x.new_empty((n, hidden_size))
            _hc_gate_mix_2d_kernel[((n // rows) * num_tiles,)](
                x,
                gate,
                y,
                x.stride(0),
                gate.stride(0),
                y.stride(0),
                hidden_size,
                hc_count,
                block,
                num_tiles,
                rows,
            )
            return y
    if (
        _GATE_TRITON
        and _LEAN_GATE
        and x.dim() == 2
        and x.stride(1) == 1
        and gate.stride(1) == 1
        and hidden_size % 512 == 0
    ):
        n = x.shape[0]
        num_tiles = hidden_size // 512
        y = x.new_empty((n, hidden_size))
        kern = (
            _hc_gate_mix_kernel_static
            if _GATE_TRITON_STATIC
            else _hc_gate_mix_kernel
        )
        kern[(n * num_tiles,)](
            x,
            gate,
            y,
            x.stride(0),
            gate.stride(0),
            y.stride(0),
            hidden_size,
            hc_count,
            512,
            num_tiles,
        )
        return y
    if _LEAN_GATE:
        mixed = torch.sigmoid(gate).mul(x)
        return mixed.reshape(-1, hc_count, hidden_size).mean(1).to(x.dtype)
    mixed = torch.sigmoid(gate.float()) * x.float()
    return mixed.reshape(-1, hc_count, hidden_size).mean(1).to(x.dtype)


def _hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if residual.shape[-1] % hc_count:
        raise ValueError("HC residual is not divisible by hc_count")
    hidden_size = residual.shape[-1] // hc_count
    if block_output.shape[-1] != hidden_size:
        raise ValueError("HC block output has an invalid size")
    if _combine_triton_ok(residual, block_output, injection_logits, hc_count):
        return _hc_combine_triton(
            residual, block_output, injection_logits, hc_count
        ).reshape_as(residual)
    if _LEAN_COMBINE:
        n = residual.shape[0]
        injection = _injection_gate(
            injection_logits, hc_count, residual.dtype
        ).reshape(n, hc_count, 1)
        combined = torch.addcmul(
            residual.reshape(n, hc_count, hidden_size),
            block_output.reshape(n, 1, hidden_size),
            injection,
        )
        return combined.reshape_as(residual)
    if _USE_FUSED:
        n = residual.shape[0]
        injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
        combined = torch.addcmul(
            residual.float().reshape(n, hc_count, hidden_size),
            block_output.float().reshape(n, 1, hidden_size),
            injection.reshape(n, hc_count, 1),
        )
        return combined.reshape_as(residual).to(residual.dtype)
    injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
    combined = residual.float().reshape(-1, hc_count, hidden_size)
    combined = combined + block_output.float().unsqueeze(1) * injection.unsqueeze(-1)
    return combined.reshape_as(residual).to(residual.dtype)


def _hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        _USE_TRITON
        and residual.dim() == 2
        and residual.stride(1) == 1
        and block_output.stride(1) == 1
        and injection_logits.stride(1) == 1
        and norm_weight.is_contiguous()
        and residual.shape[-1] % hc_count == 0
        and block_output.shape[-1] == residual.shape[-1] // hc_count
    ):
        n, dim = residual.shape
        hc_dim = dim // hc_count
        out = residual.new_empty(residual.shape)
        y = residual.new_empty(residual.shape)
        _hc_combine_norm_kernel[(n * hc_count,)](
            block_output,
            residual,
            injection_logits,
            norm_weight,
            out,
            y,
            block_output.stride(0),
            residual.stride(0),
            injection_logits.stride(0),
            out.stride(0),
            y.stride(0),
            hc_dim,
            hc_count,
            W_SHARED=norm_weight.numel() == hc_dim,
            EPS=eps,
            BLOCK_SIZE=512,
            NUM_TILES=triton.cdiv(hc_dim, 512),
        )
        return out, y
    if _USE_LEAN:
        shape = residual.shape
        hidden_size = shape[-1] // hc_count
        n = shape[0]
        if _cn_row_ok(
            residual, block_output, injection_logits, norm_weight, hc_count
        ):
            # Delivery path for ``QWEN38_HC_CNROW``: one launch produces both
            # the rounded combine (which the caller keeps as the multi-stream
            # state) and its grouped GemmaRMSNorm.  The combine is not written
            # and read back between the two halves.
            return _hc_combine_norm_row(
                residual,
                block_output,
                injection_logits,
                norm_weight,
                eps,
                hc_count,
            )
        if _combine_triton_ok(
            residual, block_output, injection_logits, hc_count
        ):
            # Delivery path: combine in one launch, then hand the result to the
            # same ``npu_rms_norm`` the eager path uses.  Only the combine glue
            # is replaced -- the tuned Ascend C normalisation kernel stays,
            # which is the difference between this and ``HC_FUSED=6`` (that one
            # fused the norm too, into a Triton kernel, and lost 6.3 %).
            combined = _hc_combine_triton(
                residual, block_output, injection_logits, hc_count
            )
        elif _LEAN_COMBINE:
            injection = _injection_gate(
                injection_logits, hc_count, residual.dtype
            ).reshape(n, hc_count, 1)
            combined = torch.addcmul(
                residual.reshape(n, hc_count, hidden_size),
                block_output.reshape(n, 1, hidden_size),
                injection,
            )
        else:
            injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
            combined = torch.addcmul(
                residual.float().reshape(n, hc_count, hidden_size),
                block_output.float().reshape(n, 1, hidden_size),
                injection.reshape(n, hc_count, 1),
            ).to(residual.dtype)
        # ``combined`` is already split as ``[n, HC, HS]``, so the norm runs on
        # ``(n*HC)`` rows of ``hidden_size`` while the affine still groups by
        # HC: a per-branch checkpoint weight is ``[HC, HS]``, not ``[1, HS]``.
        # Handing this to ``_rms_norm_groups`` (which re-derives the group
        # length from the last dim) would double-divide the stream length.
        if _NORM_TRITON and _norm_triton_ok(combined, norm_weight):
            # One launch replaces ``npu_rms_norm(rows, ones, eps)`` plus the
            # ``(1 + w)`` affine multiply.  ``combined`` is the contiguous
            # ``[n, HC, HS]`` block; flattening it to ``[n, HC * HS]`` gives the
            # grouped kernel exactly the rows (length ``HS``) and the group axis
            # (``HC``) that the eager pair normalises and scales.
            normed = _grouped_gemma_rmsnorm_triton(
                combined.reshape(n, hc_count * hidden_size),
                norm_weight,
                eps,
                hc_count,
            ).reshape(shape)
        elif _USE_GEMMA_NORM and norm_weight.numel() == hc_count * hidden_size:
            # Mode 7: one ``npu_gemma_rms_norm`` replaces the ``npu_rms_norm``
            # plus the ``[HC, HS]`` affine multiply.  Restricted to the
            # per-branch checkpoint weight, because the op needs ``gamma`` to
            # have exactly the trailing two axes of the operand.
            normed = torch_npu.npu_gemma_rms_norm(
                combined.reshape(n, hc_count, hidden_size),
                norm_weight.reshape(hc_count, hidden_size),
                eps,
            )[0].reshape(shape)
        elif _AFFINE_RMS and norm_weight.numel() == hc_count * hidden_size:
            # Per-branch affine, applied by the op: the operand stays 3-D so
            # the ``[HC, HS]`` gamma broadcasts along the branch axis.
            normed = torch_npu.npu_rms_norm(
                combined.reshape(n, hc_count, hidden_size),
                _affine_view_dtype(
                    norm_weight, hc_count, hidden_size, combined.dtype
                ),
                eps,
            )[0].reshape(shape)
        else:
            rows = combined.reshape(n * hc_count, hidden_size)
            if _HC_NORM_W and norm_weight.numel() == hidden_size:
                # ``rms_norm(rows, ones) * (1 + w)`` with the affine applied by
                # the op: one fewer kernel, identical maths.  The fp32 gamma is
                # the same operand ``_rms_norm_groups`` already hands the op in
                # its ``weight.numel() == group_dim`` branch.
                normed = torch_npu.npu_rms_norm(
                    rows, _one_plus_weight_f32(norm_weight), eps
                )[0].reshape(n, hc_count, hidden_size)
            else:
                normed = torch_npu.npu_rms_norm(
                    rows, _ones_last(hidden_size, torch.float32), eps
                )[0].reshape(n, hc_count, hidden_size)
                normed = normed * _affine_view_dtype(
                    norm_weight, hc_count, hidden_size, normed.dtype
                )
            normed = normed.reshape(shape)
        return combined.reshape(shape), normed
    if _USE_FUSED:
        shape = residual.shape
        hidden_size = shape[-1] // hc_count
        n = shape[0]
        injection = (2.0 * torch.sigmoid(injection_logits.float() / hc_count))
        combined = torch.addcmul(
            residual.float().reshape(n, hc_count, hidden_size),
            block_output.float().reshape(n, 1, hidden_size),
            injection.reshape(n, hc_count, 1),
        )
        normed = torch_npu.npu_rms_norm(
            combined.reshape(n * hc_count, hidden_size),
            _ones_last(hidden_size, torch.float32),
            eps,
        )[0].reshape(n, hc_count, hidden_size)
        normed = normed * _affine_view(
            norm_weight, hc_count, hidden_size
        )
        return (
            combined.reshape(shape).to(residual.dtype),
            normed.reshape(shape).to(residual.dtype),
        )
    combined = _hc_combine(
        residual, block_output, injection_logits, hc_count
    )
    normalized = _grouped_gemma_rmsnorm(
        combined, norm_weight, eps, hc_count
    )
    return combined, normalized


def _same_shape_fake(x: torch.Tensor, *args) -> torch.Tensor:
    del args
    return x.new_empty(x.shape)


def _hc_gate_mix_fake(
    x: torch.Tensor, gate: torch.Tensor, hc_count: int
) -> torch.Tensor:
    del gate
    return x.new_empty((x.shape[0], x.shape[1] // hc_count))


def _hc_combine_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del block_output, injection_logits, hc_count
    return residual.new_empty(residual.shape)


def _hc_combine_norm_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    return residual.new_empty(residual.shape), residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="qwen4_exp_grouped_gemma_rmsnorm",
    op_func=_grouped_gemma_rmsnorm,
    fake_impl=_same_shape_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_silu",
    op_func=_hc_silu,
    fake_impl=_same_shape_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_gate_mix",
    op_func=_hc_gate_mix,
    fake_impl=_hc_gate_mix_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine",
    op_func=_hc_combine,
    fake_impl=_hc_combine_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine_norm",
    op_func=_hc_combine_norm,
    fake_impl=_hc_combine_norm_fake,
)


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_grouped_gemma_rmsnorm(
        x, weight, eps, num_groups
    )


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_silu(x, hc_count)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, hc_count)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_combine(
        residual, block_output, injection_logits, hc_count
    )


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.qwen4_exp_hc_combine_norm(
        residual,
        block_output,
        injection_logits,
        norm_weight,
        eps,
        hc_count,
    )


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
]
