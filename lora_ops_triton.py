"""Triton bgmv/sgmv ops exposed as torch custom ops (torch.library).

The kernels are registered in the ``vllm_ascend_triton`` namespace so that
torch._dynamo treats them as opaque, allowed-in-graph nodes -- the same
mechanism the stock ``torch.ops._C_ascend.*`` ops use.  All runtime checks
(dtype / NR / token-count consistency) and the AscendC fallbacks live inside
the eager impls, which execute outside the traced graph (during aclgraph
capture recording the impls run eagerly and their kernel launches get
recorded into the graph).

Kernel launches go through a C++ launcher (rtKernelLaunch on a flat-packed
arg buffer, ~14us/launch instead of the triton python path's ~66us+):
- per-(kernel, constexpr, dtype) case: compile once via warmup, register
  once, verify-retry warmup until launches actually land.
- CANN quirk: the first launches of a freshly registered binary are silently
  dropped until the device-side load settles (~tens of ms); the verify-retry
  loop launches on a dummy output until it changes.  The syncs in that loop
  are only on case creation -- vllm warmup populates all cases before
  aclgraph capture, so no sync happens during capture.  If that assumption
  ever breaks (EE1016 during capture), set TRITON_LORA_CPP_VERIFY=0.
- TRITON_LORA_CPP=0 restores the plain triton launch path.
- Indices are converted to int32 in the wrappers (same as the plain triton
  path; int64 indices are a separate follow-up optimization).

The module also keeps plain-Triton wrappers (``bgmv_shrink``, ...) that the
custom op impls call and that test_compare.py uses directly.
"""
import ctypes
import os
import struct
import time

import torch
from torch.library import custom_op, register_fake

from vllm_ascend.lora import lora_ops_triton_kernels as K

_announced = False
_NR_MAX = 16  # prefix-sum request mapping is O(NR^2); beyond this use AscendC

_cast_scratch: dict = {}

_TIMING = os.environ.get("TRITON_LORA_TIME", "") != ""
_TIMER = {}  # name -> [count, total_s]


def _timing_start(name):
    return time.perf_counter() if _TIMING else None


def _timing_end(name, t0):
    if t0 is None:
        return
    c, s = _TIMER.get(name, (0, 0.0))
    total = s + time.perf_counter() - t0
    _TIMER[name] = (c + 1, total)
    if (c + 1) % 200 == 0:
        print(f"[triton-lora] TIMING {name}: {total * 1e6 / (c + 1):.1f} "
              f"us/call ({c + 1} calls)", flush=True)


def _to_int32(t: torch.Tensor, tag: str = "") -> torch.Tensor:
    """int64 -> int32 via a reused scratch buffer.

    Decode calls this 2x per op (indices + seq lens); allocating a fresh
    int32 tensor per call costs ~160us through the NPUCachingAllocator's
    32-padding path (per-call microbench: 4.8ms/step of the 5.5ms total
    triton wrapper cost), while the cast kernel itself is ~20us.  Reusing a
    per-numel scratch keeps only the cast kernel.  Single-stream ordering
    makes sharing safe across calls; ``tag`` separates index vs seq scratch
    when their numels collide (both are NR).
    """
    # v2: identity.  Every kernel already narrows on load
    # (`tl.load(indices + ...).to(tl.int32)`), so int64 tensors are fine as-is.
    # The cast used to launch two extra device kernels per op call -- which
    # aclgraph captures and then replays on EVERY decode step -- and it froze a
    # _cast_scratch pointer into the captured graph.  Set TRITON_LORA_CAST=1 to
    # restore the old behaviour.
    if os.environ.get("TRITON_LORA_CAST", "0") == "0":
        return t
    key = (tag, t.numel())
    s = _cast_scratch.get(key)
    if s is None or s.device != t.device:
        s = torch.empty(t.numel(), dtype=torch.int32, device=t.device)
        _cast_scratch[key] = s
    return s.copy_(t)


def _announce():
    global _announced
    if not _announced:
        mode = ("C++ rtKernelLaunch" if _cpp_enabled() else "triton python path")
        if _TIMING:
            mode += " + TIMING"
        print(f"[triton-lora] dispatch ACTIVE ({mode})", flush=True)
        _announced = True


def _triton_dtype_ok(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float16, torch.bfloat16)


def _ascend_bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling):
    torch.ops._C_ascend.bgmv_shrink(inputs, lora_a_weights, lora_indices_tensor, output_tensor, scaling)


def _ascend_bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                        slice_offset, slice_size):
    torch.ops._C_ascend.bgmv_expand(inputs, lora_b_weights, lora_indices_tensor,
                                    output_tensor, slice_offset, slice_size)


def _ascend_sgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor,
                        seq_len_tensor, scaling):
    torch.ops._C_ascend.sgmv_shrink(inputs, lora_a_weights, lora_indices_tensor,
                                    seq_len_tensor, output_tensor, scaling)


def _ascend_sgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                        seq_len_tensor, slice_offset, slice_size):
    torch.ops._C_ascend.sgmv_expand(inputs, lora_b_weights, lora_indices_tensor,
                                    seq_len_tensor, output_tensor, slice_offset, slice_size)


def _sgmv_triton_ok(seq_len_tensor: torch.Tensor) -> bool:
    # NOTE: no device syncs allowed here -- this runs during aclgraph capture
    # (EE1016: synchronizing a captured stream is not supported).  The kernel
    # maps token rows via prefix sum and skips rows beyond the total, so a
    # sum-vs-batches check is not needed.
    return int(seq_len_tensor.numel()) <= _NR_MAX


# ---- C++ launcher (rtKernelLaunch direct) ----

_CPP_DIR = os.path.dirname(os.path.abspath(__file__))
_CPP_SRC = os.path.join(_CPP_DIR, "lora_cpp_launcher.cpp")
_CPP_SO = os.path.join(_CPP_DIR, "lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so")

_CPP_STATE = None          # (CDLL, ffts_addr) or (None, None) on failure
_CPP_CASES = {}            # key -> case dict
_CPP_FAILED = set()        # keys that must use the triton fallback


# ---- native kPrivateUse1 impls (DECOMMISSIONED for serve, opt-in only) ----
#
# lora_native_ops.cpp registered the same schemas on the PrivateUse1 and
# Python dispatch keys so torch_npu's aclgraph would capture their kernel
# launches with no python on the step path.  It does NOT work in serve:
# aclgraph execution dispatches these ops through the python dispatcher with
# wrapper tensors, and C++ kernels receive them raw -- trace_only() bailed on
# every call (dbg log: only "trace_only pyobj", zero real launches), so the
# captured graph contained no-op nodes and LoRA was silently NEVER applied
# (probe 09-02: openscad outputs byte-identical to the base model).
# Python-key PYTHON impls are the only form that reaches real NPU tensors in
# serve (the python dispatcher unwraps wrapper tensors for python functions).
# They launch through the ctypes fast path (~14us), so native mode below is
# now opt-in (TRITON_LORA_NATIVE=1) for standalone debugging only.

_NATIVE_SO = os.path.join(_CPP_DIR, "lora_native_ops.so")
_NATIVE_SRC = os.path.join(_CPP_DIR, "lora_native_ops.cpp")
_NATIVE = None  # ctypes.CDLL of lora_native_ops.so, or None on failure

_DT_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int32": torch.int32,
    "torch.float32": torch.float32,
}

_KEY_KERNELS = {
    "bgmv_shrink": K.bgmv_shrink,
    "bgmv_expand": K.bgmv_expand,
    "sgmv_shrink_kernel": K.sgmv_shrink_kernel,
    "sgmv_expand": K.sgmv_expand,
}


def _native_build():
    """g++ the native impls .so (links libtorch + CANN runtime)."""
    if os.path.exists(_NATIVE_SO):
        return _NATIVE_SO
    import subprocess
    import sysconfig
    from torch.utils import cpp_extension as ce
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME") or "/usr/local/Ascend/ascend-toolkit/latest"
    cann_inc = os.path.join(cann, "aarch64-linux", "pkg_inc")
    cann_lib = os.path.join(cann, "aarch64-linux", "lib64")
    if not os.path.isdir(cann_inc):  # older layout: <cann>/include + lib64
        cann_inc, cann_lib = os.path.join(cann, "include"), os.path.join(cann, "lib64")
    flags = ["g++", "-std=c++17", "-fPIC", "-O2", "-shared"]
    for i in ce.include_paths(device_type=None) + [
            sysconfig.get_paths()["include"], cann_inc]:
        flags += ["-I", i]
    torch_lib = ce.library_paths()
    for l in torch_lib + [cann_lib]:
        flags += ["-L", l, "-Wl,-rpath," + l]
    flags += ["-o", _NATIVE_SO, _NATIVE_SRC, "-ltorch", "-lc10",
              "-lruntime", "-lascendcl"]
    r = subprocess.run(flags, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("g++ failed:\n" + r.stderr[-3000:])
    return _NATIVE_SO


def _cpp_stream_getter():
    return int(torch.npu.current_stream().npu_stream)


def _cpp_parse_key(key_s):
    name, kvs_s, dts_s = key_s.split("|")
    kvs = {}
    for kv in kvs_s.split(","):
        k, v = kv.split("=", 1)
        kvs[k] = float(v) if "." in v else int(v)
    return name, kvs, dts_s.split(",")


def _cpp_dummy_args(name, kwargs, dts):
    """Dummy example args for case compilation (shapes only need to be
    consistent with the constexprs; B is the grid, fixed at 8).  Indices
    come first-call-shape-free; the C++ key carries everything else."""
    i32 = torch.int32
    if name == "bgmv_shrink":
        H, R, L = kwargs["H"], kwargs["R"], kwargs["L"]
        x = torch.empty(8, H, dtype=_DT_MAP[dts[0]], device="npu")
        w = torch.empty(L, R, H, dtype=_DT_MAP[dts[1]], device="npu")
        idx = torch.zeros(8, dtype=i32, device="npu")
        y = torch.zeros(8, R, dtype=_DT_MAP[dts[3]], device="npu")
        return x, w, idx, y, 1.0
    if name == "bgmv_expand":
        R, Ho, L = kwargs["R"], kwargs["Ho"], kwargs["L"]
        ydt = _DT_MAP[dts[3]]
        x = torch.empty(8, R, dtype=_DT_MAP[dts[0]], device="npu")
        w = torch.empty(L, Ho, R, dtype=_DT_MAP[dts[1]], device="npu")
        idx = torch.zeros(8, dtype=i32, device="npu")
        y = torch.zeros(8, kwargs["Y_HO"], dtype=ydt, device="npu")
        return x, w, idx, y, y
    if name == "sgmv_shrink_kernel":
        H, R, L, NR = kwargs["H"], kwargs["R"], kwargs["L"], kwargs["NR"]
        x = torch.empty(8, H, dtype=_DT_MAP[dts[0]], device="npu")
        w = torch.empty(L, R, H, dtype=_DT_MAP[dts[1]], device="npu")
        idx = torch.zeros(NR, dtype=i32, device="npu")
        # token counts must sum to B (8) so every row maps to a request and
        # the verify-retry sees a nonzero write
        seq = torch.full((NR,), 8 // NR, dtype=i32, device="npu")
        y = torch.zeros(8, R, dtype=_DT_MAP[dts[4]], device="npu")
        return x, w, idx, seq, y
    # sgmv_expand
    R, Ho, L, NR = kwargs["R"], kwargs["Ho"], kwargs["L"], kwargs["NR"]
    ydt = _DT_MAP[dts[4]]
    x = torch.empty(8, R, dtype=_DT_MAP[dts[0]], device="npu")
    w = torch.empty(L, Ho, R, dtype=_DT_MAP[dts[1]], device="npu")
    idx = torch.zeros(NR, dtype=i32, device="npu")
    seq = torch.full((NR,), 8 // NR, dtype=i32, device="npu")
    y = torch.zeros(8, kwargs["Y_HO"], dtype=ydt, device="npu")
    return x, w, idx, seq, y, y


def _cpp_miss_handler(key_s):
    """Called by the native C++ impls when a case key is not yet bound.
    Compiles + verifies the case and binds key -> rtFunction handle.
    Returns False when the case cannot be built here (C++ impl then falls
    back to the AscendC native op so results stay correct)."""
    try:
        if torch.npu.is_current_stream_capturing():
            return False  # no compile/sync during capture
        name, kwargs, dts = _cpp_parse_key(key_s)
        kernel_fn = _KEY_KERNELS[name]
        args = _cpp_dummy_args(name, kwargs, dts)
        case = _CPP_CASES.get(key_s)
        if case is None:
            case = _cpp_make_case(kernel_fn, kwargs, args, (8,))
            if case is None:
                _CPP_FAILED.add(key_s)
                return False
            _CPP_CASES[key_s] = case
        _NATIVE.lora_native_bind_case(key_s.encode(), case["func"])
        return True
    except Exception as e:
        print(f"[triton-lora] native miss-handler failed {key_s}: {e}",
              flush=True)
        _CPP_FAILED.add(key_s)
        return False


def _native_setup():
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE
    try:
        so = _native_build()
        CL = ctypes.CDLL(so)
        CL.lora_native_bind_case.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        CL.lora_native_set_handler.argtypes = [ctypes.py_object]
        CL.lora_native_set_stream_getter.argtypes = [ctypes.py_object]
        CL.lora_native_set_handler(_cpp_miss_handler)
        CL.lora_native_set_stream_getter(_cpp_stream_getter)
        _NATIVE = CL
        print("[triton-lora] native kPrivateUse1 impls ACTIVE", flush=True)
    except Exception as e:
        print(f"[triton-lora] native impls unavailable: {e}", flush=True)
        _NATIVE = None
    return _NATIVE


def _cpp_enabled() -> bool:
    return os.environ.get("TRITON_LORA_CPP", "1") != "0"


def _cpp_setup():
    global _CPP_STATE
    if _CPP_STATE is not None:
        return _CPP_STATE
    try:
        so = _CPP_SO
        if not os.path.exists(so):
            from triton.backends.ascend.utils import _build_npu_ext
            so = _build_npu_ext("lora_cpp_launcher", _CPP_SRC,
                                kernel_launcher="torch")
        CL = ctypes.CDLL(so)
        CL.lora_register_kernel.restype = ctypes.c_uint64
        CL.lora_register_kernel.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                            ctypes.c_uint64, ctypes.c_char_p,
                                            ctypes.c_int]
        CL.lora_launch_flat.restype = ctypes.c_uint64
        CL.lora_launch_flat.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                                        ctypes.c_int32, ctypes.c_void_p,
                                        ctypes.c_uint64]
        CL.lora_get_ffts_addr.restype = ctypes.c_uint64
        CL.lora_get_ffts_addr.argtypes = [ctypes.c_int]
        CL.lora_peek_stub.restype = ctypes.c_uint64
        CL.lora_peek_stub.argtypes = [ctypes.c_uint64]
        _CPP_STATE = (CL, CL.lora_get_ffts_addr(torch.npu.current_device()))
    except Exception as e:
        print(f"[triton-lora] C++ launcher unavailable, triton fallback: {e}",
              flush=True)
        _CPP_STATE = (None, None)
    return _CPP_STATE


def _cpp_case_key(kernel_fn, kwargs, tensors):
    # v2: two fixes.  (a) `tensors` is really example_args and may hold python
    # scalars -- bgmv_shrink passes `scaling` -- so .dtype must be guarded, or
    # every bgmv_shrink call raises AttributeError.  (b) sorted() + str() per
    # kwarg on each of ~1200 op calls per forward is pure host tax; kwargs is
    # built with a fixed key order at every call site, so the values alone
    # identify the case.
    return (kernel_fn.__name__,
            tuple(kwargs.values()),
            tuple(t.dtype for t in tensors if isinstance(t, torch.Tensor)))


_NUM_AIV = None


def _num_aiv() -> int:
    """Physical AIV block count.  triton-ascend clamps blockDim to this in its
    own launcher (backends/ascend/driver.py, `enable_auto_map_parallel_blocks`)
    and the compiled kernel walks the logical grid -- which is passed in the arg
    buffer -- with a grid-stride loop."""
    global _NUM_AIV
    if _NUM_AIV is None:
        for mod, cls in (("triton.backends.ascend.driver", "NPUUtils"),
                         ("triton.backends.ascend.utils", "NPUUtils")):
            try:
                import importlib
                _NUM_AIV = int(getattr(importlib.import_module(mod), cls)()
                               .get_aivector_core_num())
                break
            except Exception:
                continue
        if not _NUM_AIV:
            _NUM_AIV = 40
    return _NUM_AIV


def _cpp_launch(case, grid_x, ptrs, floats=(), ints=()):
    CL, _ = _cpp_setup()
    b = case["buf"]
    off = 24  # [ffts][syncBlockLock][workspace]
    for p in ptrs:
        struct.pack_into("<Q", b, off, p)
        off += 8
    for f in floats:
        struct.pack_into("<f", b, off, f)
        off += 4
    for i in ints:
        struct.pack_into("<i", b, off, int(i))
        off += 4
    off = (off + 3) & ~3
    struct.pack_into("<iii", b, off, grid_x, 1, 1)
    # v2 FIX: blockDim must be clamped to the physical block count.  Passing the
    # full logical grid (up to max_num_batched_tokens) cost 3.2x device time at
    # B=256 and 12.9x at B=1024, measured on 910B4 with the identical binary.
    nb = _num_aiv()
    block = grid_x if grid_x < nb else nb
    return CL.lora_launch_flat(case["func"],
                               torch.npu.current_stream().npu_stream,
                               block, b, off + 12)


def _cpp_peek_stub(func):
    CL, _ = _cpp_setup()
    if CL is None:
        return 0
    return CL.lora_peek_stub(func)


def _cpp_make_case(kernel_fn, kwargs, example_args, grid):
    """Compile + register one (kernel, constexpr, dtype) case and verify the
    launches land.  Returns case dict or None (caller falls back to triton).

    example_args: kernel positional args with the real tensors of the first
    call; tensors[2] is indices and tensors[-1] is the output for all 4
    kernels.  Compile/warmup and the verify-retry run on dummy output + dummy
    indices so the real output_tensor is never written here.
    """
    CL, ffts = _cpp_setup()
    if CL is None:
        return None

    tensors = [t for t in example_args if isinstance(t, torch.Tensor)]
    # v2: ints (the expand kernel takes a runtime row count) must be packed as
    # <i, not <f, in the verify-retry launch below.
    floats = [f for f in example_args if isinstance(f, float)]
    int_args = [f for f in example_args
                if isinstance(f, int) and not isinstance(f, bool)]
    ti = [i for i, a in enumerate(example_args)
          if isinstance(a, torch.Tensor)]

    # dummy args for compile+verify, DATA-INDEPENDENT: x and w -> ones so the
    # kernel output is deterministically nonzero regardless of what the serve
    # passes (vllm's LoRA warmup may use zeroed dummy weights -> real-data
    # verify would read a legitimate all-zero result and false-negative).
    # indices -> zeros (all -> LoRA 0); every y-position arg -> zero dummies
    # (the two-y kernels write the FIRST y and read the second, so both must
    # be dummies, and the real output tensors are never written here).
    dummy_args = list(example_args)
    dummy_args[ti[0]] = torch.ones_like(example_args[ti[0]])
    dummy_args[ti[1]] = torch.ones_like(example_args[ti[1]])
    dummy_args[ti[2]] = torch.zeros_like(example_args[ti[2]])
    y_positions = [ti[-1]]
    if len(ti) >= 2 and tuple(tensors[-2].shape) == tuple(tensors[-1].shape):
        y_positions = [ti[-2], ti[-1]]
    for p in y_positions:
        dummy_args[p] = torch.zeros_like(example_args[p])

    compiled = kernel_fn.warmup(*dummy_args, **kwargs, grid=grid)
    data = bytes(compiled.kernel)
    buf = ctypes.create_string_buffer(data)
    mode = getattr(compiled.metadata, "mix_mode", "aiv")
    func = CL.lora_register_kernel(compiled.name.encode(), buf, len(data),
                                   mode.encode(),
                                   torch.npu.current_device())
    if not func:
        return None

    ptrs_d = [t.data_ptr() for t in dummy_args if isinstance(t, torch.Tensor)]
    n = 24 + 8 * len(ptrs_d) + 4 * (len(floats) + len(int_args)) + 16
    cbuf = ctypes.create_string_buffer(n)
    struct.pack_into("<QQQ", cbuf, 0, ffts, 0, 0)
    case = dict(func=func, buf=cbuf, nptrs=len(ptrs_d), nfloats=len(floats))

    if os.environ.get("TRITON_LORA_CPP_VERIFY", "1") == "0":
        return case

    # verify-retry: fresh registrations drop launches until the device-side
    # load settles; retry until any dummy y changes from zero.
    ctx = []
    try:
        if torch.compiler.is_compiling():
            ctx.append("is_compiling")
    except Exception:
        pass
    try:
        if torch.npu.is_current_stream_capturing():
            ctx.append("stream_capturing")
    except Exception:
        pass
    t0 = time.perf_counter()
    tries = 0
    while True:
        for p in y_positions:
            dummy_args[p].zero_()
        ret = _cpp_launch(case, grid[0], ptrs_d, floats, int_args)
        torch.npu.synchronize()
        tries += 1
        y_sum = sum(float(dummy_args[p].abs().sum()) for p in y_positions)
        if y_sum != 0.0:
            print(f"[triton-lora] case OK {kernel_fn.__name__} "
                  f"kw={sorted(kwargs.items())} tries={tries} "
                  f"ctx={ctx}", flush=True)
            return case
        if time.perf_counter() - t0 > 10.0:
            # cross-check: does the triton path write on the same dummy args?
            try:
                for p in y_positions:
                    dummy_args[p].zero_()
                kernel_fn[(grid[0],)](*dummy_args, **kwargs)
                torch.npu.synchronize()
                tri_sum = sum(float(dummy_args[p].abs().sum())
                              for p in y_positions)
            except Exception as e:
                tri_sum = f"ERR {e}"
            try:
                stub = _cpp_peek_stub(func)
            except Exception:
                stub = 0
            print(f"[triton-lora] WARN: {kernel_fn.__name__} warmup launches "
                  f"never landed, triton fallback. kw={sorted(kwargs.items())} "
                  f"shapes={[tuple(t.shape) for t in tensors]} "
                  f"x_sum={float(tensors[0].abs().sum()):.1f} "
                  f"w_sum={float(tensors[1].abs().sum()):.1f} "
                  f"idx[:8]={[int(v) for v in tensors[2].flatten().tolist()[:8]]} "
                  f"seq[:8]={[int(v) for v in tensors[3].flatten().tolist()[:8]] if len(tensors) > 4 else '-'} "
                  f"ret={ret:#x} ctx={ctx} tri_sum={tri_sum} "
                  f"stub={stub:#x}", flush=True)
            return None
        time.sleep(0.05)


def _cpp_get_case(kernel_fn, kwargs, example_args, grid):
    key = _cpp_case_key(kernel_fn, kwargs, example_args)
    case = _CPP_CASES.get(key)
    if case is not None:
        return case
    if key in _CPP_FAILED:
        return None
    try:
        case = _cpp_make_case(kernel_fn, kwargs, example_args, grid)
    except Exception as e:
        print(f"[triton-lora] WARN: case setup failed for "
              f"{kernel_fn.__name__}: {e}", flush=True)
        case = None
    if case is None:
        _CPP_FAILED.add(key)
    else:
        _CPP_CASES[key] = case
    return case


# ---- plain wrappers (used by the custom op impls and tests) ----

def bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling=1.0):
    t0 = _timing_start("bgmv_shrink")
    B, H = inputs.shape
    # vllm packs linear lora_a as [L, 1, R, H]; drop the middle 1 dim
    w = lora_a_weights.reshape(lora_a_weights.shape[0], -1, lora_a_weights.shape[-1])
    L, R, _ = w.shape
    idx32 = _to_int32(lora_indices_tensor, "idx")
    kwargs = dict(H=H, R=R, L=L)
    if _cpp_enabled():
        case = _cpp_get_case(K.bgmv_shrink, kwargs,
                             (inputs, w, idx32, output_tensor, scaling), (B,))
        if case is not None:
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx32.data_ptr(),
                                  output_tensor.data_ptr()], [float(scaling)])
            _timing_end("bgmv_shrink", t0)
            return output_tensor
    K.bgmv_shrink[(B,)](inputs, w, idx32, output_tensor, scaling,
                        H=H, R=R, L=L)
    _timing_end("bgmv_shrink", t0)
    return output_tensor


def _expand_blk_ho(R: int) -> int:
    # fp32 accumulator tile R x BLOCK_HO must fit 64KB UB
    return 128 if R * 256 * 4 > 64 * 1024 else 256


def bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                add_inputs=True):
    return bgmv_expand_slice(inputs, lora_b_weights, output_tensor,
                             lora_indices_tensor, 0, output_tensor.size(1), add_inputs)


def bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                      slice_offset, slice_size, add_inputs=True):
    t0 = _timing_start("bgmv_expand_slice")
    B, R = inputs.shape
    # vllm packs linear lora_b as [L, 1, Ho, R]; drop the middle 1 dim
    w = lora_b_weights.reshape(lora_b_weights.shape[0], -1, lora_b_weights.shape[-1])
    L, Ho, _ = w.shape
    idx32 = _to_int32(lora_indices_tensor, "idx")
    kwargs = dict(R=R, Ho=Ho, L=L, BLOCK_HO=_expand_blk_ho(R),
                  Y_HO=output_tensor.size(1), SLICE_OFF=slice_offset)
    if _cpp_enabled():
        case = _cpp_get_case(K.bgmv_expand, kwargs,
                             (inputs, w, idx32,
                              output_tensor, output_tensor), (B,))
        if case is not None:
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx32.data_ptr(),
                                  output_tensor.data_ptr(),
                                  output_tensor.data_ptr()])
            _timing_end("bgmv_expand_slice", t0)
            return output_tensor
    K.bgmv_expand[(B,)](inputs, w, idx32,
                        output_tensor, output_tensor, **kwargs)
    _timing_end("bgmv_expand_slice", t0)
    return output_tensor


_V2 = os.environ.get("TRITON_LORA_V2", "1") != "0"
_V2_EXACT = 1 if os.environ.get("TRITON_LORA_EXACT", "1") != "0" else 0


_V2_BLK_CACHE = {}


def _v2_blk(H: int) -> int:
    """Largest power-of-two <= 512 dividing both H and the 11776 AscendC window
    (so the per-window accumulator restart that keeps down_proj matching stays
    on a block boundary)."""
    b = _V2_BLK_CACHE.get(H)
    if b is None:
        b = 64
        for c in (512, 256, 128, 64):
            if H % c == 0 and 11776 % c == 0:
                b = c
                break
        _V2_BLK_CACHE[H] = b
    return b


_V2_CFG_CACHE = {}


def _v2_expand_cfg(Ho: int, R: int, NR: int, B: int):
    """(BLOCK_HO, TB).

    TB>1 makes one program own TB consecutive token rows so they share a single
    loaded weight tile.  That is only correct when every row in the group maps
    to the SAME lora id, which is guaranteed exactly when the batch is a single
    segment (NR == 1).  `compute_meta` collapses consecutive equal ids, so an
    all-one-adapter batch always yields NR == 1; with NR > 1 a group could
    straddle a segment boundary, so fall back to one row per program, which is
    per-row correct by construction.

    TB is bucketed by B so a B=1 decode step does not pay for 3 masked-off
    rows; every bucket computes each real row identically, so the bucket
    choice cannot change results.
    """
    if NR != 1:
        tb = 1
    elif B <= 2:
        tb = B
    elif B <= 8:
        tb = 4
    else:
        tb = 8
    key = (Ho, R, NR, tb)
    got = _V2_CFG_CACHE.get(key)
    if got is not None:
        return got
    bh = 128
    while bh > 32 and Ho % bh:
        bh //= 2
    while tb > 1 and tb * bh * R * 4 > 96 * 1024:
        tb //= 2
    _V2_CFG_CACHE[key] = (bh, tb)
    return bh, tb


def sgmv_shrink_v1(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                   seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                   token_nums, scaling):
    return _sgmv_shrink_impl(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                             seq_len_tensor, lora_indices_tensor, batches,
                             max_seq_length, token_nums, scaling)


def sgmv_shrink(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                token_nums, scaling):
    if not _V2:
        return _sgmv_shrink_impl(inputs, lora_a_weights, output_tensor,
                                 b_seq_start_loc, seq_len_tensor,
                                 lora_indices_tensor, batches, max_seq_length,
                                 token_nums, scaling)
    t0 = _timing_start("sgmv_shrink")
    B, H = inputs.shape
    # vllm packs lora_a as [L, 1, R, H]; the kernel needs only L, R and the base
    # pointer, and reshaping a contiguous tensor keeps the same data_ptr, so the
    # reshape is materialised only on the plain-triton fallback path.
    sh = lora_a_weights.shape
    L, R = sh[0], sh[-2]
    idx32 = _to_int32(lora_indices_tensor, "idx")
    seq32 = _to_int32(seq_len_tensor, "seq")
    blk = _v2_blk(H)
    # v2 shrink now masks its partial tail (oh < H), so any H is handled
    # natively -- no fallback needed.  blk always divides the 11776 window so
    # the main loop stays exact; only the final tail block can be partial.
    kwargs = dict(scale=scaling, H=H, R=R, L=L, NR=seq_len_tensor.numel(),
                  BLK=blk, NJ=blk // 64, EXACT=_V2_EXACT)
    # decode batches leave most of the 40 AIV cores idle on a (B,) grid; the
    # (B*R,) variant has the identical per-element summation order, so this
    # dispatch cannot change results.
    if B * R <= 40:
        kern, grid = K.sgmv_shrink_v2s, B * R
    else:
        kern, grid = K.sgmv_shrink_v2, B
    if _cpp_enabled():
        case = _cpp_get_case(kern, kwargs,
                             (inputs, lora_a_weights, idx32, seq32,
                              output_tensor), (grid,))
        if case is not None:
            _cpp_launch(case, grid, [inputs.data_ptr(), lora_a_weights.data_ptr(),
                                     idx32.data_ptr(), seq32.data_ptr(),
                                     output_tensor.data_ptr()])
            _timing_end("sgmv_shrink", t0)
            return output_tensor
    w = lora_a_weights.reshape(L, R, H)
    kern[(grid,)](inputs, w, idx32, seq32, output_tensor, **kwargs)
    _timing_end("sgmv_shrink", t0)
    return output_tensor


def _sgmv_shrink_impl(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                      seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                      token_nums, scaling):
    t0 = _timing_start("sgmv_shrink")
    m1 = _timing_start("sgmv_shrink|prep")
    B, H = inputs.shape
    # vllm packs linear lora_a as [L, 1, R, H]; drop the middle 1 dim
    w = lora_a_weights.reshape(lora_a_weights.shape[0], -1, lora_a_weights.shape[-1])
    L, R, _ = w.shape
    idx32 = _to_int32(lora_indices_tensor, "idx")
    seq32 = _to_int32(seq_len_tensor, "seq")
    kwargs = dict(H=H, R=R, L=L, NR=seq_len_tensor.numel(), scale=scaling)
    m2 = _timing_start("sgmv_shrink|lookup")
    m3 = None  # v2: was only bound inside the _cpp_enabled() branch -> NameError
    if _cpp_enabled():
        case = _cpp_get_case(K.sgmv_shrink_kernel, kwargs,
                             (inputs, w, idx32, seq32,
                              output_tensor), (B,))
        m3 = _timing_start("sgmv_shrink|launch")
        if case is not None:
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx32.data_ptr(),
                                  seq32.data_ptr(),
                                  output_tensor.data_ptr()])
            _timing_end("sgmv_shrink", t0)
            _timing_end("sgmv_shrink|prep", m1)
            _timing_end("sgmv_shrink|lookup", m2)
            _timing_end("sgmv_shrink|launch", m3)
            return output_tensor
    _timing_end("sgmv_shrink|prep", m1)
    _timing_end("sgmv_shrink|lookup", m2)
    _timing_end("sgmv_shrink|launch", m3)
    K.sgmv_shrink_kernel[(B,)](inputs, w, idx32, seq32,
                               output_tensor, **kwargs)
    _timing_end("sgmv_shrink", t0)
    return output_tensor


def sgmv_expand(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                token_nums, add_inputs=False):
    return sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                             seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                             token_nums, 0, output_tensor.size(1), add_inputs)


def sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                      seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                      token_nums, slice_offset, slice_size, add_inputs=False):
    if not _V2:
        return _sgmv_expand_slice_impl(inputs, lora_b_weights, output_tensor,
                                       b_seq_start_loc, seq_len_tensor,
                                       lora_indices_tensor, batches, max_seq_length,
                                       token_nums, slice_offset, slice_size,
                                       add_inputs)
    t0 = _timing_start("sgmv_expand_slice")
    B, R = inputs.shape
    sh = lora_b_weights.shape          # [L, 1, Ho, R]; reshape keeps data_ptr
    L, Ho = sh[0], sh[-2]
    idx32 = _to_int32(lora_indices_tensor, "idx")
    seq32 = _to_int32(seq_len_tensor, "seq")
    bh, tb = _v2_expand_cfg(Ho, R, seq_len_tensor.numel(), B)
    # v2 expand now masks its partial last chunk (ho < Ho), so any Ho is
    # handled natively; ceil so the tail chunk is launched.
    nchunk = (Ho + bh - 1) // bh
    grid = ((B + tb - 1) // tb) * nchunk
    kwargs = dict(R=R, Ho=Ho, L=L, NR=seq_len_tensor.numel(), BLOCK_HO=bh,
                  Y_HO=output_tensor.size(1), SLICE_OFF=slice_offset,
                  NCHUNK=nchunk, TB=tb)
    if _cpp_enabled():
        case = _cpp_get_case(K.sgmv_expand_v2, kwargs,
                             (inputs, lora_b_weights, idx32, seq32,
                              output_tensor, output_tensor), (grid,))
        if case is not None:
            _cpp_launch(case, grid, [inputs.data_ptr(), lora_b_weights.data_ptr(),
                                     idx32.data_ptr(), seq32.data_ptr(),
                                     output_tensor.data_ptr(),
                                     output_tensor.data_ptr()])
            _timing_end("sgmv_expand_slice", t0)
            return output_tensor
    w = lora_b_weights.reshape(L, Ho, R)
    K.sgmv_expand_v2[(grid,)](inputs, w, idx32, seq32, output_tensor,
                              output_tensor, **kwargs)
    _timing_end("sgmv_expand_slice", t0)
    return output_tensor


def _sgmv_expand_slice_impl(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                            seq_len_tensor, lora_indices_tensor, batches,
                            max_seq_length, token_nums, slice_offset, slice_size,
                            add_inputs=False):
    t0 = _timing_start("sgmv_expand_slice")
    m1 = _timing_start("sgmv_expand_slice|prep")
    B, R = inputs.shape
    # vllm packs linear lora_b as [L, 1, Ho, R]; drop the middle 1 dim
    w = lora_b_weights.reshape(lora_b_weights.shape[0], -1, lora_b_weights.shape[-1])
    L, Ho, _ = w.shape
    idx32 = _to_int32(lora_indices_tensor, "idx")
    seq32 = _to_int32(seq_len_tensor, "seq")
    kwargs = dict(R=R, Ho=Ho, L=L, NR=seq_len_tensor.numel(),
                  BLOCK_HO=_expand_blk_ho(R), Y_HO=output_tensor.size(1),
                  SLICE_OFF=slice_offset)
    m2 = _timing_start("sgmv_expand_slice|lookup")
    m3 = None  # v2: see sgmv_shrink
    if _cpp_enabled():
        case = _cpp_get_case(K.sgmv_expand, kwargs,
                             (inputs, w, idx32, seq32,
                              output_tensor, output_tensor), (B,))
        m3 = _timing_start("sgmv_expand_slice|launch")
        if case is not None:
            _cpp_launch(case, B, [inputs.data_ptr(), w.data_ptr(),
                                  idx32.data_ptr(),
                                  seq32.data_ptr(),
                                  output_tensor.data_ptr(),
                                  output_tensor.data_ptr()])
            _timing_end("sgmv_expand_slice", t0)
            _timing_end("sgmv_expand_slice|prep", m1)
            _timing_end("sgmv_expand_slice|lookup", m2)
            _timing_end("sgmv_expand_slice|launch", m3)
            return output_tensor
    _timing_end("sgmv_expand_slice", t0)
    _timing_end("sgmv_expand_slice|prep", m1)
    _timing_end("sgmv_expand_slice|lookup", m2)
    _timing_end("sgmv_expand_slice|launch", m3)
    K.sgmv_expand[(B,)](inputs, w, idx32, seq32,
                        output_tensor, output_tensor, **kwargs)
    _timing_end("sgmv_expand_slice", t0)
    return output_tensor


# ---- torch custom ops (dynamo-opaque, eager impls run during graph capture) ----
#
# Registration happens at the bottom of the file.  DEFAULT: these python
# impls are registered at the python dispatch key via torch.library.custom_op
# (the only form that runs on real tensors in serve).  Only with explicit
# TRITON_LORA_NATIVE=1 is the C++ .so loaded instead (standalone debugging).


def _op_bgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                    scaling: float) -> None:
    _announce()
    if (_triton_dtype_ok(inputs) and _triton_dtype_ok(lora_a_weights)
            and lora_a_weights.shape[-1] == inputs.shape[-1]):
        bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling)
    else:
        _ascend_bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling)


def _fake_bgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                      output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                      scaling: float) -> None:
    return None


def _op_bgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                          slice_offset: int, slice_size: int) -> None:
    _announce()
    if (_triton_dtype_ok(lora_b_weights)
            and lora_b_weights.shape[-1] == inputs.shape[-1]):  # linear [L,1,Ho,R], not embedding
        bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                          slice_offset, slice_size, True)
    else:
        _ascend_bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                            slice_offset, slice_size)


def _fake_bgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                            output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                            slice_offset: int, slice_size: int) -> None:
    return None


def _op_sgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor, scaling: float) -> None:
    _announce()
    if (_triton_dtype_ok(inputs) and _triton_dtype_ok(lora_a_weights)
            and lora_a_weights.shape[-1] == inputs.shape[-1]
            and _sgmv_triton_ok(seq_len_tensor)):
        sgmv_shrink(inputs, lora_a_weights, output_tensor, None, seq_len_tensor,
                    lora_indices_tensor, int(inputs.shape[0]), int(inputs.shape[1]),
                    int(seq_len_tensor.numel()), scaling)
    else:
        _ascend_sgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor,
                            seq_len_tensor, scaling)


def _fake_sgmv_shrink(inputs: torch.Tensor, lora_a_weights: torch.Tensor,
                      output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor, scaling: float) -> None:
    return None


def _op_sgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                          lora_indices_tensor: torch.Tensor,
                          slice_offset: int, slice_size: int) -> None:
    _announce()
    if (_triton_dtype_ok(lora_b_weights)
            and lora_b_weights.shape[-1] == inputs.shape[-1]  # linear [L,1,Ho,R], not embedding
            and _sgmv_triton_ok(seq_len_tensor)):
        sgmv_expand_slice(inputs, lora_b_weights, output_tensor, None, seq_len_tensor,
                          lora_indices_tensor, int(inputs.shape[0]), int(inputs.shape[1]),
                          int(seq_len_tensor.numel()), slice_offset, slice_size, True)
    else:
        _ascend_sgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                            seq_len_tensor, slice_offset, slice_size)


def _fake_sgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                            output_tensor: torch.Tensor, seq_len_tensor: torch.Tensor,
                            lora_indices_tensor: torch.Tensor,
                            slice_offset: int, slice_size: int) -> None:
    return None


# ---- op registration (python impls by default; native C++ only opt-in) ----

_OPS = (
    ("vllm_ascend_triton::bgmv_shrink", _op_bgmv_shrink, _fake_bgmv_shrink,
     ("output_tensor",)),
    ("vllm_ascend_triton::bgmv_expand_slice", _op_bgmv_expand_slice,
     _fake_bgmv_expand_slice, ("output_tensor",)),
    ("vllm_ascend_triton::sgmv_shrink", _op_sgmv_shrink, _fake_sgmv_shrink,
     ("output_tensor",)),
    ("vllm_ascend_triton::sgmv_expand_slice", _op_sgmv_expand_slice,
     _fake_sgmv_expand_slice, ("output_tensor",)),
)


_SCHEMA_LIB = None  # keep the DEF Library alive (namespace dies with the handle)


def _define_schemas():
    """Schema-only registration; NPU dispatch then falls through to the C++
    TORCH_LIBRARY_IMPL(..., PrivateUse1) kernels so torch_npu's aclgraph
    capture records them exactly like the AscendC ops.

    NOTE: the DEF Library object MUST be kept alive for the whole process --
    when the handle is garbage-collected torch destroys the namespace with
    everything defined in it (custom_op works around this by leaking).
    """
    global _SCHEMA_LIB
    try:
        lib = torch.library.Library("vllm_ascend_triton", "DEF")
        for s in (
            "bgmv_shrink(Tensor inputs, Tensor lora_a_weights, Tensor output_tensor, Tensor lora_indices_tensor, float scaling) -> ()",
            "bgmv_expand_slice(Tensor inputs, Tensor lora_b_weights, Tensor output_tensor, Tensor lora_indices_tensor, int slice_offset, int slice_size) -> ()",
            "sgmv_shrink(Tensor inputs, Tensor lora_a_weights, Tensor output_tensor, Tensor seq_len_tensor, Tensor lora_indices_tensor, float scaling) -> ()",
            "sgmv_expand_slice(Tensor inputs, Tensor lora_b_weights, Tensor output_tensor, Tensor seq_len_tensor, Tensor lora_indices_tensor, int slice_offset, int slice_size) -> ()",
        ):
            try:
                lib.define(s)
            except RuntimeError:
                pass  # already defined
        _SCHEMA_LIB = lib
    except RuntimeError:
        pass  # namespace already defined


if os.environ.get("TRITON_LORA_NATIVE", "0") != "0":
    # opt-in standalone-only mode (serve has no python impl then -> LoRA dead,
    # see header comment).  DEFAULT is python impls below.
    _define_schemas()
    _native_setup()

for _opname, _opfn, _fakfn, _mut in _OPS:
    if _NATIVE is not None:
        # native .so loaded (explicit TRITON_LORA_NATIVE=1): schema + fake
        # only, dispatch goes to the C++ PrivateUse1 kernels.
        register_fake(_opname, _fakfn)
    else:
        # DEFAULT: python-key python impl (06:10-era shape, the only form
        # proven to run on real tensors inside serve).  FakeTensor tracing is
        # still served by the Fake key via register_fake below.
        try:
            custom_op(_opname, mutates_args=_mut)(_opfn)
        except RuntimeError:
            # schema already defined: register the python impl directly
            torch.library.impl(_opname, "Python", _opfn)
        register_fake(_opname, _fakfn)


