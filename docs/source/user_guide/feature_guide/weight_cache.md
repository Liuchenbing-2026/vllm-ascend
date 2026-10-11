# Weight Cache (Fast Restart)

## Overview

vLLM 0.31 introduced the weight cache daemon (`vllm preload`): one resident
process per device holds the post-quantized, TP-sharded weights in device
memory and serves them to engines over Unix sockets as device IPC handles.
Restarting an engine then maps the weights instead of re-reading and
re-converting them from disk, cutting engine restart time from minutes to
seconds.

vllm-ascend enables this feature on Ascend NPUs (upstream vLLM only allows
CUDA/ROCm). The daemon and the engine communicate through torch_npu's NPU IPC
storage sharing, and weights stored in FRACTAL_NZ layout are normalized to ND
in the daemon and cast back to NZ in the engine after mapping (a lossless,
device-side layout conversion).

## Requirements

- vLLM >= 0.31
- torch_npu with NPU IPC storage sharing (`torch.UntypedStorage._share_npu_` /
  `_new_shared_npu`); shipped with current torch_npu releases. If missing,
  `vllm preload` fails fast with a clear error.
- Linux (the daemon uses Unix domain sockets), same-user daemon and engine
  processes.

## Usage

Start the daemon once (it holds the weights resident):

```bash
vllm preload --model /path/to/model --tensor-parallel-size 4
```

Start (and restart) engines against the running daemon:

```bash
vllm serve /path/to/model --tensor-parallel-size 4 --load-format ipc_cache
```

If the daemon is unavailable, not ready yet, or its cached weights do not
match the engine configuration, the engine falls back to loading from disk
by default; pass `--model-loader-extra-config '{"fallback": false}'` to make
it fail instead.

Useful options:

- `--weight-cache-health-port <port>`: exposes `/health` on the preload
  launcher (200 once all daemons are ready, 503 otherwise), so orchestration
  can wait for the cache before starting engines.
- `--weight-cache-master-port <port>`: rendezvous port for the daemon's own
  TP/DP group; must differ from the engine's master port. Required for
  multi-node (`--nnodes > 1`).
- Data parallelism and MTP/EAGLE/EAGLE3 draft models are supported: the
  launcher starts a separate draft daemon group automatically and engines
  load the draft weights from it.
- `--model-loader-extra-config '{"mode": "copy"}'`: clone the weights into
  engine-owned memory and release the daemon's cache afterwards, instead of
  the default zero-copy sharing.

## Ascend-specific notes

- **Verified quantization methods**: unquantized linear/MoE and
  `W8A8_DYNAMIC` (linear and MoE). Other Ascend quantization methods are
  rejected explicitly by `check_ipc_quant_support` rather than silently
  mis-loading; support can be extended per method.
- **Unsupported modes**: configurations that replace registered weights with
  per-expert lists (dynamic EPLB weight lists, CANN MegaMoE list mode for
  unquantized MoE) are rejected, because those modes delete the registered
  weight tensors the daemon exports.
- **NZ weights**: the daemon keeps ND-normalized copies (~1x weight memory)
  and every engine pays one fast device-side ND->NZ cast per NZ tensor at
  load time. Zero-copy sharing still applies; no disk or CPU round trip is
  involved.
- **Sleep mode**: do not combine zero-copy IPC weights with sleep-mode
  weight offloading; use `mode: copy` or disable sleep mode.
- **`vllm snapshot`**: not supported on Ascend. Upstream snapshots rely on
  CRIU plus the NVIDIA-only `cuda-checkpoint` plugin to freeze and restore
  device state, and CANN provides no equivalent. `vllm snapshot
  create/restore` raises a clear error on NPU.
- **GPU memory utilization**: in zero-copy mode the daemon's cached weights
  stay resident on the NPU, so the engine's default
  `--gpu-memory-utilization 0.9x` startup check can fail even though the
  weights themselves are shared. Lower it so that
  `utilization * total <= free` (e.g. `--gpu-memory-utilization 0.65`),
  matching upstream's own weight-cache test setups; KV cache sizing still
  uses the actually free memory.
- **vLLM v0.31.0 release lane**: the preload feature reached NPU on the
  v0.31.0 lane, where a few vLLM-main APIs are absent (routed-experts
  capture, `GPUInputBatch` alias, tuple-style `get_mamba_groups`,
  `_seq_lens_cpu`, `randomize_inputs`, `prefill_end`). vllm-ascend detects
  each at runtime and falls back, so the same tree runs on both lanes. On
  CANN builds whose public ops package lacks Ascend910B binaries for newer
  custom ops (e.g. `AddRmsNormBias`, `CausalConv1d`), the affected kernels
  are probed once at worker init and fall back to the `torch_npu`
  implementations. Rebuilding the vendor package with kernel binaries
  (`COMPILE_CUSTOM_KERNELS=1`, i.e. `build.sh --pkg`) restores full op
  support; GDN models (Qwen3.5) have been validated end to end on
  Ascend910B4 with the rebuilt package. One build-time hazard: third-party
  CANN vendor packages installed in the same Python environment (e.g. the
  `fla_npu` pip package) can register same-named ops with different
  signatures; CANN's opc consistency check then fails the build with
  `invalid input nums` errors. Neutralize such packages (e.g. rename their
  `opp/` directory) before compiling custom ops.
