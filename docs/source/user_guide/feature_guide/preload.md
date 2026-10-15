# Preload weights for engine restart

The NPU weight-cache transport shares unquantized ND weights between a persistent
preload daemon and inference workers through native torch_npu IPC. Engines can
restart without reading the checkpoint and copying its weights to the NPU again.
KV cache initialization, compilation and graph capture are still engine startup
work; preload is not a full engine snapshot.

## Requirements

- Apply the paired vLLM CLI/platform weight-cache change and the Ascend transport
  change. The v0.30.0rc2 image alone does not provide the NPU preload feature.
- Use the same checkpoint, dtype, tensor parallel layout, physical devices and
  runtime settings for the daemon and the engine.
- Use unquantized ND weights: `--additional-config '{"weight_nz_mode":0}'`.
  Quantized weights, internal NZ layouts and sleep mode are rejected.
- Set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` before launching either
  process. The validation image uses torch-npu 2.10.0.post7 and CANN 9.2.0.
- Reserve enough device memory for the resident weights and the engine's KV cache
  and workspaces. Keep other workloads off the assigned devices when benchmarking.
- Set a positive `--shutdown-timeout` on the engine (for example, `30`). The
  default zero timeout can kill EngineCore before IPC references are released,
  so NPU IPC loading rejects it. Forced kills are not a graceful restart.

## Start the daemon and engine

In both terminals, source the CANN environment and set the allocated physical
device IDs. The example uses two devices and a private Unix socket directory:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES='<NPU_ID_0>,<NPU_ID_1>'
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MODEL_PATH='<LOCAL_CHECKPOINT>'
```

Terminal 1:

```bash
vllm preload "$MODEL_PATH" --dtype bfloat16 --tensor-parallel-size 2 \
  --additional-config '{"weight_nz_mode":0}' --enforce-eager \
  --weight-cache-socket-dir /tmp/npu-preload
```

Wait for `Weight cache daemon READY`, then start terminal 2:

```bash
vllm serve "$MODEL_PATH" --dtype bfloat16 --tensor-parallel-size 2 \
  --shutdown-timeout 30 \
  --additional-config '{"weight_nz_mode":0}' --enforce-eager \
  --load-format ipc_cache \
  --model-loader-extra-config '{"socket_dir":"/tmp/npu-preload","fallback":false}'
```

Restart only terminal 2. Confirm each rank logs `Mapped ... tensors from the
weight cache daemon (zero_copy mode)` and does not load checkpoint shards.
The daemon must remain alive while any zero-copy engine uses its allocations.
Stop engines before stopping the daemon. SIGTERM removes the daemon sockets;
lock files can remain and do not indicate a live cache.

## Modes and validation

| Setting | Meaning |
| --- | --- |
| `mode: "zero_copy"` | Share device weights; the default mode for repeated restarts. |
| `mode: "copy"` | Clone into engine-owned device memory, then release the producer's weights. The cache must be reloaded before another cache hit. |
| `fallback: false` | Reject cache misses and fingerprint mismatches. Use this during acceptance testing. |
| `fallback: true` | Allow the existing disk loader when the cache is unusable. Check logs before claiming a cache hit. |

Compare default disk loading and at least two restarts with identical prompts,
generation parameters, device allocation and KV budget. Check generated tokens
and log probabilities, report model-loading time separately from process-to-health
time, and measure the initial preload cost separately. A health response alone
does not validate model output or prove that IPC was used.

The daemon in this version rejects pipeline and data parallelism. NPU system tests
must cover the requested model and parallel layout; CUDA/ROCm results do not
establish NPU support. Multinode, quantized models and other hardware revisions
are outside this port's validation scope.
