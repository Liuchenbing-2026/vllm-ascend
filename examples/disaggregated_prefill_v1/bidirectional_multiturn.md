# Experimental multi-turn Mooncake pull on Ascend

This opt-in extension lets Prefill reuse full computed KV blocks retained by
Decode after a previous turn. KV tensors move directly through the existing
Mooncake `ascend` READ transport. The router stores metadata and token-prefix
hashes; it does not store KV tensors. This path does not use a Mooncake Store.

| Component | Reference |
| --- | --- |
| Ascend base | `v0.30.0rc2`, `8fde6d4e335fe300e2b949aab0e31659cce0811d` |
| Core base used by the rc2 Dockerfile | vLLM `v0.30.0`, `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Required core streaming companion | [6c48dce1beb16000ebffd937402608fdd86bde82](https://github.com/Liuchenbing-2026/vllm/commit/6c48dce1beb16000ebffd937402608fdd86bde82) |
| Base image tag | `quay.io/ascend/vllm-ascend:v0.30.0rc2` |
| Transfer connector | `MooncakeConnectorV2` |
| Router | Native Rust binary with `--kv-connector mooncake-pull` |
| Software checks | 217 Mooncake CPU tests; 780 companion Router tests |
| Device data plane | Both 4 MiB directions match; corrected diagnostic exits normally |
| Model correctness | Real Qwen3-8B, two turns, all 512 output IDs match OFF |
| Warmed TTFT | All three fresh AB/BA/AB pairs passed the per-length gain gate |
| P prefix caching enabled | Generated-page partial READ and local-hit zero-READ cases passed |

Keep an exact image digest and the installed package versions from the actual
run. A tag and the Dockerfile's defaults do not establish the image's contents.
The tested image digest was
`sha256:6348403a0ebb19f52bc1b7497de1daeb6df0e4650f9051c169e47a011527a6c3`,
with core `0.30.0+empty`, Ascend `0.30.0rc2`, Torch `2.10.0+cpu`,
Torch-NPU `2.10.0.post7`, and Mooncake `0.3.11.post1` on 910B4-1.

The fixed core base omits `kv_transfer_params` from streaming chat and completion
responses. A real-model run completed with identical output IDs but missed reuse
on every turn. Apply the separate core companion above to both OFF and ON:
its four runtime files add terminal-chunk metadata without changing model or
runner code. Eight focused CPU stream cases and a 13-case compatibility subset
passed. This dependency is required in addition to this Ascend adaptation and
the native Router; successful ordinary streaming alone does not prove reuse.

The standalone diagnostic byte-checked P→D and updated D→P on shared NPUs.
Its initial implementation imported Mooncake before selecting the NPU device,
then aborted during native teardown. Memory unregistration and releasing
Torch storage alone did not resolve it. Selecting the device **before importing
`mooncake.engine`** fixed the diagnostic: two runs exited normally, with both
workers returning zero. Restoring only the old order reproduced both aborts.
The internal native heap failure is not otherwise diagnosed. Preserve those
failed runs and keep nonzero worker exits as `success=false`.

The standalone device-buffer result does not exercise a complete model.
A separate real Qwen3-8B BF16 run on two 910B4-1 devices completed two turns
with identical inputs and all 512 output IDs matching OFF. Native P-side READ
logs and transfer metrics agree and cover a complete generated-token KV page,
in addition to the previous prompt. That run uses TP1, eager execution, a
16 GiB KV pool per endpoint and P prefix caching disabled. It establishes
targeted model reuse. Separate warmed performance validation completed all
three AB/BA/AB pairs with fresh engines, identical model/package hashes and
the same immutable image in all six arms. Each arm had equal warmup, followed
by two conversations of three turns at each initial prompt length. Every
pair passed the unchanged per-length gain gate, exact input/output-ID checks
and physical READ accounting. This is a serial, prefix-eviction workload;
general task accuracy and production goodput were not evaluated.
The host uses driver `26.0.rc1` and CANN `9.2.0` (`V100R001C25B114`).
The source uses no changed model runner.

A separate fresh deployment enabled P prefix caching with all other model and
connector settings unchanged. Its long-output case matched all output IDs and
read only the complete generated-token page missing from P. A second case
matched an ordinary-Prefill reference while forwarding a remote candidate and
performing no reverse READ, because P already had the required prefix. Both
endpoints reported no failed SDK transfers. These are correctness checks;
they are excluded from the prefix-eviction TTFT results.

## Supported configuration

Use TP/PP/DP/PCP/DCP=1, full attention and no speculative decoding. Hybrid,
sliding-window, MLA and compressed-cache layouts are excluded. Both engines
must enable `bidirectional_kv_xfer`; both also need the ordinary P→D connector.
The legacy `MooncakeConnectorV1` is unchanged.

```json
{
  "kv_connector": "MooncakeConnectorV2",
  "kv_role": "kv_producer",
  "engine_id": "bidir-prefill",
  "kv_port": 29520,
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "bidirectional_kv_xfer": true,
    "decoder_kv_blocks_ttl": 480,
    "kv_recompute_threshold": 64
  }
}
```

On Decode use `kv_role: kv_consumer`, a distinct engine ID, and `kv_port: 29530`.
`decoder_kv_blocks_ttl` controls unclaimed completed D requests. The recompute
threshold compares remote full tokens absent from P's local prefix cache.
Only full computed blocks are advertised: the final generated token has not
necessarily had its KV computed, and an incomplete page is recomputed.

Send `conversation_id` and `return_token_ids: true` through a router which
checks the exact rendered token prefix. Keep the router TTL below the engine
TTL. The client path covered by protocol tests uses token-ID `/v1/completions` and preserves every
generated ID. Rewritten histories must fall back to ordinary Prefill.

Before reverse DMA, P claims the retained D request. An expired or duplicate
claim fails through native KV-load failure handling. Configure `recompute`.
A successful claim pins source pages until that consumer's completion ACK;
an ACK from a different consumer cannot release the source. This prevents a
TTL timer from recycling a source page during an in-flight READ.

**Failure limit:** a receiver process that dies after claiming may never ACK.
Its claimed D pages then remain allocated until the source engine is restarted.
There is no distributed lease recovery in this experimental extension. Do not
interpret the retention TTL as an in-flight transfer timeout or production HA.

## Real-device acceptance commands

For a bounded data-plane smoke test, run the diagnostic **as a module** from
the repository root. This also avoids `tools/bisect` shadowing Python's
standard-library module. It checks P→D and updated D→P buffer contents on two
mapped physical NPUs. It starts no model and establishes no KV-lifecycle or
TTFT result:

```bash
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.listen_port":29666}'
python3 -m tools.test_mooncake_bidirectional_read \
  --devices 0 1 --local-ip <host-ip> --bytes 4194304 \
  --output /work/transport-result.json
```

Each process owns a source and destination buffer; the SDK/device context has
additional memory costs. Record the real resource occupancy separately.
When using host physical IDs with `ASCEND_RT_VISIBLE_DEVICES`, map the complete
physical device list into the container and restrict each process with its
runtime mask. Mapping only host devices 4 and 5 renumbers the container's
runtime devices; then a mask of `4,5` exposes zero devices. Record this mapping
and select the same two physical devices in both arms.
HCCS requires 2 MiB-aligned registered device addresses; the diagnostic aligns
both buffers. On a shared NPU the default HIXL port 16666 may already be bound.
Choose a free per-NPU `comm_resource_config.listen_port` in the existing
`ASCEND_GLOBAL_RESOURCE_CONFIG` JSON. `HCCL_NPU_SOCKET_PORT_RANGE=auto` alone
does not override HIXL's explicit listen port. These settings apply to the new
task's processes. See the
[HCCL port configuration](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0144.html)
and [Mooncake HCCS requirements](https://github.com/kvcache-ai/Mooncake/blob/d750cddafa4b62387ca135949595008c51e05815/docs/source/design/transfer-engine/transport/ascend_direct_transport.md).

Build the matching Rust router for the target CPU architecture. Run the launcher
inside a dedicated rc2 container with the task source, driver libraries, two
free NPU devices, the fixed model and the router binary mounted. Its preflight
rejects devices with existing processes and occupied task ports.

```bash
export MODEL=/models/qwen3-8b
export ROUTER_BIN=/work/bin/vllm-router
export P_DEVICE=0 D_DEVICE=1
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.listen_port":29666}'

# Terminal 1: keep each arm in a new directory and restart both engines/router.
LEG=off RUN_DIR=/work/pair1-off \
  bash examples/disaggregated_prefill_v1/serve_bidirectional_multiturn.sh

# Terminal 2: benchmark.py comes from the companion Router branch.
python3 /work/router/examples/bidirectional_kv/benchmark.py measure \
  --transport mooncake --leg off --pair 1 --model bidir \
  --url http://127.0.0.1:29512 --prefill http://127.0.0.1:29510 \
  --output /work/results-pair1-off

# Stop only the task-owned launcher/container; restart with LEG=on.
LEG=on RUN_DIR=/work/pair1-on \
  bash examples/disaggregated_prefill_v1/serve_bidirectional_multiturn.sh
python3 /work/router/examples/bidirectional_kv/benchmark.py measure \
  --transport mooncake --leg on --pair 1 --model bidir \
  --url http://127.0.0.1:29512 --prefill http://127.0.0.1:29510 \
  --output /work/results-pair1-on

python3 /work/router/examples/bidirectional_kv/benchmark.py compare \
  --off /work/results-pair1-off/results.json \
  --on /work/results-pair1-on/results.json --output /work/comparison-pair1.json
```

The strict gate checks identical input/output token IDs, later-turn router
candidate receipts, P-side `vllm:mooncake_bytes_transferred_sum` OFF=0/ON>0,
and at least 1.05× median later-turn TTFT improvement at every tested length.
Repeat AB/BA/AB with fresh engines. Preserve failed arms and all raw evidence.
P prefix caching is disabled here to model eviction. Exact generated IDs are a
regression check, not a complete task-quality benchmark. The launcher and
benchmark commands above are templates: the recorded run used physical P2/D3,
`MODEL=/models/Qwen3-8B`, `KV_MEMORY_BYTES=17179869184`, and the fixed core
companion in both arms. Preserve the actual Docker command, image identity, model hashes,
warmup and formal-request records when reproducing the acceptance result.
