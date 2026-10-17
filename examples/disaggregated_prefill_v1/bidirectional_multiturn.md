# Experimental multi-turn Mooncake pull on Ascend

This opt-in extension lets Prefill reuse full computed KV blocks retained by
Decode after a previous turn. KV tensors move directly through the existing
Mooncake `ascend` READ transport. The router stores metadata and token-prefix
hashes; it does not store KV tensors. This path does not use a Mooncake Store.

| Component | Reference |
| --- | --- |
| Ascend base | `v0.30.0rc2`, `8fde6d4e335fe300e2b949aab0e31659cce0811d` |
| Core base used by the rc2 Dockerfile | vLLM `v0.30.0`, `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Base image tag | `quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2` |
| Transfer connector | `MooncakeConnectorV2` |
| Router | Native Rust binary with `--kv-connector mooncake-pull` |
| Software checks | 217 Mooncake CPU tests; 780 companion Router tests |
| Device data plane | Both 4 MiB directions match; strict diagnostic fails native teardown |
| Model/TTFT | Not measured |

Keep an exact image digest and the installed package versions from the actual
run. A tag and the Dockerfile's defaults do not establish the image's contents.
The tested image digest was
`sha256:6348403a0ebb19f52bc1b7497de1daeb6df0e4650f9051c169e47a011527a6c3`,
with core `0.30.0+empty`, Ascend `0.30.0rc2`, Torch `2.10.0+cpu`,
Torch-NPU `2.10.0.post7`, and Mooncake `0.3.11.post1` on 910B4-1.

The standalone diagnostic completed and byte-checked P→D and updated D→P on
shared NPUs. Its workers then aborted during native teardown; explicit memory
unregistration and releasing Torch storage did not resolve the failure. The
root cause is pending. Nonzero worker exits keep `success=false`; this is not
a fully passed device test. This diagnostic does not exercise the new engine
KV lifecycle or a complete model. The source uses no changed model runner.

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
P prefix caching is disabled here to model eviction; also test it enabled before
making a broader operating claim. Exact generated IDs are a regression check,
not a complete task-quality benchmark. The commands are prepared, not a claim
that model, physical KV transfer or performance acceptance has passed.
