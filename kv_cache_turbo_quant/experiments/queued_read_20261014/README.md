# Queue-aware native TQ4 reader (candidate, 2026-10-14)

The 66-byte/head codec, quantizer, ACLNN API and Ascend C device reader are unchanged. Only the Torch wrapper submits through `OpCommand::RunOpApiV2` on `.stream(false)`, keeping descriptors and Tensor storage alive until submission. The intended benefit is avoiding the synchronous queue-empty path. Whole-model ON/OFF benefit remains unaccepted pending the matched matrix.

Base: `c894ba84c9bb1325dd6b689b1cf41a9ce3f26f6c`; branch: `perf/tq-queued-read-20261014`. Exact delivery SHA is supplied by the archive after push. Native kernel/API remain from the base experiment `native_read_20261013`; build those first with its recorded build command. No framework source changed: vLLM `2cf0a6915ce544dc493a0990f2ea38d81601128a`, vLLM-Ascend `96df623103f921df1a4488170d106f36689acb59`.

Actual image: `vllm-ascend:dspark-a2-028`; image ID `sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`; RepoDigests empty, clean-image/public-pull rebuild unverified. Existing dedicated container `tq-opt-20261011-model`, physical 910B4-1 cards 2/5, CANN 9.1.0, Python 3.12.13, torch 2.10.0+cpu, torch_npu 2.10.0.post4, Triton-Ascend 3.2.2. Host `/data02/tq-opt-20261011` maps to `/ws`; model mounted read-only at `/models/Qwen3-30B-A3B`. Extra dependency inventory and fixed framework export/source verification are in the task archive.

Actual build (source/glue under `/ws/queued_read_20261014`):

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export TORCH_EXTENSIONS_DIR=/ws/queued_read_20261014/torch-extensions MAX_JOBS=2
/ws/.venv/bin/python /ws/queued_read_20261014/torch_ext/build_queued_read.py
```

`evidence/gate-attempt1/command.json` records the complete precision command and environment; `evidence/memory-attempt1/command.json` records the complete kernel-filtered mssanitizer command. 15 original codec cases retain exact BF16-bit equality; 20 additional producer/read/consumer ordering cases cover default and nondefault streams plus immediately released temporary inputs. Native memcheck reports 38 calls without access errors. LeakCheck still reports the separately tracked 2MiB process-exit leak, also observed without the new reader; its root cause is open. This is not a statement that all memory issues are resolved. No new semantic quality dataset or hardware racecheck was run.

The actual durable benchmark entry point is `model-runtime/scripts/host.py` (host) followed by `run_write_ab.py` (container), with `monitor.py` recording exit. Its paths are fixed to `/ws/queued_model_20261014`; required historical manifests/gates remain explicit dependencies. Actual deployed scripts are preserved byte for byte, rather than rewritten as a supposedly standalone launcher.

```bash
docker exec -e TASK_NPU_CARDS=2,5 tq-opt-20261011-model bash -lc \
 'source /usr/local/Ascend/cann-9.1.0/set_env.sh && exec /ws/.venv/bin/python -u /ws/queued_model_20261014/scripts/run_write_ab.py'
```

`model-runtime/scripts/serve.sh` contains all service arguments and environment variables. Use `bf16` for OFF, `candidate` for this candidate, `optimized` for the previous native-reader baseline. All have TP=2, KV bytes/card=25769803776, max_num_seqs=512, budget=4096, model_len=8192, eager, prefix caching and async scheduling disabled. Scenes are 20/20 C16, 2000/200 C60, 8000/1 C256, one warmup and three complete formal cohorts per mode, with shared payloads and exact per-request usage. Profile and warmup cohorts are excluded; failures are retained and prohibit acceptance. Single-token output TPOT is null. The controller separately compares 440 greedy output IDs between previous and queued native TQ; that check is not BF16 semantic quality evaluation.

Verified here: existing-container build, strict codec/ordering gates, positive kernel memcheck coverage. The formal model matrix is running independently and must be read from the task archive. Static reproduction audit and from-scratch reproduction are distinct; the latter has not been done.
