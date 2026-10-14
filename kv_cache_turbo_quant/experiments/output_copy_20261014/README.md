# Output rotation write experiment, 2026-10-14

| Scope | Evidence |
|---|---|
| Base | `f80f115f4d05a6361f3fcf5af0686871847f929f` |
| Branch | `perf/tq-output-copy-20261014` |
| Change | Decode and prefill-with-history write FP32 rotation results through `output.copy_`, avoiding the explicit BF16 intermediate |
| CPU | 24 exact cases, including noncontiguous caller output and untouched prefix/tail |
| NPU | The same 24 exact cases plus 20 nondefault-stream producer/consumer cases |
| Unchanged | FP32 rotation, attention, actual 66-byte/head packed KV, native quantizer and paired reader |
| Whole model | Pending; the microbenchmarks are not TTFT, TPOT or TQ-on versus BF16-off gains |

The test executes the actual output blocks extracted from the baseline and candidate source. It compares complete caller buffers byte for byte. The microbenchmark uses FP32 incoming fixtures and 16 query heads of dimension 128. Each row below includes all three alternating samples, ten warmups and 100 measured calls per sample. Device event intervals include host submission gaps. The production FIA output may additionally require the unchanged BF16-to-FP32 input cast.

| Block | Rows | Baseline μs | Candidate μs | Reduction |
|---|---:|---:|---:|---:|
| Decode rotation and write | 16 | 131.146 | 56.428 | 56.97% |
| Decode rotation and write | 60 | 130.456 | 56.019 | 57.06% |
| Decode rotation and write | 4096 | 91.841 | 56.650 | 38.32% |
| Prefill history rotation and write | 16 | 135.347 | 55.195 | 59.22% |
| Prefill history rotation and write | 60 | 123.746 | 53.886 | 56.45% |
| Prefill history rotation and write | 4096 | 144.236 | 56.862 | 60.58% |

Actual runtime: 910B4-1, physical card 3 on the migrated host; Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post4, CANN 9.1.0. Image `vllm-ascend:dspark-a2-028`, local image ID `sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`, no registry RepoDigest reported. This uses a preserved runtime on an existing image, not a clean-from-image reproduction. CI and a new semantic-quality evaluation have not run.

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
/ws/.venv/bin/python verify_output_copy.py \
  --baseline /ws/paired_model_inference_20261014/candidate_write/integration/kvtq_store.py \
  --candidate /ws/output_copy_gate_20261014/candidate.py \
  --device cpu --output cpu.json
/ws/.venv/bin/python verify_output_copy.py \
  --baseline /ws/paired_model_inference_20261014/candidate_write/integration/kvtq_store.py \
  --candidate /ws/output_copy_gate_20261014/candidate.py \
  --device npu --output npu.json
```

The complete commands and original results are retained alongside this file. Paired service profiling and formal benchmarks must run separately with fixed TP=2, 24 GiB/card KV, eager, prefix/async disabled, budget 4096 and max_num_seqs 512. The target of nonregression at 20/20 C16 and a gain at 2000/200 C60 against BF16 has not been met.
