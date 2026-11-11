# FlashStates benchmark artifacts

This directory contains the **validated** benchmark path and raw results for the
FlashStates hybrid-state frontier port.

## Validated track: vLLM v0.32.0 core

The numbers below were produced with the vLLM core patches in
`patches/flashstate_v032/` applied to vLLM **v0.32.0** (`5df63a7`), running in
`quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc` on 8×910B4 with
`/data02/GLM-5.3-Flash-w8a8-b0829` (W8A8, TP8).

```bash
# apply the core patches to a vLLM v0.32.0 tree
git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm.git vllm
cd vllm
git am /path/to/patches/flashstate_v032/*.patch

# serve (frontier on)
vllm serve /data02/GLM-5.3-Flash-w8a8-b0829 \
  --trust-remote-code --enable-prefix-caching \
  --mamba-cache-mode align --mamba-frontier-checkpoint \
  --max-model-len 20480 --max-num-seqs 8 --tensor-parallel-size 8 \
  --enforce-eager --prefix-cache-eviction lru
```

### Multi-turn benchmark (`bench_frontier.py`)

4 sessions × 3 turns × 128 output, 3K prompt:

| metric | frontier off | frontier on |
|---|---|---|
| later-turn latency (128 out) | 33.53s (n=7) | **31.72s** (n=8), −5.4% |
| frontier hits | 0 | **2,284 tokens** |
| prefix-cache hits | 22,400 | 25,324 |

```bash
python3 bench_frontier.py frontier_on
```

### vLLM bench (`prefix_repetition`, 16K input / 1K output / 8 concurrency)

```bash
vllm bench serve --backend openai \
  --model /data02/GLM-5.3-Flash-w8a8-b0829 \
  --base-url http://127.0.0.1:8000 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 15937 \
  --prefix-repetition-suffix-len 63 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 1024 \
  --num-prompts 8 --max-concurrency 8 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99
```

| run | duration (s) | output tok/s | TTFT mean / P90 | TPOT mean | E2EL mean |
|---|---|---|---|---|---|
| frontier-on (warm) | 256.25 | 31.97 | 3.10 / 3.35 s | 247.17 ms | 255.96 s |
| frontier-off (warm) | 262.41 | 31.22 | 8.05 / 9.45 s | 248.26 ms | 262.02 s |
| frontier-off (cold) | 356.46 | 22.98 | 92.05 / 92.37 s | 258.18 ms | 356.18 s |

`prefix_repetition` uses a 15,937-token prefix; the block-aligned hit is
15,360, and the frontier serves the remaining **577 tokens**. That is why TTFT
drops from 8.05 s to 3.10 s while E2EL (dominated by 1K decode) only drops 2.4%.
The cold run is the first-prefill cost and is not a fair pair.

### 30K context is not feasible on this hardware

`--max-model-len 32768 --max-num-seqs 8` exits 137 during startup: 38.31 GB of
weights per card plus 32K KV does not fit in 64 GB HBM. 20K is the largest
`max-model-len` that starts; the 16K-input benchmark above uses 20,480.

Raw outputs are in `results/` (`vllm_bench_on_prefix.txt`,
`vllm_bench_off_warm.txt`, `vllm_bench_off_prefix.txt`).

## WIP track: vllm-ascend main

`vllm_ascend/hybrid_checkpoint/` and
`vllm_ascend/patch/platform/patch_hybrid_checkpoint.py` are the runtime-patch
port for the vllm-ascend `main` branch (vLLM v0.30.0). It imports on vLLM 0.30
but has **not** been runtime-validated and does not produce the numbers above.
Use the vLLM v0.32.0 core patch series for reproduction until that port is
completed.