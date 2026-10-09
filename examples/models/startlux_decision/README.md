# StartLux-Decision inference on Ascend

This recipe uses native vLLM classification models for the dense 4B and MoE
35B-A3B checkpoints. The models inherit the Qwen3.5 implementations in vLLM;
Ascend supplies its existing attention and MoE operators. No model class is
registered or patched in the hardware plugin.

## Revisions and environment

- vLLM fork branch: `Liuchenbing-2026/vllm:startlux_decision_npu_eager`, based on
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (v0.30.0).
- Ascend fork branch: `Liuchenbing-2026/vllm-ascend:startlux_decision_npu_eager`, based
  on `a8fcedb03d93e60efceddbfc912406f7fa491d57` with the existing packed-GDN
  normalization change `eb68728c5989645edcd5f43db82d711ff7412aef`.
- The Ascend Dockerfile and `.github/vllm-main-verified.commit` specify the
  above vLLM base. Use paired revisions, not arbitrary latest branches.
- Base image: `quay.io/ascend/vllm-ascend@sha256:2c8aac4281e56953764a7fa60773cec734342d32d9d6d76c0c8f3a8660a64b85`.
- Official prompt and evaluation source:
  `StartLuxLabs/StartLux-Decision@0e7a2e81b9c92756e26d8edd843a44d50e362669`.
- Checkpoints: `startlux-models/StartLux-Decision-4B` and
  `startlux-models/StartLux-Decision-35B-A3B` on Hugging Face, or the same
  checkpoint names under `StartLuxAI` on ModelScope. Preserve model licenses.
- The tested ModelScope weight revisions are
  `9113683823edce9311109fff84cf9719394835d8` (4B) and
  `e6dafa13615ba6858e9489416216a9b46c36e45c` (35B-A3B).
- Runtime: Python 3.12.13, PyTorch 2.10.0+cpu, torch-npu 2.10.0.post4,
  Transformers 5.14.1 and triton-ascend 3.2.2. Check actual import paths:
  installed package metadata can describe the image rather than mounted source.

Create a new container and a virtual environment for this recipe. Install the
paired editable repositories following the Ascend installation guide. The
vLLM install uses `VLLM_TARGET_DEVICE=empty`; build Ascend C++ extensions when
switching away from the compiled revision supplied by the image. Installing
CUDA dependencies from the official StartLux requirements is unnecessary:
only its renderer and answer protocol are imported.

The initial experiment reused matching, previously built Ascend extensions.
Those binaries differ from the base image's plugin binaries. On a fresh machine,
build the selected Ascend checkout; do not mix its Python sources with arbitrary
image extensions. No C++ or operator source is changed by this recipe.

## Run

Set paths to the checkouts and downloaded checkpoint inside your container:

```bash
export MODEL_PATH=/models/StartLux-Decision-4B
export VLLM_SOURCE=/workspace/vllm
export STARTLUX_SOURCE=/workspace/StartLux-Decision
export PYTHON_BIN=/workspace/.venv/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
bash examples/models/startlux_decision/serve.sh --enforce-eager
```

For the 35B-A3B checkpoint, set `TENSOR_PARALLEL_SIZE=4` and select four assigned
NPUs. The default HTTP listener is loopback. Configure `PORT` to run separate
services. Begin with eager execution; removing `--enforce-eager` must be
validated against eager outputs before making graph-mode accuracy claims.

The launcher defaults the existing vLLM setting
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. In-process scheduling queues the questions
before executing the batch. The background engine can execute the first question
before the rest arrive, causing additional forward passes. Set the variable to
`1` to reproduce that baseline. Tensor parallel workers remain separate when
TP is greater than one. Recheck both probabilities and latency when changing
scheduling or graph shapes.

### Exact token-count piecewise graphs

The separate `startlux_decision_npu_graph` branch pairs this recipe with
vLLM commit `ed87be591604ee6201b764c3acdee01ad0c12b1b`, based on the eager
revision above. It adds an opt-in vLLM dispatch policy: captured token counts
use graphs, while other counts run without graph padding or replay. The default
vLLM padding policy is unchanged. Ascend requires no new model or runner patch.

For 4B, use the same environment and checkpoint as the eager baseline:

```bash
bash examples/models/startlux_decision/serve.sh --compilation-config \
  '{"mode":3,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[256,384,512,768,891,1024,2048,4096],"cudagraph_allow_padding":false}'
```

These capture sizes are deployment settings, not model constants. The 891-token
entry corresponds to the pinned three-field benchmark request. Keep in-process
scheduling enabled to preserve the batching used during validation. Do not add
`--enforce-eager` to this graph command.

The 4B full seven-suite evaluation and per-decision comparison with the frozen
eager baseline completed at vLLM `31fbeda5a868bcf654d2712e858fe424b2f98d27`.
The follow-up `ed87be59` only adds a guard against returning eager when the caller
explicitly forbids it, with four failing-before/passing-after unit cases. All
20 dispatcher unit cases passed; the full model suite has not been rerun at
this follow-up revision. The acceptance contract requires maximum absolute
probability error at most 0.001 and no changed argmax decisions. Runtime logs
also confirmed actual ACL graph replay. This is selective piecewise replay,
not a claim that every request uses a graph. The 35B graph configuration still
requires its own full evaluation; retain eager for that model until verified.

### Stable tensor-parallel comparisons

Before comparing 35B TP4 eager and graph outputs, first repeat identical requests
in the same eager service. The default communication configuration produced
different probabilities on repeated requests in this environment. These existing
determinism settings restored identical probabilities on the fixed 56-request,
88-decision subset:

```bash
export HCCL_DETERMINISTIC=true
export LCCL_DETERMINISTIC=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Apply the same settings to both modes. They follow the existing Ascend
[accuracy troubleshooting guidance](../../../docs/source/faqs.md), without
changing model weights or adding environment variables. The subset is a
repeatability check, not full-suite graph acceptance. Preserve the earlier
predictions, record the configuration change, and collect a fresh complete
eager reference before the full graph comparison. The absolute probability
tolerance of 0.001 and zero changed decisions remain unchanged. Compare latency
under the same determinism settings as well.

The launcher argument contract is covered by
`tests/ut/test_startlux_recipe.py`, including both eager and graph flags.

```bash
curl -s http://127.0.0.1:18190/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"Paris is the capital of France.","questions":{"city":{"type":"choice","instructions":"Which city is the capital of France?","criteria":{"Paris":null,"Berlin":null}}}}'
```

Each question is rendered independently with the official thinking-off prompt.
The native pooler returns the last token's raw A–Z logits, with the selected
output-head rows projected in FP32. The reference protocol then restricts the
candidates, applies checkpoint-specific temperatures, and returns probabilities.
It also retains score ordering, yes/no semantics and wide-choice tournaments.
This text example explicitly rejects images; image and 256K-context support
are not established by text-suite validation.

## Validate

Run the focused model tests from the vLLM checkout:

```bash
.venv/bin/python -m pytest tests/models/multimodal/pooling/test_startlux_decision.py
```

Use the official pinned evaluation bundle, without changing labels or prompts:

```bash
cd "$STARTLUX_SOURCE"
bash eval/fetch_benchmarks.sh
"$PYTHON_BIN" eval/suites.py predict --endpoint http://127.0.0.1:18190 \
  --out outputs/startlux-npu
"$PYTHON_BIN" eval/suites.py score outputs/startlux-npu \
  --json outputs/startlux-npu-metrics.json
"$PYTHON_BIN" eval/latency.py http://127.0.0.1:18190/v1/systemone 200
```

The latency script warms up 20 times, then times 200 serial HTTP requests, each
with choice, yes/no and score fields. Report its actual processed token count;
request-content tokens and separately rendered question tokens are different.
Before reporting the seven-suite average, require all 12,351 decisions, zero
missing rows and zero invalid results. Use a fresh output directory for each run.

Keep full predictions, final metrics, exact weight hashes and timing evidence
in the local experiment archive. Serving startup and CPU unit tests alone do
not establish seven-suite accuracy, tensor-parallel equivalence or latency.
