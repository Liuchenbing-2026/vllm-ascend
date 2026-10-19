# Clef on Ascend — integration candidate

This deployment candidate targets the original `Cloudflare/clef` 27B checkpoint,
revision `0b331204bb13fbd2ca93a64df1956f5b55478ce5`. It reuses vLLM's
configuration-driven `ClefFlashForJointSchema` and native
`/v1/systemone` service. It does not register a model in the Ascend plugin.
**NPU execution and accuracy have not been validated for this candidate.**

## Required sources

| Component | Source inspected | Status |
| --- | --- | --- |
| Inspected vLLM source | `83c9a547bf9a3b4f2653fb5766886bdc5dba35be` | Configuration detection, secondary head loading and joint request chain inspected; not a validated installed pair |
| Ascend candidate base | `ee29cc6fc24d20d47948776764059f05601170a6` | Deployment/test candidate only |
| Ascend verified vLLM | `c4df37dfcc7f72ca41cd43801a468526ecf99a2b` | Unchanged; not replaced by this recipe or claimed to support the required original-Clef chain |

The inspected vLLM snapshot and Ascend's declared dependencies are not an
installable, verified combination yet. A compatible installed pair is a
prerequisite to running the command below. This change does not alter dependency
pins or supply a ready-to-run image. The previously inspected rc2 and nightly
images did not provide the complete original-Clef joint-head/request chain.

The Ascend Dockerfile defaults to vLLM v0.31.0, while image build workflows
can override the tag with a commit. Check the actual import paths, source
commits and compiled extensions. Do not assume these two main snapshots
form a tested pair. Use an isolated container for subsequent device validation.

## Proposed eager entry point

For the original `Cloudflare/clef`, retain the
checkpoint's `joint_head_config.json` and `joint_head.safetensors` beside its
backbone weights. Do not force a plain Qwen architecture or a last-token
classification head: Clef reads question/option spans across the whole prompt.
The upstream class name does not select a different checkpoint: secondary
weights must come from the same original checkpoint/revision. Clef-Flash and
Clef-Omni model weights are outside this candidate's scope.

The launcher requires nonempty local backbone configuration and both head
files before invoking vLLM. It does not verify weight hashes or architecture
compatibility. Keep the download revision and file manifest separately. Its
eager, uncached, unchunked defaults keep each joint prompt together for initial
validation; they are not performance recommendations.

```bash
export MODEL_PATH='<LOCAL_CLEF_CHECKPOINT>'
export ASCEND_RT_VISIBLE_DEVICES='<ASSIGNED_NPU_IDS>'
export TENSOR_PARALLEL_SIZE='<TP_SIZE>'
bash examples/models/clef/serve.sh
```

Once a compatible runtime starts successfully, the intended entry point is
`/v1/systemone` on loopback. Example text request:

```bash
curl --fail http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"clef","state":"The customer was charged twice for one order.","questions":{"team":{"type":"choice","instructions":"Which team handles this?","criteria":{"billing":null,"shipping":null}},"duplicate":{"type":"noul","instructions":"Was the customer charged twice?"},"severity":{"type":"score","instructions":"Rate the payment issue.","criteria":["none","minor","major"]}}}'
```

## Device acceptance still required

`tests/ut/test_clef_recipe.py` checks process arguments and rejects incomplete
local checkpoints before launch. It uses a capture executable, not vLLM or
model weights. Full installed-framework import, HTTP execution and real NPU
inference remain unverified.

Compare against the fixed checkpoint's reference implementation using identical
prompts, options, temperature and inputs. Check all three question types,
mixed multi-question requests, reordered options, malformed schemas and long
inputs. Record task accuracy separately from maximum probability difference.
For images, compare the actual processed image/token inputs before comparing
outputs. Passing text evaluation does not establish image support.

Neither graph replay nor model accuracy is claimed by this launcher. A causal
Clef joint head is not interchangeable with StartLux Instant's bidirectional
last-position readout; do not apply the StartLux attention override to Clef.
Publish accuracy only after device verification; keep timing results private.
