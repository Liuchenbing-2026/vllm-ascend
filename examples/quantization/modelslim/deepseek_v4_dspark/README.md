# DeepSeek V4 DSpark full W8A8 quantization

This directory contains the reproducible ModelSlim recipe used to quantize the
complete DeepSeek V4 target plus DSpark draft checkpoint. It produces dynamic
per-token INT8 activations and per-channel INT8 weights for the selected
attention, FFN, and DSpark `main_proj` linears.

The end-to-end wrapper also finalizes the QuaRot basis and shared output head.
Use the finalized checkpoint for serving; the raw ModelSlim directory is an
intermediate artifact.

## Prerequisites

- Ascend toolkit environment sourced before the run.
- ModelSlim available as `msmodelslim` in the active Python environment.
- ModelSlim must recognize the model type `DeepSeek-V4-Flash-DSpark`.
- A compatible target/MTP W8A8 checkpoint that contains the canonical
  `head.weight` with the same dtype and shape as the DSpark checkpoint.
- Enough storage for the raw checkpoint and one private patched shard. The
  default finalized output symlinks all other files to the raw output.

The verified machine run used ModelSlim `26.0.0a1` at source revision
`9d8f50a31de27811f120806f78d6c1166b5278b7`. Its DSpark adapter was a local,
untracked ModelSlim extension. That extension is deliberately not vendored
here: the old local implementation declared that model code had been ported
from another implementation. Supply a DSpark adapter that you own or whose
license and provenance have been reviewed.

## Run

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /path/to/modelslim-venv/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=0

bash examples/quantization/modelslim/deepseek_v4_dspark/quantize_w8a8.sh \
  /models/DeepSeek-V4-Flash-DSpark \
  /models/DeepSeek-V4-Flash-DSpark-w8a8-raw \
  /models/DeepSeek-V4-Flash-w8a8-mtp \
  /models/DeepSeek-V4-Flash-DSpark-w8a8-final
```

Pass `copy` as the fifth argument to create a self-contained final directory,
or `hardlink` when the raw and final directories share a filesystem. The
default `symlink` mode is storage-efficient, but the raw output must remain in
place.

The wrapper refuses to overwrite either output directory. It invokes the
equivalent full-model ModelSlim command:

```bash
msmodelslim quant \
  --model_path "$MODEL_PATH" \
  --save_path "$RAW_OUTPUT" \
  --model_type DeepSeek-V4-Flash-DSpark \
  --config_path deepseek_v4_flash_dspark_w8a8.yaml \
  --trust_remote_code True
```

## Why finalization is required

The recipe applies global QuaRot. The tested integration consumes the DSpark
decoder in the rotated basis while using a canonical shared output head. The
finalizer therefore:

1. verifies all indexed shards, `mtp.0.main_proj.weight=W8A8_DYNAMIC`, and the
   QuaRot artifact;
2. records `dspark_quarot_draft_basis=rotated_decoder` and
   `dspark_quarot_hc_head_basis=canonical` in `config.json`;
3. replaces only `head.weight` with the compatible canonical tensor using a
   bounded-memory streaming copy;
4. verifies the copied tensor byte-for-byte and writes
   `.dspark_w8a8_finalize.json`.

Serving the unfinalized output can cause a severe acceptance-rate regression
because decoder states and the output head are then interpreted in different
bases.

The recorded machine run completed successfully on 2026-06-29. Its source
recipe (semantically reproduced by the normalized YAML in this directory) had
SHA-256
`363b18d9839425b8e11e0e75b195c548a46a202b84398a9be3ac1a991cc3b0fc`.
Acceptance and `ais-bench` comparisons for the finalized checkpoint are under
[`benchmarks/evidence/pr11196-dspark-quant-20260713`](../../../../benchmarks/evidence/pr11196-dspark-quant-20260713/README.md).
