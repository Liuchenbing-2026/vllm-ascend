# Quantization Guide

Model quantization is a technique that reduces model size and computational overhead by lowering the numerical precision of weights and activations, thereby saving memory and improving inference speed.

`vLLM Ascend` supports multiple quantization methods. This guide provides instructions for using different quantization tools and running quantized models on vLLM Ascend.

> **Note**
>
> You can choose to convert the model yourself or use the quantized model we uploaded.
> See <https://www.modelscope.cn/models/vllm-ascend/Kimi-K2-Instruct-W8A8>.
> Before you quantize a model, ensure sufficient RAM is available.

## Quantization Tools

vLLM Ascend supports models quantized by two main tools: `ModelSlim` and `LLM-Compressor`.

### 1. ModelSlim (Recommended)

[ModelSlim](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) is an Ascend-friendly compression tool focused on acceleration, using compression techniques, and built for Ascend hardware. It includes a series of inference optimization technologies such as quantization and compression, aiming to accelerate large language dense models, MoE models, multimodal understanding models, multimodal generation models, etc.

#### Installation

To use ModelSlim for model quantization, install it from its [Git repository](https://gitcode.com/Ascend/msmodelslim):

```bash
# Install 26.0.0 version, this is currently the latest stable branch
git clone https://gitcode.com/Ascend/msmodelslim.git -b 26.0.0

cd msmodelslim

bash install.sh
```

#### Model Quantization

The following example shows how to generate W8A8 quantized weights for the [Qwen3-MoE model](https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen3-MOE/README.md).

**Quantization Script:**

```bash
cd example/Qwen3-MOE

# Support multi-card quantization
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

# Set model and save paths
export MODEL_PATH="/path/to/your/model"
export SAVE_PATH="/path/to/your/quantized_model"

# Run quantization script
python3 quant_qwen_moe_w8a8.py --model_path $MODEL_PATH \
--save_path $SAVE_PATH \
--anti_dataset ../common/qwen3-moe_anti_prompt_50.json \
--calib_dataset ../common/qwen3-moe_calib_prompt_50.json \
--trust_remote_code True
```

After quantization completes, the output directory will contain the quantized model files.

For more examples, refer to the [official examples](https://gitcode.com/Ascend/msmodelslim/tree/master/example).

### 2. LLM-Compressor

[LLM-Compressor](https://github.com/vllm-project/llm-compressor) is a unified compressed model library for faster vLLM inference.

#### Installation

```bash
pip install llmcompressor
```

#### Model Quantization

`LLM-Compressor` provides various quantization scheme examples.

##### Dense Quantization

An example to generate W8A8 dynamic quantized weights for dense model:

```bash
# Navigate to LLM-Compressor examples directory
cd examples/quantization/llm-compressor

# Run quantization script
python3 w8a8_int8_dynamic.py
```

##### MoE Quantization

An example to generate W8A8 dynamic quantized weights for MoE model:

```bash
# Navigate to LLM-Compressor examples directory
cd examples/quantization/llm-compressor

# Run quantization script
python3 w8a8_int8_dynamic_moe.py
```

For more content, refer to the [official examples](https://github.com/vllm-project/llm-compressor/tree/main/examples).

The quantization types currently supported by LLM-Compressor can be viewed in the `vllm_ascend/quantization/compressed_tensors_config.py` file.

### 3. Native FP8 checkpoints (no quantization step needed)

Many models are now published directly in FP8, for example [Qwen/Qwen3.8-27B-FP8](https://modelscope.cn/models/Qwen/Qwen3.8-27B-FP8) and [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). Their `config.json` carries `"quant_method": "fp8"` together with a `weight_block_size`, meaning every quantized weight is stored as `float8_e4m3fn` plus one `float32` scale per weight block. vLLM Ascend detects and serves these checkpoints as published, so no offline re-quantization is required:

```bash
vllm serve /path/to/Qwen3.8-27B-FP8 --trust-remote-code
vllm serve /path/to/GLM-5.3-Flash --tensor-parallel-size 8 --trust-remote-code
```

How the weights are executed depends on the hardware:

- **950PR&950DT Products**: the block scales are re-grouped into MXFP8 at load time, so weights stay at one byte per element and the native FP8 matmul is used.
- **Other Ascend generations**: the block scales are resolved into the model dtype at load time and served by the BF16 matmul. Numerically equivalent to the checkpoint, but plan for roughly twice the weight memory.

Layers the checkpoint left unquantized, such as `visual.merger.*` on multimodal models, are listed in `ignored_layers` or `modules_to_not_convert` and are served unquantized.

Only the block-wise flavour is supported. A native FP8 checkpoint without `weight_block_size` (per-tensor or per-channel scales) has no Ascend execution path yet; re-quantize it with ModelSlim or LLM-Compressor instead.

## Running Quantized Models

Once you have a quantized model which is generated by **ModelSlim**, you can use vLLM Ascend for inference by specifying the `--quantization ascend` parameter to enable quantization features, while for models quantized by **LLM-Compressor**, it is not necessary to add this parameter.

### Offline Inference

```python
import torch

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
# Set sampling parameters
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="/path/to/your/quantized_model",
          max_model_len=4096,
          trust_remote_code=True,
          # Set appropriate TP and DP values
          tensor_parallel_size=2,
          data_parallel_size=1,
          # Set an unused port
          port=8000,
          # Set serving model name
          served_model_name="quantized_model",
          # Specify `quantization="ascend"` to enable quantization for models quantized by ModelSlim
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### Online Inference

```bash
# Corresponding to offline inference
python -m vllm.entrypoints.api_server \
    --model /path/to/your/quantized_model \
    --max-model-len 4096 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --served-model-name quantized_model \
    --trust-remote-code 
```

## MXFP8 on Ascend 950

Ascend 950PR/950DT (A5) executes MXFP8 matrix multiplication natively. In an MXFP8 checkpoint, weights and activations are `float8_e4m3fn` and each 32-element group along the reduction dimension carries one E8M0 scale. ModelSlim records this recipe as `W8A8_MXFP8` in `quant_model_description.json`, so a valid checkpoint contains entries such as:

```json
{
    "model.language_model.layers.0.self_attn.q_proj.weight": "W8A8_MXFP8",
    "model.language_model.layers.0.self_attn.q_proj.weight_scale": "W8A8_MXFP8"
}
```

The checkpoint stores roughly one byte per weight plus the E8M0 scale tensors.

### Obtaining MXFP8 weights

Quantize the BF16 model with the ModelSlim MXFP8 recipe, or use a published checkpoint:

- [Eco-Tech/Qwen3.8-27B-w8a8-mxfp8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8-mxfp8)
- [Eco-Tech/Qwen3.6-35B-A3B-w8a8-mxfp8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8-mxfp8)

Do not mix the weights of one checkpoint with the scale files of another: the `W8A8_MXFP8` description and the `weight_scale` tensors must come from the same ModelSlim export.

### Serving an MXFP8 checkpoint

ModelSlim checkpoints require `--quantization ascend`:

```bash
vllm serve /path/to/Qwen3.8-27B-w8a8-mxfp8 \
    --trust-remote-code \
    --quantization ascend \
    --served-model-name qwen3.8 \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

This serves the MXFP8 weights with the default KV cache dtype. The checkpoint above does not carry an MXFP8 KV recipe, so this is the supported configuration for it. On a single 950PR card with 128 GB HBM, this configuration uses about 34 GiB for weights and reports about 73 GiB of available KV cache at `--max-model-len 65536`.

#### Optional: MXFP8 KV cache

`--kv-cache-dtype mxfp8` stores the KV cache in MXFP8 as well, but it requires two things beyond the weight-only path:

1. The checkpoint must carry the MXFP8 KV recipe: `quant_model_description.json` must contain the `fa_v.scale` / `v_cache_scale` entries. `Eco-Tech/Qwen3.8-27B-w8a8-mxfp8` does not, so serving it with `--kv-cache-dtype mxfp8` only logs `V scales fall back to the neutral default` and does not give a correct MXFP8 cache.
2. The installed `cann_ops_transformer` package must provide the QFA metadata op. Check it before starting:

```bash
python -c "from cann_ops_transformer.ops import quant_flash_attn_metadata"
```

If the import fails, the engine aborts during profiling with `ImportError: cannot import name 'quant_flash_attn_metadata'`; install a CANN build that ships this op first.

When both conditions hold:

```bash
vllm serve /path/to/<mxfp8-kv-checkpoint> \
    --trust-remote-code \
    --quantization ascend \
    --kv-cache-dtype mxfp8 \
    --served-model-name qwen3.8 \
    --tensor-parallel-size 1 \
    --max-model-len 65536
```

This selects the C8-MXFP attention path. K is quantized per token group with dynamic E8M0 scales; V uses the per-channel scale from the checkpoint. The path is installed per full-attention layer, so hybrid models keep their remaining layers on the configured default cache policy.

The MXFP8 KV cache is allocated in 512-token kernel blocks. When `--block-size` is omitted, vLLM Ascend sets it to 512 automatically; an explicit value must be a multiple of 512.

### Supported models

The following configurations are documented and tested in this repository:

| Model | Hardware | Notes |
|---|---|---|
| Qwen3.8-27B-w8a8-mxfp8 | 950PR/950DT | MXFP8 weights only; the checkpoint has no MXFP8 KV recipe, so serve it without `--kv-cache-dtype mxfp8`. See the [model tutorial](../../tutorials/models/Qwen3.8-27B.md). On 950PR, the W8A8 INT8 checkpoint is currently the recommended weight. |
| MiniMax-M3 | 950DT | MXFP8; text/image/video |
| DeepSeek V4-Flash / V4-Pro | 950DT | Native mixed MXFP8/MXFP4 weights |

For a model that is not listed, use its BF16 or W8A8 INT8 checkpoint until its tutorial lists an MXFP8 variant.

### Limitations

- Native MXFP8 execution is available on Ascend 950PR/950DT. On A2/A3, use W8A8 INT8 or BF16.
- `--kv-cache-dtype mxfp8` additionally requires the checkpoint's MXFP8 KV recipe and a `cann_ops_transformer` build that exports `quant_flash_attn_metadata`; the weight-only MXFP8 path does not.
- `--kv-cache-dtype mxfp8` currently supports full-attention layers only. It does not support PCP/DCP, sliding-window attention, non-causal attention, or hamming sparse KV compression.
- PD disaggregation with an MXFP8 KV cache requires `MooncakeConnectorV1`, which transfers the FP8 K/V tensors together with both E8M0 scale caches.
- W4A8/W4A4 MXFP4 checkpoints use a different weight layout. Follow the per-model tutorial instead of combining their scale files with MXFP8 weights.

### Quick check

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
```

A successful response confirms that the MXFP8 weights, the E8M0 scales and, if enabled, the MXFP8 KV cache were loaded.

## References

- [ModelSlim GitCode](https://gitcode.com/Ascend/msmodelslim)
- [LLM-Compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/features/quantization/)
