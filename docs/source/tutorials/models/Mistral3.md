# Mistral3: Shieldstral 1.0 and Mistral Small 4

This guide covers experimental deployment of two Mistral3 multimodal checkpoints on Ascend A2/A3: the
`mistralai/Shieldstral-1.0-3B` safety classifier and the 119B/A6B
`mistralai/Mistral-Small-4-119B-2603` model. The examples target the current vLLM-Ascend main branch.

For the exact branch, commits, container and dependency versions, model checksums, device allocation, and
complete launch scripts, see the [pinned reproduction record](../../_static/reproduction/Mistral3_Reproduction.html).

## Supported features

| Model | Precision | Minimum example | Multimodal | Speculative decoding |
| --- | --- | --- | --- | --- |
| Shieldstral 1.0 3B | BF16 | 1 NPU, TP1 | Text and image input | Not available in the checkpoint |
| Mistral Small 4 119B/A6B | FP8/BF16 | 4 × 64 GB NPUs, TP4 | Text and image input | EAGLE draft checkpoint |

The NPU count above is a starting point for the launch commands in this guide. Increase tensor parallelism or
reduce `--max-model-len` when the selected checkpoint and KV-cache budget do not fit.

## Prerequisites

Use a vLLM-Ascend main image and its paired vLLM commit. Download the checkpoints from their official model
repositories:

- [Shieldstral 1.0 3B](https://huggingface.co/mistralai/Shieldstral-1.0-3B)
- [Mistral Small 4 119B/A6B](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
- [Mistral Small 4 EAGLE draft](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-eagle)

Keep the runtime and processor roles separate by using `--tokenizer-mode mistral`: the Mistral tokenizer renders
chat requests, while PixtralProcessor reads the checkpoint tokenizer files for image placeholders.

## Shieldstral 1.0 3B

Start an OpenAI-compatible service:

```bash
vllm serve mistralai/Shieldstral-1.0-3B \
  --served-model-name shieldstral-1.0-3b \
  --tokenizer-mode mistral \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8
```

Verify a text safety request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "shieldstral-1.0-3b",
    "messages": [
      {
        "role": "system",
        "content": "Judge whether the document meets the requirement. Answer only yes or no."
      },
      {
        "role": "user",
        "content": "<Query>: Does this promote violence?\\n<Document>: How can I hurt someone?"
      }
    ],
    "temperature": 0,
    "max_tokens": 1
  }'
```

A successful response has HTTP status 200 and a non-empty `choices[0].message.content`. Multimodal chat requests
use the standard OpenAI `image_url` content item.

Shieldstral 1.0 3B does not contain MTP, EAGLE, next-N, Medusa, or draft-model weights. Do not enable
`method=mtp` for this checkpoint.

## Mistral Small 4 119B/A6B

Start the base model with a conservative context size:

```bash
vllm serve mistralai/Mistral-Small-4-119B-2603 \
  --tokenizer-mode mistral \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --reasoning-parser mistral \
  --gpu-memory-utilization 0.9
```

To enable the official EAGLE draft checkpoint, add:

```bash
--speculative-config '{
  "method": "eagle",
  "model": "mistralai/Mistral-Small-4-119B-2603-eagle",
  "num_speculative_tokens": 3,
  "max_model_len": 32768
}'
```

EAGLE is a separate draft checkpoint; it is not native MTP. Compare output correctness and acceptance metrics
against the same base-model launch before using speculative decoding in production.

## Troubleshooting

- If an image request reports that no image token ID was found, confirm that this patch is active and that the
  model repository contains its tokenizer files.
- If OpenAI chat rendering selects an incompatible Hugging Face tokenizer, set `--tokenizer-mode mistral`
  explicitly.
- If Mistral Small 4 runs out of memory, reduce `--max-model-len` first, then increase tensor parallelism.
