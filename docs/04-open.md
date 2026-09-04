# 04 —— 未解问题与已知限制

## 1. 多模态：模型是多模态的，但官方模板把图片关掉了

这是本次最需要上游确认的一条。

**模型确实是多模态的**，官方模型卡原话：

> We introduce GLM-5.3-Flash, **the first natively multimodal model in the GLM-5 series** … our
> latest **30T-token multimodal pre-training corpus**

产物侧也全部对得上：`vision_config`（24 blocks / patch 14 / out_hidden 4096）、
347 个 `model.visual.*` 张量、`image_token_id 154854` 与 `<|begin_of_image|>` 等真实 token、
随包发的 `image_processing_glm5_next.py` + `processor_config.json`；
vllm-ascend 也把 `Glm5NextForConditionalGeneration` 注册到 706 行的 `glm5_next_multimodal.py`，
服务启动时解析出的架构就是它。

**但官方随包的 `chat_template.jinja` 是纯文本模板**，遇到 image content part 直接拒绝：

```jinja
{%- elif item is mapping and item.type in ['image','image_url','video',...] -%}
    {%- set media_type = item.type | replace('_url','') | replace('input_','') -%}
    {{- "<reminder>You are unable to process this " ~ media_type ~
        " because you don't have multi-modal input ability. Try different methods.</reminder>" }}
```

而且 README frontmatter 写的是 `pipeline_tag: text-generation`。

后果：模板不发 `<|image|>` 占位符，
`Glm4vMultiModalProcessor._get_prompt_updates` 的
`PromptReplacement(target=hf_processor.image_token, ...)` 找不到替换位置：

```
AssertionError: Failed to apply prompt replacement for mm_items['image'][0]
  vllm/multimodal/processing/processor.py:1565 _apply_prompt_updates
```

### 我们准备了什么

`serve/chat_template_mm.jinja` —— 从官方模板派生，**只改那一个分支**，
改成发 `<|begin_of_image|><|image|><|end_of_image|>` / `<|begin_of_video|><|video|><|end_of_video|>`，
其余字节完全一致（`serve/make_mm_chat_template.py` 可复现）。

已离线验证渲染正确：

```
[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|begin_of_image|><|image|><|end_of_image|>What colour is this?<|assistant|><think>
<|image|> -> [154854]   <|begin_of_image|> -> [154830]   <|end_of_image|> -> [154831]
```

### 但没有启用，也不建议自作主张启用

**这是在覆盖厂商明确关掉的开关。** 分不清是「这版不开放图像输入」还是「模板发错了」，
两种可能的处理方式完全不同。要用先跟上游确认。

用法（确认之后）：给 `serve.sh` 加一行
`--chat-template /path/to/chat_template_mm.jinja`。

**状态：图像通路一次都没跑通过（[U]）。** ViT 权重在、代码在、模板备好了，仅此而已。

## 2. b0829 vs b0829se 的 A/B 没做

`b0829`（共享专家量化）和 `b0829se`（共享专家保持 BF16）是干净的单变量对照，
但**精度和性能都没测**。共享专家在每个 token 上都参与计算，
量化它省 ~1 GB 权重、但可能影响精度 —— 值不值得只能实测。

需要：同一套评测集跑两遍，比精度；同一套 bench 跑两遍，比吞吐/TTFT。

## 3. MTP 的 conv state 差 6144 B

`ops/kimi_kda_state.py:29` 的 `get_mamba_state_shape_from_config` 与
`models/glm5_next.py:1698` 的逐层 `get_state_shape` 在 MTP 下对 KDA conv state 长度不一致，
前者多算 6144 B。

对 APC 无害（只用来定 page 大小，且被 `page_size_padded` 盖住），
但意味着**真实 KDA conv state 可能没有给投机解码留回滚槽位**。
症状要在「硬崩 / 拒绝后输出错 / 悄悄降低接受率 / 无影响」之间判别。
实测接受率 73% 说明至少不是灾难性的，但没有排除「本可以更高」。

## 4. 内存预算没调优

TP8 + EP + `gpu-memory-utilization 0.9` 下 KV 只有 **14.11 GiB / 337,547 tokens**，
每卡 59 GB 里权重占 ~39 GB。剩下 20 GB 的去向没有逐项拆解过。

已知一个反直觉的约束（[F]，实测）：
KDA prefill 的 `recurrent_state[state_indices]` gather（`ops/kimi_kda.py:427`）开销
**随状态缓存规模走**，`gpu-memory-utilization` 吃太满会自己撑死自己 ——
迷你模型 TP2 上 0.85 时单次 6-token prompt 就要 20.13 GiB 而 OOM，0.35 通过。
真权重 TP8 上 0.9 目前没出事，但**没有在长 prompt / 高并发下压过**，
不能保证生产负载下安全。

没试过的省内存旋钮：`finegrained_tp_config`（oproj / lmhead / embedding / mlp 各自的 TP）、
`enable_shared_expert_dp`、`--kv-cache-dtype`、`enable_kv_nz`。

## 5. 六个分析维度未经对抗验证

`docs/03-findings.md` 里只有 prefix cache 一节经过 3 视角对抗验证
（12 条 verdict 全部 `refuted=false / confidence=high`）。
图模式 / MTP / 量化加载 / 多模态 / 内存 / 交互矩阵这几节是单遍代码阅读 + 实测，
其中标 [F] 的都有实测或明确的 file:line 支撑，标 [I] 的只是推断。

## 6. QuaRot 的 rot.weight 只在 MTP 层被消费

`glm5_next.py:104` 的 `MTP_ROT_WEIGHT_NAME = "rot.weight"` 把它路由到 MTP 层。
主干模型是否需要对应的运行时旋转（还是导出时已经就地旋转、自洽）**没有独立确认**（[U]）。

风险形态：如果本该应用的旋转被静默跳过，模型仍能输出流畅文本、
只是精度悄悄退化 —— `17*23=391` 这类冒烟测试**抓不到**。
要抓需要拿同一批 prompt 对 BF16 与 w8a8 比 prefill top-1 logits，或跑标准评测集。
