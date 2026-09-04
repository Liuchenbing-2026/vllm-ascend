# 04 —— 未解问题与已知限制

## 1. 多模态：模型是多模态的，但官方模板把图片吃掉了

**模型确实是多模态的**，官方模型卡原话：

> We introduce GLM-5.3-Flash, **the first natively multimodal model in the GLM-5 series** … our
> latest **30T-token multimodal pre-training corpus**

**除了对话模板，整条链路是好的**（以下经 3 视角对抗验证）：

- ViT 用 `quant_config=None` 构建，**结构上不可能被量化碰到**；347 个 `model.visual.*`
  经 `model.visual.` → `visual.` 一一对应装上 [F]
- **vLLM 启动时的 `profile_run` 已经在设备上用 bf16 跑过一次 `model.embed_multimodal(...)`**
  ——视觉塔是能加载、能跑的 [F]
- 文件不缺：transformers 5.16 优先读嵌套的 `processor_config.json`，
  它就是 legacy `preprocessor_config.json` 的现代替代 [F]
- **prefix cache 覆盖图像 token 是正确的**：`_gen_mm_extra_hash_keys` 会把
  `(内容哈希, 块内偏移)` 打进每一个与图像重叠的 640-token block [F]
- vllm-ascend 内置的 `Glm5NextImageProcessor` **总是**胜过 checkpoint 里 `auto_map` 的远程代码
  （`auto_map` / `image_processor_type` 两个键在 `glm5_next_multimodal.py:434-443` 被显式剥掉），
  `--trust-remote-code` 不改变这一点 [F]

### 唯一的 blocker 是模板

`chat_template.jinja` 的 `visible_text` 宏（第 50 行定义，第 59-61 行）把每个
image/video content part 映射成一句字面文本：

```jinja
{%- elif item is mapping and item.type in ['image','image_url','video',...] -%}
    {{- "<reminder>You are unable to process this " ~ media_type ~
        " because you don't have multi-modal input ability. Try different methods.</reminder>" }}
```

整个 8617 字节的模板里 `image` 只匹配到这一行，`<|image|>` / `<|begin_of_image|>` /
`<|end_of_image|>` **一次都没出现过**。

**为什么这个分支会被走到**：同一个宏里的 `{%- for item in content -%}` 循环让 vLLM 把内容格式
自动判成 `openai`（`renderers/hf.py:416-418`，日志里有
`Detected the chat template content format to be 'openai'`）。openai 格式下
`_parse_chat_message_content_part` 返回裸 dict `{"type":"image"}`，
模型自带的 `get_placeholder_str` **被丢弃** —— 占位符本该由模板发，而这个模板不发。

于是 `Glm4vMultiModalProcessor._get_prompt_updates`（`glm4_1v.py:1567-1571`）
要找的 `target = hf_processor.image_token = "<|image|>"` 找不到：

```
AssertionError: Failed to apply prompt replacement for mm_items['image'][0]
  vllm/multimodal/processing/processor.py:1565
```

（已在本机复现，见 `/data02/glm53_tiny/logs/real_full.log`，HTTP 500。）

### 三种修法，推荐第一种

1. **`--chat-template-content-format string`**（推荐，**不碰厂商模板**）
   内容以纯字符串到达模板，其中已由 `_get_full_multimodal_text_prompt`
   （`chat_utils.py:1355-1405`）拼好 `<|begin_of_image|><|image|><|end_of_image|>`，
   模板的 `content is string` 分支原样输出。
   想保留图片在原位而不是被前置，再加 `--interleave-mm-strings`。
2. `--chat-template <改过的模板>` —— 本仓库的 `serve/chat_template_mm.jinja` 就是这个
   （在原有 media 分支**之前**插入 image/video 两个分支，其余字节不动）。
   保留 openai 的交错语义。
3. 免重启的临时口子：把 Jinja 直接放进**单次请求**的 `chat_template` 字段
   （`renderers/hf.py:272-276` 里它优先级最高，且每请求重跑格式判定）。
   **注意**：这条路没有端到端跑过（[U]）——`resolve_chat_template` 是把裸 Jinja 串交给
   `tokenizer.get_chat_template()`，transformers 5.16 收到裸模板而非模板*名*时的行为只从
   vLLM 侧读了代码，没实测。

### 但没有启用

**这是在覆盖厂商明确关掉的开关。** 分不清是「这版不开放图像输入」还是「模板发错了」，
处理方式完全不同。要开先跟上游确认。

**状态：图像通路一次都没产出过正确答案（[U]）。** 视觉塔的语义正确性
（2×2 merge 块内的权重顺序、downsample/merger 的方向、`Glm5NextSiluAndMul` 在
swiglu_limit=10.0 处的截断）目前只由「profiling 时没崩」支撑，**不是由任何输出支撑的**。
修好模板后欠一次已知图像的对拍。

### 顺带记下的几条

- **`--mm-processor-kwargs` 是有效的**（我最初判断它对本模型无效，被 3/3 推翻）：
  服务级或单请求的 `max_image_tokens` / `min_image_tokens` 会作为 callable kwarg 传下去。
  另外 `--limit-mm-per-prompt '{"image":1,"video":0}'` 也有效。
- **mrope 没有启用** —— 图像 token 拿的是普通顺序位置。
- **MTP drafter 看不到图像 embedding**，图像 prefill 之后头几个 token 的接受率会低。
  **不要拿纯文本的 73% 去调 `num_speculative_tokens`**，图像流量要单独量。
- 视频完全没碰过。`Glm5NextVideoProcessor` 在（max_frames 2048 / fps 2 /
  max_image_tokens 240000），`get_supported_mm_limits` 允许 `video:1`，
  编码器预算按视频定尺寸（16384）——真实视频会是这台机器见过的最大单次编码任务。

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

## 5. 哪些结论经过对抗验证，哪些没有

经过 3 视角对抗验证的只有两块：

- **prefix cache**：16 条 finding，12 条 verdict 全部 `refuted=false / confidence=high`
- **多模态**：11 条 finding；blocker（模板吃掉占位符）3/3 确认；
  其中我原本判断的「`--mm-processor-kwargs` 对本模型无效」被 **3/3 推翻**，已改正

**没有**经过对抗验证的：图模式 / MTP / 量化加载 / 内存预算 / 交互矩阵。
这几节是单遍代码阅读 + 实测 —— 标 [F] 的都有实测或明确 file:line 支撑，标 [I] 的只是推断。

（第二轮审计原本还要查 QuaRot rot.weight、内存拆解、MTP conv state 三项，
但那三个维度的 agent 全部 stalled，所以第 3/4/6 节仍然是空的。）

## 6. QuaRot 的 rot.weight 只在 MTP 层被消费

`glm5_next.py:104` 的 `MTP_ROT_WEIGHT_NAME = "rot.weight"` 把它路由到 MTP 层。
主干模型是否需要对应的运行时旋转（还是导出时已经就地旋转、自洽）**没有独立确认**（[U]）。

风险形态：如果本该应用的旋转被静默跳过，模型仍能输出流畅文本、
只是精度悄悄退化 —— `17*23=391` 这类冒烟测试**抓不到**。
要抓需要拿同一批 prompt 对 BF16 与 w8a8 比 prefill top-1 logits，或跑标准评测集。
