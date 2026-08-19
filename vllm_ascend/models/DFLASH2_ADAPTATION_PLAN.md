# DFlash 2 → vllm-ascend 适配方案（差异分析）

> 日期：2026-08-19
> 基线：本地树 `C:\Users\wzy85\vllm-ascend`（分支 `agent/glm52-dspark-investigation`，vLLM v0.23.0 + DSpark backport）
> 上游：vLLM PR [#52816](https://github.com/vllm-project/vllm/pull/52816)（head `19c9351904df4c63042671bc67a866ca48dc7d6f`，755 行，未合并）
> 参考：SGLang PR [#35371](https://github.com/sgl-project/sglang/pull/35371)、DFlash 2 官方博客（inco.ai/blog/dflash2）

## 1. 上游改动面（vLLM #52816）

三个层次，共 11 个文件：

| 层 | 文件 | 内容 |
|---|---|---|
| 模型 | `vllm/model_executor/models/qwen3_dflash2.py`（+346） | `DFlashGroupedConv`（两抽头动态深度卷积）、`DFlash2Qwen3DecoderLayer`（attention/mlp 各包一层 conv）、`CandidateSelector`（低秩双线性路径打分）、`compute_candidates`（top-k + TP all-gather + output_multiplier + softcap）、`input_embedding_scale` |
| Worker | `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`（+224） | `DFlash2Speculator(DFlashSpeculator)`：`_generate_draft` 尾部跑 `compute_candidates` → `candidate_selector` → `_sample_path`（Triton `_selector_walk_kernel`）→ `_cache_draft_logits`（Triton） |
| 配置 | registry / `VllmConfig._is_dflash2_draft` / `use_v2_model_runner` | 注册 `DFlash2DraftModel` 架构；DFlash2 草稿强制 V2 runner（selector 只在 V2 speculator 里，V1 会静默降级为 DFlash1） |
| 基类微调 | `qwen3_dflash.py`（+15 -4） | `decoder_layer_cls` / `model_cls` 类属性化；`_dflash_layer_causal` 支持顶层 `is_causal` 覆盖 |

关键特性：**全部为 Python + Triton，无新 C++ 算子**。

## 2. 计算语义（移植必须逐点对齐）

### 2.1 分组动态深度卷积 `_grouped_conv`
```
out[i,c] = Σ_t (base[t,c] + delta[i,t,g(c)]) * x[i-t,c]
```
- taps=2（`conv_kernel_size`），`group_size`=`conv_group_size`（16 通道共享一个 delta 修正）
- 块边界处 tap 归零（块内第 0 位只乘自身系数；乘子掩码 `position >= tap`）
- `DFlashGroupedConv.prepare/finish`：kernel_projection 一次投影出两侧 delta；base_kernel 是学习参数 `[2, taps, hidden]`
- block_size = 1 + num_speculative_tokens

### 2.2 候选路径打分 `_score_edges`
```
edge(p→c) = <A[p] * project(h), B[c]> + unary[c]
```
- predecessor/successor codebook：`[vocab, rank]` 参数表；hidden 投影 `[hidden→rank]`
- 每个位置取目标 lm_head 的 **top-K 候选**（K=selector_top_k），锚点=上一步验证通过的 token
- 后续 walk：T=0 贪心取 max；T>0 用逆 CDF（Gumbel）采样，返回 q 分布供 lossless rejection sampling

### 2.3 路径 walk（上游 Triton 单程序/请求）
- 每请求一个 program，K 个分数常驻寄存器，slot 间依赖是 program 内循环（避免每步一个 kernel）
- top-k：FlashInfer radix（CUDA），否则 `torch.topk`（Ascend 直接走此分支）

## 3. vllm-ascend v0.23 树的映射（差异清单）

| 上游（#52816） | vllm-ascend v0.23 对应物 | 适配动作 |
|---|---|---|
| `DFlash2Qwen3ForCausalLM` 等模型类 | `vllm_ascend/models/qwen3_dspark.py` 同款模式（自有模型文件 + 上游 qwen3_dflash 基类） | 新建 `vllm_ascend/models/qwen3_dflash2.py` |
| 架构注册 `DFlash2DraftModel` | vllm 0.23 registry + vllm-ascend 现有 dspark 注册先例（patch_speculators_dspark.py） | 新建 `vllm_ascend/patch/platform/patch_speculators_dflash2.py` |
| `DFlash2Speculator`（V1 gpu speculator 子类） | `vllm_ascend/spec_decode/dflash_proposer.py`（AscendDflashProposer，MRV1 proposer） | 新建 `AscendDflash2Proposer` 并接入 `get_spec_decode_method` |
| 强制 V2 runner | v0.23 无 V2 runner；Ascend 用 proposer 分发 | 按 draft architecture `DFlash2DraftModel` 分发到 AscendDflash2Proposer（对等语义） |
| FlashInfer top-k | 不可用 | 上游已备 `torch.topk` 回退；Ascend 固定走 topk |
| Triton `_selector_walk_kernel`（用 vllm gumbel `tl_rand32/64`） | triton-ascend 3.2.1；gumbel 工具为 CUDA 语义 | 首版用 torch 向量化参考实现（正确性优先），Triton 优化留 v2；预留接口 |
| `_cache_draft_logits_kernel` scatter | 纯 scatter | torch 实现 |

## 4. Ascend 侧风险点（已知）

1. **walk 的 host 下发**：torch 版按 step 循环（≤16 步、每步 [num_reqs, top_k] 小张量）会产生多次小算子下发，热路径压力大。缓解：优先尝试 triton-ascend 移植（去掉 randint，gumbel 噪声改为预生成 + 索引）；初版正确性优先，性能版做单算子 A/B 后再定。
2. **draft_logits 内存**：[max_reqs, steps, vocab] fp32；Qwen3.8 vocab≈150k，256 请求 × 8 步 ≈ 1.2GB。与上游一致，接受；可在 v2 优化为稀疏 q 分布。
3. **unquantized lm_head 硬要求**：DFlash2 需要未量化目标 lm_head 做 top-k；Ascend 侧 Modelslim 量化目标模型需确认 lm_head 是否保持 fp16/bf16（GLM-5.2-w4a8 类目标模型需单独验证，可能限跑）。
4. **TP 下 top-k 两段式**：`compute_candidates` 先各 rank top-k 再 all-gather 再全局 top-k；Ascend HCCL all-gather 可用（vllm distributed API），注意 org_vocab_start_index/padding 语义与 0.23 版本一致。
5. **checkpoint 加载**：新参数前缀 `attention_conv.*`、`mlp_conv.*`、`candidate_selector.*`；确认 z-lab/incoai 模型卡与 vllm 0.23 WeightsMapper 兼容（DFlash 基类已处理 midlayer 等，新增前缀需验证）。
6. **图模式**：conv/selector 纯 torch 算子 + torch.topk 可进 ACLGraph；walk 若为 torch 版可能无法整图捕获——初版 `--enforce-eager` 或 PIECEWISE，图模式后续验证。

## 5. 落地步骤（本地 vllm-ascend 树，分支 agent/dflash2-adaptation）

- [x] **S1 模型侧**：`vllm_ascend/models/qwen3_dflash2.py` + `_dflash2_math.py`
  - 纯 torch 移植 conv/selector/einsum；去掉 flashinfer 分支（Ascend 固定 torch.topk）
  - v0.23 差异处理：DecoderLayer 无 layer_idx；Model/LM 构造体按 dspark 惯例完整复制（基类硬编码类名）
  - 注册 `DFlash2DraftModel`（`vllm_ascend/models/__init__.py`）
- [x] **S2 worker 侧**：`vllm_ascend/spec_decode/dflash2_proposer.py`
  - `AscendDflash2Proposer(AscendDflashProposer)`，复用 `uses_markov_head` 分发钩子进 selector 尾部
  - **v1 范围**：确定性贪心 walk（=上游 T=0 语义，占收益大头）；T>0 的 Gumbel 路径 walk + 真实 q 分布缓存留 v2
  - walk 为 num_steps 次小 torch 算子（上游是单 Triton program）；triton-ascend 优化留 v2
  - v1 限制：不支持 lmhead TP（与 dspark 一致）
- [x] **S3 配置与分发**：`patch/platform/patch_speculators_dflash2.py`
  - v0.23 `update_dflash` 强制覆写架构为 DFlashDraftModel → patch 保留以 "DFlash" 开头的声明架构（对齐上游 #52816 规则）
  - `get_spec_decode_method` 增加 `DFlash2DraftModel` 分支
- [x] **S4 测试**：`tests/ut/models/test_qwen3_dflash2.py`
  - 上游两个数学单测移植 + 边界/确定性测试 + walk 顺序参考对拍 + tie-break；**9/9 CPU 全绿**
  - 关键修复记录：walk 的对齐索引（叉积索引 bug）已修
- [ ] **S5 上机 runbook**：模型 z-lab/Qwen3.8-27B-DFlash2 + 目标 Qwen3.8-27B；单算子 golden → 小模型 e2e → 8K 压测；A/B 对照（DFlash1 vs DFlash2 接受长度）
- [ ] **S6 交付**：exact-commit archive + manifest（沿用 stem handoff 工具）

## 6. 验收口径（对齐上游数据）

| 指标 | 上游参考值 | Ascend 验收 |
|---|---|---|
| Qwen3.8-27B 平均接受长度 | DFlash2 4.80 vs MTP 4.28 / DSpark 3.62 | 复现 DFlash1→DFlash2 提升 ≥ 15% 方向一致 |
| 输出无损 | rejection sampling 保证 | 与不开投机逐 token 对比 lossless |
| 时延开销 | conv+selector 共 +1.3% | profiling 确认占比，不阻塞收益 |
