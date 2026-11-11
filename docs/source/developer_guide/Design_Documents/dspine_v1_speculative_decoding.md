# DSpine 草稿模型接入（v1 runner）

DSpine（*Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for
Speculative Decoding*，arXiv 2609.36173）是在 DFlash 并行草稿主干上加了逐层 code bridge 的
投机解码草稿。本分支把 DSpine 接到 Ascend 的 **v1 (MRV1) 投机解码通路**上；MRV2 的
DSpine 集成见 `adapt/dspine-v033-main-20261111`。

## 检查点特征

DSpine checkpoint 的 `config.json` 声明 `architectures: ["DFlashDraftModel"]`，并在
`dflash_config` 里带 `selected_prefix_code_bridge`（`code_size`/`rank`/`eps`）。权重里没有
`embed_tokens` / `lm_head`（与目标共享），块大小 `block_size=16`，`head_topk=16`，
`draft_block_causal=true`。

## 接入方式

复用 `dflash` 方法，不新增 speculative method：

- `patch/platform/patch_speculative_config.py`：检测到 code bridge 后把架构改写成
  `DFlashDSpineDraftModel`，并置 `is_causal=true`（vLLM 的 DFlash 模型据此给每个 draft
  layer 打开块内 causal attention）。
- `models/qwen3_dspine.py`：`DSpineQwen3ForCausalLM`（`model_cls = DSpineQwen3Model`）
  与 `DSpineCodeBridge`。backbone 直接继承 vLLM 的 `DFlashQwen3Model` / 解码层，只在
  `forward` 里插入逐层写回，并记录最后一层的 `raw/rms/packed` 供修正尾部使用。权重严格
  校验（漏张量/多张量/形状不符直接报错），`out.weight` 必须为零。
- `models/dspine_math.py`：纯 torch 数学（无 vLLM 依赖）。逐层写回、词表级 `TG/TSV`
  表、K-lane 串行末写修正。
- `worker/v1/spec_decode/dspine_proposer.py`：`AscendDspineProposer` 继承
  `AscendDflashProposer`。复用 DFlash 的输入构造、context-KV 预写和 graph 机制；在
  `load_model` 里建表（一次 ~0.1s，常驻约 466 MB bf16），并复用基类给 DFlash 系草稿
  预留的自定义尾部钩子（`use_dflash2_selector` + `compute_draft_token_ids`）跑串行修正。
  `set_inputs_first_pass` 把 draft attention 置为 causal。

## 使用

```bash
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method":"dflash","num_speculative_tokens":15,"model":"tencent/Qwen3-8B-DSpine-b16"}'
```

`num_speculative_tokens` 必须等于 `block_size - 1`（b16 为 15）。参考启动脚本：
`tools/dspine/serve_dspine.sh`。

## 约束

- 仅支持 **MRV1**（`VLLM_USE_V2_MODEL_RUNNER=0`）。当前 main 已移除 `vllm_ascend/worker/v1`
  与对应的 `vllm/v1/spec_decode` 提议器接口，因此本分支基于仍带 MRV1 的
  `9ea9aa2`（vllm main `24420b7a`，`VLLM_VERSION=0.33.1.dev1`）。
- 只支持 dense Qwen3 系、bf16、`selected_prefix_terminal=false`、serial 解码、
  无 grouped conv 的 DSpine checkpoint；其它形态在 config 解析阶段报错。
- 需要 TP=1、PP=1（修正尾部读完整 head；draft 与目标共享 embedding/head）。
- 采样请求按 greedy 草稿处理（与现有 DFlash v1 一致），参考实现的 sampled chain 未移植。

## 实测（q38，单卡 910B4-1，Qwen3-8B，16 条真实技术提示 × 128 tokens，greedy，batch=1）

| 方案 | 吞吐 | 接受长度 AL |
|---|---:|---:|
| 无投机 | 15.4 tok/s | — |
| DFlash b16 | 21.0 tok/s | 2.24 |
| DSpine b16 | 22.6 tok/s | 3.05 |

DSpine 相对无投机 1.47×，相对 DFlash：AL +36%、吞吐 +7.6%（每步修正开销约 +8%）。
开放域短提示的绝对 AL 低于论文 benchmark 口径，方向和幅度一致。