# DFlash 2 → vllm-ascend 上机验证 Runbook

> 基线：本地树 `agent/dflash2-adaptation` 分支（vLLM v0.23.0 + DSpark backport 之上）
> 上游：vllm#52816（未合并，head `19c9351904df4c63042671bc67a866ca48dc7d6f`）
> 目标产物：DFlash2 草稿模型 + selector 尾部的 Ascend 适配，v1 = 确定性贪心 walk（上游 T=0 语义）

## 0. 前置检查（每轮上机前）

```bash
ss -lntp | grep -E '8000|8001|8079'
ps -ef | grep -i vllmworker
npu-smi info
```

## 1. 部署与启动

### 1.1 安装

```bash
# 在验证容器内：卸载原 vllm-ascend 后安装本分支
pip uninstall -y vllm-ascend
pip install -e /path/to/vllm-ascend   # agent/dflash2-adaptation 分支
```

### 1.2 启动（首个 smoke）

```bash
vllm serve /data1/Qwen3.8-27B \
  --speculative-config '{
    "method": "dflash",
    "model": "z-lab/Qwen3.8-27B-DFlash2",
    "num_speculative_tokens": 7
  }' \
  --tensor-parallel-size 4 --port 8000 \
  --no-enable-prefix-caching
```

预期日志（无则排查分发）：

1. 模型注册命中：日志含 `DFlash2Qwen3ForCausalLM`（或加载日志显示 DFlash2 权重前缀
   `attention_conv` / `mlp_conv` / `candidate_selector`）；
2. 分发命中：启动时出现 `AscendDflash2Proposer`（可加临时日志确认）；
3. **未命中时最可能的静默失败**：架构被 v0.23 updater 覆写 → 检查
   `patch_speculators_dflash2.py` 是否被加载（`patch/platform/__init__.py` import 链）。

### 1.3 门禁一：架构未被降级

```python
from vllm.config import SpeculativeConfig
# 构造后检查 draft_model_config.hf_config.architectures == ["DFlash2DraftModel"]
```

## 2. 验证顺序（门禁递进，禁止跳级）

### G1 单算子 golden（CPU 侧已 9/9，NPU 侧复验）

- `grouped_conv` / `score_edges` / `selector_walk` 在 NPU tensor 上对拍 CPU 参考
  （把 `tests/ut/models/test_qwen3_dflash2.py` 的输入搬到 `torch.npu` 再对拍）；
- `compute_candidates` 在 TP4 下与逐 rank 手算 top-k 对齐（重点看
  `org_vocab_start_index` 偏移与 padding 置 -inf）。

### G2 小模型 e2e（Qwen3-0.6B/1.7B 级，或 z-lab 的 4B DFlash 系）

- 目标：走通 `AscendDflash2Proposer` 全链（context-KV 预计算 → conv 骨干 →
  compute_candidates → selector walk → 验证）；
- 接受长度 > 0 且输出与不开投机一致（T=0）。

### G3 真模型正确性（Qwen3.8-27B + z-lab/Qwen3.8-27B-DFlash2）

- T=0、固定 seed、`--no-enable-prefix-caching`，同一 prompt 两次输出 SHA256 一致；
- 输出与 Dense 逐 token 一致（投机解码 lossless 要求）。

### G4 A/B 接受长度（核心收益门禁）

| 配置 | 上游参考（Qwen3.8-27B） |
|---|---|
| MTP（原生） | 4.28 |
| DSpark（社区 drafter） | 3.62 |
| **DFlash2（本次适配）** | **4.80** |

- Ascend 验收：DFlash2 平均接受长度 **≥ DFlash1 同配置 +15%**，方向一致即可判通过；
- 指标：`benchmark_serving.py` 输出里的 spec 接受统计（或 logits debug 记录）。

### G5 稳定性/性能

- 8K 输入并发 4/8 压测：无 507015/MTE 越界、无重复输出漂移；
- profiling：conv+selector 尾部时延占比（上游 +1.3% 参考值）；
- 图模式：v1 先 `--enforce-eager` 验证；FULL 图模式（selector walk 为 torch 循环，
  预计只能 PIECEWISE/尾部 eager）作为 v2 目标，不做 v1 门禁。

## 3. 已知 v1 限制（上机时明确记录，不判失败）

1. **T>0 请求走确定性贪心 walk**（上游 T=0 语义）：收益仍真实（上游 T=0 增益
   +1.05 tokens），T>0 的 Gumbel 路径 + 真实 q 分布缓存留 v2；
2. **不支持 lmhead TP**（与 DSpark 相同限制）：报错即关 `lmhead_tp`；
3. **walk 为 torch 小算子循环**：长请求下 host 下发压力待 v2 triton-ascend 优化；
4. DFlash2 需要**未量化 lm_head**：Modelslim 量化目标模型不可用（上游同要求）。

## 4. 回退

任一门禁失败：保留日志（`vllm serve` 全量 + plog），回退分支到
`agent/glm52-dspark-investigation`，报告第一处分歧点（模型加载 / 分发 /
计算路径），禁止猜测性 hotfix。

## 5. 验证产物清单

- G1 对拍脚本与结果
- G2/G3 输出 SHA256 记录
- G4 A/B JSON（接受长度、吞吐、TTFT/TPOT）
- G5 profiling 报告 + 稳定性结论
- 与 DFLASH2_ADAPTATION_PLAN.md 的差距回填（如上游 #52816 合入后的 main2main 对齐动作）
