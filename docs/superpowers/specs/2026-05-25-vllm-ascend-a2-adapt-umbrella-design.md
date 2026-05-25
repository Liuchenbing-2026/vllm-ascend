# vllm-ascend A2 适配 — 总设计 (umbrella)

- 日期：2026-05-25
- 工作分支：`a2-adapt`（从 `1383e98` 切出）
- 涵盖 feature：5 个独立子项目
- 输出形式：本地 clone + 7 commit（1 docs + 1 infra + 5 feature），不主动 push
- 验收阶段：spec 阶段 = build/lint/AST 通过；NPU runtime 测试由 user 后续在 192.168.99.41 deepseek-v4-t1 容器内跑

## 1. 背景

- 5 个 EP / Attention 通信优化特性当前在 A3（Ascend910C / NpuArch `ASCEND910_93`）上能跑、在 A2（Ascend910B 系列 / NpuArch 220）上跑不起来。
- 本套设计为 A2 加 device-guard + A2 专用通信实现，A3 / A5 / 310P 行为零变化。
- 5 feature：
  - **F01** `dsa_cp 关 all2all`：DeepSeek Sparse Attention + Context Parallel 在 A2 上禁用 all2all 通信
  - **F2.1** `用 all2all`：A2 EP 支持 ALLTOALL comm 路径（上游 `vllm-ascend` 当前 A2 elif 只产 MC2/ALLGATHER）
  - **F2.2** `dispatch & combine`：A2 启用 `dispatch_ffn_combine` HCCL primitive 单独路径（bs 有约束）
  - **F2.3** `PP + ep_gather`：PP > 1 时 A2 用 vllm 主仓 `deep_gemm_utils.ep_gather`
  - **F2.4** `PP + MoE 大融合算子`：A2 上 PP + dispatch_combine + ep_gather 完整融合

## 2. Scope 决定

- 一份 umbrella spec（本文件）+ 5 份 sub-spec：
  - `2026-05-25-a2-adapt-f01-dsa-cp-disable-all2all.md`
  - `2026-05-25-a2-adapt-f21-moe-alltoall.md`
  - `2026-05-25-a2-adapt-f22-moe-dispatch-combine.md`
  - `2026-05-25-a2-adapt-f23-moe-pp-ep-gather.md`
  - `2026-05-25-a2-adapt-f24-moe-pp-fused-mc2.md`
- 实施顺序固定：infra → F01 → F2.1 → F2.2 → F2.3 → F2.4（F2.4 依赖 F2.2 + F2.3 的模块）

## 3. 代码组织（候选 B）

- **原地 device-guard**：现有文件加 `if get_ascend_device_type() == AscendDeviceType.A2:` 分支；新增 impl 用同目录 `*_a2.py` helper 控制行宽。
- 不采用候选 A（镜像 `_310p/` 新建 `_a2/` 子目录）：代码重复 + drift 风险。
- 不采用候选 C（patch-only）：A2 是默认硬件，不应走 patch 通道。
- 关键入口已经是 A2/A3/_310P/A5 显式 elif：`vllm_ascend/ascend_forward_context.py:233 select_moe_comm_method`、`vllm_ascend/ops/fused_moe/moe_comm_method.py:55 setup_moe_comm_method`。

## 4. 共享 idiom

### 4.1 Device guard

```python
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
if get_ascend_device_type() == AscendDeviceType.A2:
    # A2 path
```

不增 `is_a2()` helper；`vllm_ascend/utils.py` 只暴露 `is_310p`，A2 用枚举判断保持风格一致。

### 4.2 Env vars

放 `vllm_ascend/envs.py` 的 `env_variables` dict，遵循 AGENTS.md `VLLM_ASCEND_*` 命名：

- `VLLM_ASCEND_A2_DSA_CP_DISABLE_ALL2ALL`（F01，int 0/1，默认 1）
- `VLLM_ASCEND_A2_MOE_COMM`（F2.1-F2.4，取值 `auto|alltoall|dispatch_combine|pp_ep_gather|pp_fused|none`，默认 `auto`）
- `VLLM_ASCEND_A2_DISPATCH_COMBINE_BS_MIN`（F2.2，int，默认 1）
- `VLLM_ASCEND_A2_DISPATCH_COMBINE_BS_MAX`（F2.2，int，默认 = `mc2_tokens_capacity` 运行期解析）

A2 之外的 device 完全忽略这 4 个 env。

### 4.3 AscendConfig user-facing

在 `vllm_ascend/ascend_config.py:AscendConfig.__init__` 加：

```python
a2_adapt_config = additional_config.get("a2_adapt_config", {})
self.a2_adapt_config = A2AdaptConfig(**a2_adapt_config)
```

`A2AdaptConfig` 字段镜像 env，`additional_config` 优先级 > env > 默认。同 dataclass 风格如 `AscendFusionConfig` / `EplbConfig`。

### 4.4 MoECommType 扩 enum

`vllm_ascend/ascend_forward_context.py:26` 在 4 个现有 enum 后追加：

```python
class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3
    DISPATCH_COMBINE = 4   # F2.2 新增
    PP_EP_GATHER = 5       # F2.3 新增
    PP_FUSED_MC2 = 6       # F2.4 新增
```

A3/A5/_310P 的 `select_moe_comm_method` 分支不会返回新 enum；A2 elif 在 `a2_moe` 取相应字符串或 `auto` 命中条件时返回。

### 4.5 命名与目录

| 模块 | 文件 |
|---|---|
| A2 MoE comm impl（F2.1-F2.4 全部新 impl） | `vllm_ascend/ops/fused_moe/comm_a2.py` |
| A2 token dispatcher（如需 fork） | `vllm_ascend/ops/fused_moe/token_dispatcher_a2.py` |
| A2 DSA-CP alt 路径（F01） | `vllm_ascend/attention/context_parallel/attention_cp_a2.py`（实际 F01 候选 A 不创建文件，预留位置） |
| 选择层 | 改原 `ascend_forward_context.py`、`moe_comm_method.py`、`ascend_config.py`、`envs.py` |

### 4.6 默认 OFF 原则

- F01 默认 ON（A2 上 dsa_cp 默认不走 all2all）—— 因为 user 反馈 A2 上当前不可用，默认就该关。
- F2.1-F2.4 默认 `auto`：A2 上的 EP 路径在 `auto` 下保持现有 MC2/ALLGATHER 行为；user 主动设 `additional_config.a2_adapt_config.moe_comm = "alltoall" / "dispatch_combine" / "pp_ep_gather" / "pp_fused"` 才切到新 impl。
- 例外：F2.4 `auto` + PP > 1 + 满足条件 → 默认返回 `PP_FUSED_MC2`（A2 PP 模式下的最优路径），其他情况兜底 MC2/ALLGATHER。

### 4.7 测试占位

每 feature 在 `tests/ut/<area>/test_*_a2.py` 加 dummy UT（仅 select 函数 / config 解析的单测，不上 NPU）：

- monkey-patch `get_ascend_device_type` 返回 A2
- 给定 env / additional_config 输入，断言 `select_moe_comm_method` 返回预期 enum
- 不引入 torch_npu 真实算子调用

NPU runtime UT / e2e 由 user 后续在 192.168.99.41 上跑。

## 5. 交付与验收

### 5.1 Commit 布局（共 7 commit = 1 docs + 1 infra + 5 feature）

| # | Commit message | 内容 |
|---|---|---|
| 0 | `docs(a2): umbrella + 5 sub-specs for A2 adaptation` | 本 spec 目录 6 文件 |
| 1 | `feat(a2): add A2AdaptConfig and env vars infrastructure` | envs.py 4 新 env、ascend_config.py 新 dataclass、ascend_forward_context.py MoECommType 扩 3 enum |
| 2 | `feat(a2): disable all2all on dsa_cp for A2 devices` | F01 |
| 3 | `feat(a2/moe): add A2 all2all comm method` | F2.1 |
| 4 | `feat(a2/moe): add A2 dispatch_combine comm method` | F2.2 |
| 5 | `feat(a2/moe): add A2 PP+ep_gather comm method` | F2.3 |
| 6 | `feat(a2/moe): add A2 PP+fused-MC2 comm method` | F2.4 |

每 commit 自带 sign-off `Signed-off-by: <user 邮箱待 user 改>`（spec 阶段占位本机邮箱）。

### 5.2 验收口径

1. `python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('vllm_ascend/**/*.py', recursive=True)]"` AST parse 全过。
2. `ruff check vllm_ascend/ops/fused_moe/ vllm_ascend/attention/ vllm_ascend/ascend_forward_context.py vllm_ascend/envs.py vllm_ascend/ascend_config.py` 通过。
3. `git log --oneline a2-adapt ^main` 显示 7 commit（1 docs + 1 infra + 5 feature），全部 sign-off。
4. `grep -RInE "TODO|FIXME|XXX" vllm_ascend/ops/fused_moe/comm_a2.py` 等新文件无占位关键字。
5. 每 feature commit 在 message body 附 `Test: python -c "from vllm_ascend.<module> import *"` 的本机 dry-run 结果（无 torch_npu 时跳过 import，不算失败）。

### 5.3 Push 流程

- 完成 7 commit 后不主动 push。
- spec 末尾留命令：`git remote add <name> <url>; git push -u <name> a2-adapt`，等 user 给 remote 后再代跑（需 user 二次确认）。

## 6. 与 AGENTS.md 的偏离

| AGENTS.md 要求 | 本设计处理 |
|---|---|
| `git commit -s` sign-off | 用本地占位邮箱 sign-off，user 改 commit 替换 |
| 新 env 必须有 review | 4 个新 env 在 spec 4.2 集中列；reviewer = user |
| 新功能必须配 UT | 只上 dummy select UT；NPU UT 待 user 在 192 上补 |
| PR 从 fork 推 | 本 spec 阶段不开 PR；user 后续手动操作 |

## 7. 关键 unknown / 风险

| 项 | 影响 feature | 解决路径 |
|---|---|---|
| `torch_npu.npu_dispatch_ffn_combine` 在 A2 上的实际 API 名 | F2.2 / F2.4 | user 在 191 实测时按 log 修正 |
| vllm 主仓 `deep_gemm_utils.ep_gather` 在 A2 上是否原生支持 | F2.3 / F2.4 | F2.3 impl 内 try/except → RuntimeError 让 user log 反馈 |
| `AlltoAllCommImpl` / `TokenDispatcherWithAll2AllV` / `PrepareAndFinalizeWithAll2All` 是否 device-agnostic | F2.1 | grep 时确认无 `assert A3`；运行期出问题再 fork 到 `comm_a2.py` |
| MoE fused kernel 在 A2 上的 bs 上界精确值 | F2.2 / F2.4 | env `VLLM_ASCEND_A2_DISPATCH_COMBINE_BS_MAX` 默认 `mc2_tokens_capacity`，user 实测调 |

## 8. 后续 step

- spec 自审完成后，user 复核 7 份 spec 文件。
- 批准后调 `superpowers:writing-plans` 出 6 个 feature commit 的实施计划（按 5.1 的顺序）。
- writing-plans 完成后切到 implementation phase，按 plan 一 commit 一 commit 落地。
