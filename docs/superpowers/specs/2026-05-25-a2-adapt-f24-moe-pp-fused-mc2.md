# F2.4：A2 上 PP + MoE 大融合算子

- 父 spec：`2026-05-25-vllm-ascend-a2-adapt-umbrella-design.md`
- 候选选定：A（新 enum + 新 impl，组合 F2.2 / F2.3 模块）
- Commit：`feat(a2/moe): add A2 PP+fused-MC2 comm method`
- User 估收益：+40%
- 备注：最终路径，依赖 F2.2 的 `DispatchCombineA2CommImpl` + F2.3 的 `PpEpGatherA2CommImpl`

## 1. 目标

A2 上 PP > 1 时把 F2.2 的 `dispatch_combine` + F2.3 的 `ep_gather` 进一步融合成单一通信 + GMM kernel；对标 A3 上的 `FUSED_MC2 mode 1` + PP 包装。

## 2. 当前 main 的相关入口

| 文件:行 | 内容 |
|---|---|
| `vllm_ascend/ascend_forward_context.py:283-305` | A3 elif 的 FUSED_MC2 路径 —— mirror 参考 |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py:PrepareAndFinalizeWithMC2` | A3 上 FUSED_MC2 用的 prepare_finalize |
| F2.2 / F2.3 sub-spec 输出 | `comm_a2.py` 内 `DispatchCombineA2CommImpl` / `PpEpGatherA2CommImpl` |

## 3. 候选方案

| 候选 | 实施 | 取舍 |
|---|---|---|
| A. **新 enum + 组合 F2.2 / F2.3**（选定） | `MoECommType.PP_FUSED_MC2 = 6`；`PpFusedMC2A2CommImpl` 内部组合 ep_gather + dispatch_combine 流水 | 语义清晰；和 F2.2 / F2.3 模块化拼装；A3 跨超验证时 mirror 容易 |
| B. 复用 A3 FUSED_MC2 enum | A2 elif 加 FUSED_MC2 决策；setup 给 A2 注册 FusedMC2CommImpl | 代码少；但 F2.2 已用独立 enum，B 会破坏 ablation 路线 |

候选 A 选定。

## 4. 实施细节

### 4.1 MoECommType.PP_FUSED_MC2 已在 infra commit 加

参考 umbrella 4.4：`MoECommType.PP_FUSED_MC2 = 6`。

### 4.2 setup_moe_comm_method 增 A2 + PP 注册

`vllm_ascend/ops/fused_moe/moe_comm_method.py`：

```python
if get_ascend_device_type() == AscendDeviceType.A2:
    pp_size = get_current_vllm_config().parallel_config.pipeline_parallel_size
    if pp_size > 1:
        _MoECommMethods[MoECommType.PP_FUSED_MC2] = PpFusedMC2A2CommImpl(moe_config)
```

（接在 F2.3 同 if 块内）

### 4.3 新写 `comm_a2.py:PpFusedMC2A2CommImpl`

```python
class PpFusedMC2A2CommImpl(MoECommMethod):
    """A2-only: PP-aware MoE comm fusing ep_gather and dispatch_combine.

    Mirrors A3 FUSED_MC2 mode 1 with PP packaging. Internally composes the
    prepare_finalize / token_dispatcher from F2.3 (ep_gather) and F2.2
    (dispatch_combine).
    """

    def __init__(self, moe_config):
        super().__init__(moe_config)
        self._ep_gather_impl = PpEpGatherA2CommImpl(moe_config)
        self._dispatch_combine_impl = DispatchCombineA2CommImpl(moe_config)

    def _get_token_dispatcher(self):
        # 复用 dispatch_combine 的 MC2 dispatcher，处理 token 路由
        return self._dispatch_combine_impl.token_dispatcher

    def _get_prepare_finalize(self):
        return PrepareAndFinalizePpFusedMC2A2(
            self.moe_config,
            ep_gather_pf=self._ep_gather_impl.prepare_finalize,
            dispatch_combine_pf=self._dispatch_combine_impl.prepare_finalize,
        )
```

新写 `PrepareAndFinalizePpFusedMC2A2(PrepareAndFinalize)`：

- `prepare`: 调 ep_gather（F2.3） → 取得 expert-major tensor → 喂给 dispatch_combine 的 prepare（F2.2）的内部 dispatch 部分。融合点：把 ep_gather 输出直接 stream 到 dispatch 算子，不落 HBM；实现上等 user 在 191 上跑 profiling 后再决定 fusion 程度。
- `finalize`: dispatch_combine 的 finalize → 反向 ep_gather（实际是 reduce_scatter 类似）→ 输出到 PP next stage。

实施时复用 `_ep_gather_impl` / `_dispatch_combine_impl` 的现成 API，避免重复底层算子调用代码。

### 4.4 select_moe_comm_method A2 elif 扩

在 F2.3 之上 prepend（PP_FUSED 优先 PP_EP_GATHER）：

```python
elif soc_version in {AscendDeviceType.A2}:
    a2_moe = get_ascend_config().a2_adapt_config.moe_comm
    ...
    pp_size = vllm_config.parallel_config.pipeline_parallel_size

    if a2_moe == "alltoall":
        moe_comm_type = MoECommType.ALLTOALL
    elif a2_moe == "dispatch_combine":
        moe_comm_type = MoECommType.DISPATCH_COMBINE
    elif a2_moe == "pp_ep_gather":
        ...  # F2.3 已定
    elif a2_moe == "pp_fused":
        if pp_size > 1 and num_tokens <= mc2_tokens_capacity:
            moe_comm_type = MoECommType.PP_FUSED_MC2
        elif pp_size > 1:
            moe_comm_type = MoECommType.PP_EP_GATHER   # fall back F2.3
        else:
            logger.warning("a2_moe='pp_fused' requires PP>1; falling back to ALLGATHER")
            moe_comm_type = MoECommType.ALLGATHER

    elif a2_moe == "auto" and pp_size > 1 and ep_world_size <= 32 and num_tokens <= mc2_tokens_capacity:
        # A2 PP 模式下默认最优路径 = PP_FUSED_MC2
        moe_comm_type = MoECommType.PP_FUSED_MC2
    elif a2_moe == "auto" and pp_size > 1 and num_tokens <= mc2_tokens_capacity and ep_world_size > 1:
        # F2.3 兜底（ep_size > 32 时退化）
        moe_comm_type = MoECommType.PP_EP_GATHER
    elif a2_moe == "auto" and ep_world_size <= 32 and bs_min < num_tokens <= bs_max and not is_draft_model:
        moe_comm_type = MoECommType.DISPATCH_COMBINE
    elif a2_moe == "auto" and num_tokens > mc2_tokens_capacity and ep_world_size >= 8:
        moe_comm_type = MoECommType.ALLTOALL
    elif num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity:
        moe_comm_type = MoECommType.MC2
    else:
        moe_comm_type = MoECommType.ALLGATHER
```

注 auto 优先级（PP > 1 时）：`PP_FUSED_MC2 > PP_EP_GATHER`；非 PP 时 `DISPATCH_COMBINE > ALLTOALL > MC2 > ALLGATHER`。User 用 `a2_moe` 字符串强制覆盖。

### 4.5 UT 占位

`tests/ut/ops/fused_moe/test_a2_pp_fused_mc2_select.py`：

- device=A2 + `a2_moe="auto"` + `pp_size=2` + `ep_world_size=8` + `num_tokens=256` → 断言 `PP_FUSED_MC2`
- device=A2 + `a2_moe="pp_fused"` + `pp_size=2` + `num_tokens > mc2_capacity` → 断言 `PP_EP_GATHER`（fall back）
- device=A2 + `a2_moe="pp_fused"` + `pp_size=1` → 断言 `ALLGATHER` + warn

## 5. Error handling

- HCCL fused kernel 不可用：`PrepareAndFinalizePpFusedMC2A2` 内部组合时由 F2.2 / F2.3 各自 RuntimeError 暴露；user 切换到 dispatch_combine / pp_ep_gather 兜底。
- bs 超约束：select 函数 fall back PP_EP_GATHER。
- pp_size = 1：fall back ALLGATHER + warn。

## 6. 改动文件清单

- `vllm_ascend/ascend_forward_context.py`（A2 elif 加 pp_fused 决策 + auto 优先级调整）
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`（setup_moe_comm_method 加 A2 + PP 的 PP_FUSED_MC2 注册）
- `vllm_ascend/ops/fused_moe/comm_a2.py`（追加 `PpFusedMC2A2CommImpl` + `PrepareAndFinalizePpFusedMC2A2`）
- `tests/ut/ops/fused_moe/test_a2_pp_fused_mc2_select.py`（新）

## 7. 验收

- AST + ruff 通过
- UT 通过
- `python -c "from vllm_ascend.ops.fused_moe.comm_a2 import PpFusedMC2A2CommImpl"` 不报错（不实例化）
- commit message 附 dry-run 结果

## 8. Known unknown

- A2 上 fused HCCL kernel（ep_gather + dispatch_combine 融合）的真实性能：user 估 +40%，实际 user 在 191 实测；spec 不预承诺收益。
- 若 A2 上无原生融合 kernel 而是两个 primitive 串调：性能可能接近 F2.3 + F2.2 简单串联。User 后续 profiling 决定是否要在 A2 上写自定义 fused 算子（不在本 spec 范围）。
