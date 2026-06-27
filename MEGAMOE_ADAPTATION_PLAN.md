# MegaMoe DispatchFFNCombine 适配方案

> **目标**：用 MegaMoe `DispatchFFNCombine` 融合算子（ops-transformer PR #6927）替换 vllm-ascend 现有 MoE 通路上的分散小算子，统一 A2 / A3 两系列的 prefill / decode 编排。
> **base**：`vllm-project/vllm-ascend:releases/v0.21.0rc` @ `fe821672`
> **工作分支**：`FutureSkyFly/vllm-ascend:megamoe`

---

## 一、现状（v0.21.0rc）

### 1.1 MoE 通信编排矩阵

|  | EP=1 | EP>1 + 小 tokens（≤ mc2_capacity）| EP>1 + 大 tokens |
|---|---|---|---|
| **A2**（910B） | ALLGATHER | `MC2`（专家≤24 + EP≥16）/ ALLGATHER | ALLGATHER ⚠️ **未接 FUSED_MC2** |
| **A3**（910C） | ALLGATHER | `FUSED_MC2`（开关满足）/ `MC2` | `FUSED_MC2`（仅开关==1）/ `ALLTOALL` |
| **_310P** | ALLGATHER | ALLGATHER | ALLGATHER |
| **A5** | ALLGATHER | `MC2` / ALLGATHER / ALLTOALL | ALLTOALL |

源：`vllm_ascend/ascend_forward_context.py:255-303`

### 1.2 FUSED_MC2 当前算子绑定

| `enable_fused_mc2` | 算子 | 量化 | 阶段 | 限制 |
|---:|---|---|---|---|
| 1 | `torch.ops._C_ascend.dispatch_ffn_combine` | W8A8 dynamic / BF16 | decode + prefill | **必须 NZ 格式**；BF16 时 scale 仍要塞空 tensor（PR 不支持 None） |
| 2 | `torch.ops._C_ascend.dispatch_gmm_combine_decode` | **仅 W8A8 dynamic** | **仅 decode** | EP≤32；非 draft model；speculative gate |

源：`vllm_ascend/ops/fused_moe/moe_comm_method.py:307-345`
源：`vllm_ascend/ops/fused_moe/fused_moe.py:116-128, 210-230`

### 1.3 FUSED_MC2 未启用时的「小算子链」

```
dispatch:  torch_npu.npu_moe_distribute_dispatch
GMM1:      torch_npu.npu_grouped_matmul / npu_grouped_matmul_swiglu_quant_weight_nz_tensor_list
swiglu:    隐式（融合在 GMM1 v3 / v_w8a8 op 里）
GMM2:      torch_npu.npu_grouped_matmul / npu_grouped_matmul_gmm2
combine:   torch_npu.npu_moe_distribute_combine
```

源：`vllm_ascend/ops/fused_moe/moe_mlp.py:149-390`、`vllm_ascend/ops/fused_moe/token_dispatcher.py`

### 1.4 现存 TODO（FusedMoE 代码里直接写明的）

| 位置 | TODO 内容 | 阻塞点 |
|---|---|---|
| `fused_moe.py:116-128` | dispatch_ffn_combine 仅支持 NZ → 期望 ND | 等下游 op 升级 |
| `fused_moe.py:212-220` | BF16 路径要传空 tensor 当 scale | C++ 签名不支持 Optional |
| `ascend_forward_context.py:287-288` | dispatch_ffn_combine EP-size guard 待解 | EP > 32 跑不了 |
| `ascend_forward_context.py:288` | dispatch_gmm_combine_decode 仅 w8a8_dynamic | 待支持 w16a16 |
| 整个 A2 路径 | 完全没 FUSED_MC2 接入 | 期待 MegaMoe 提供 A2 实现 |

---

## 二、MegaMoe 期望带来的能力（假设 ops-transformer PR #6927 落地）

> 因 gitcode 链接需登录，下列假设基于：现有 SVG 编排图命名 + 现存 TODO + 算子名 `DispatchFFNCombine`。**P0 阶段第一步要做的就是核实这些假设**。

| 能力 | 当前缺口 | MegaMoe 预期 |
|---|---|---|
| ND / NZ 双格式 | NZ-only | 同时支持 |
| Optional scale（BF16）| 强制传空 tensor | 接受 None |
| A2 平台覆盖 | 完全缺失 | A2 + A3 同接口 |
| Prefill 大 tokens 场景 | 走 ALLTOALL/ALLGATHER 小算子链 | 一体融合 |
| 大 EP（>32） | dispatch_ffn_combine 不支持 | 解锁 |
| 量化覆盖 | w8a8 dynamic 为主 | bf16/w8a8/w4a8 至少三档 |
| Draft model（MTP） | dispatch_gmm_combine_decode 屏蔽 | 解锁 |

---

## 三、适配方案（6 阶段）

### Phase 0 · 算子契约对齐（1-2 天）

**目标**：搞清楚 MegaMoe DispatchFFNCombine 的 C++ 签名、shape 约束、format 要求、量化方式枚举。

**动作**：
1. 拿到 ops-transformer PR #6927 的 `aclnn_dispatch_ffn_combine.h` 或新名（如 `aclnn_mega_moe_dispatch_ffn_combine.h`）
2. 对照现有 `dispatch_ffn_combine` 列出差异：参数名、Optional 支持、输出 layout、is_decode flag
3. 写一张差异表，归档到 `csrc/MEGAMOE_OP_DIFF.md`
4. 触发本地 op 验证脚本：`pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_dispatch_ffn_combine.py -v -k megamoe`

**交付**：差异表 + 单算子 UT 通过

---

### Phase 1 · torch binding（2-3 天）

**改动文件**：

| 文件 | 改动 |
|---|---|
| `csrc/dispatch_ffn_combine/` 新增 | 若 MegaMoe 是独立算子，加 `mega_moe_dispatch_ffn_combine_torch_adpt.h` + `op_host/` |
| `vllm_ascend/utils.py` | `CUSTOM_OPS` 列表加 `"mega_moe_dispatch_ffn_combine"` |
| `csrc/torch_binding.cpp` | 注册新 op + 旧 op 保留兼容 |
| `csrc/torch_binding_meta.cpp` | 加 meta 实现（fake 推断 shape）|

**关键决策**：**保留旧 op `dispatch_ffn_combine` 一个版本作为兼容**——通过 `VLLM_ASCEND_MEGAMOE_VARIANT` 切换，避免一次性切换炸雷。

---

### Phase 2 · A3 decode 路径接入（1 周）

**目标**：在 A3 decode 上把 MegaMoe 作为 `enable_fused_mc2==3` 选项加入（不动 1/2，保留回退）。

**改动文件**：

| 文件 | 改动 |
|---|---|
| `vllm_ascend/envs.py` | 扩 `VLLM_ASCEND_ENABLE_FUSED_MC2` 接受 `0/1/2/3`，新增 3 = MegaMoe |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py:259-348` | `FusedMC2CommImpl.fused_experts` 增 elif 分支：`enable_fused_mc2 == 3` → 调 MegaMoe |
| `vllm_ascend/ascend_forward_context.py:284-298` | A3 decode 分支加 `fused_mc2_enable == 3` 的条件路由 |
| `vllm_ascend/ops/fused_moe/fused_moe.py:122-128` | 若 MegaMoe 支持 ND，把 NZ format cast 改成条件分支 |
| `tests/ut/ops/fused_moe/test_moe_comm_method.py` | 新增 `test_fused_mc2_megamoe_path` 用例 |

**关键代码段（参考骨架）**：

```python
# moe_comm_method.py:FusedMC2CommImpl.fused_experts
elif get_ascend_config().enable_fused_mc2 == 3:
    out, expert_tokens = torch.ops._C_ascend.mega_moe_dispatch_ffn_combine(
        x=fused_experts_input.hidden_states,
        weight1=fused_experts_input.weights.w1,
        weight2=fused_experts_input.weights.w2,
        expert_idx=topk_ids,
        scale1=fused_experts_input.weights.w1_scale,     # 可为 None
        scale2=fused_experts_input.weights.w2_scale,     # 可为 None
        bias1=fused_experts_input.weights.w1_scale_bias,
        bias2=fused_experts_input.weights.w2_scale_bias,
        probs=fused_experts_input.topk_weights.to(torch.float32),
        group=self.token_dispatcher.moe_all_to_all_group_name,
        max_output_size=131072,
        swiglu_limit=fused_experts_input.swiglu_limit,
        x_active_mask=fused_experts_input.routing.mc2_mask,
        is_decode=True,
        is_a2=False,
    )
```

**验证**：DSv4 / GLM5 A3 decode 跑通 + acc 不掉点 + 单步 latency 持平或更优。

---

### Phase 3 · A2 路径接入（1.5 周）

**目标**：在 A2 上首次启用 FUSED_MC2 路径。

**改动文件**：

| 文件 | 改动 |
|---|---|
| `vllm_ascend/ascend_forward_context.py:271-279` | A2 分支整体改写：原 `MC2 / ALLGATHER` 二选一，扩成 `FUSED_MC2 / MC2 / ALLGATHER / ALLTOALL` |
| `vllm_ascend/utils.py:get_mc2_tokens_capacity` | A2 容量阈值校准（按 MegaMoe 实测 sweet spot 调）|
| `vllm_ascend/platform.py:716-720` | 解开 `enable_mc2_hierarchy_comm` 与 `enable_fused_mc2` 的互斥（如果 MegaMoe 支持） |
| `docs/source/user_guide/feature_guide/fused_mc2.md` 新增 | A2/A3 启用指南 |

**关键决策**：A2 ≠ A3，**通信组拓扑、SDMA 带宽、AIV/AIC 比例都不同**。需要按 A2 实测重订路由阈值，**不能直接复用 A3 的 num_experts_per_device ≤24 / EP≥16 阈值**。

**验证矩阵**：

| 模型 | TP | EP | DP | tokens/req | A2 期望 |
|---|---:|---:|---:|---:|---|
| DSv4 W8A8 | 1 | 32 | 32 | 128 (decode) | FUSED_MC2 |
| GLM5 W8A8 | 8 | 8 | 1 | 8192 (prefill) | FUSED_MC2 |
| Qwen3.5 MoE BF16 | 8 | 8 | 1 | 1024 (prefill) | FUSED_MC2（如 MegaMoe 支持 BF16）|

---

### Phase 4 · Prefill 路径接入（1 周）

**目标**：A2/A3 的 prefill 也接 MegaMoe（解开 `num_tokens > mc2_tokens_capacity` 走 ALLTOALL/ALLGATHER 的限制）。

**改动文件**：

| 文件 | 改动 |
|---|---|
| `vllm_ascend/ascend_forward_context.py:300-305` | Prefill 分支：当 MegaMoe 支持 large-token 时，去掉 `num_tokens > mc2_tokens_capacity` 走 ALLTOALL 的回退 |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | `FusedMC2CommImpl` 加 `is_decode` flag 透传给 op |

**风险**：prefill 走 MegaMoe 的 GEMM 计算量和 decode 差一个数量级；如果算子对大 M（token 数）维度不友好，可能反而比 ALLTOALL+GMM 慢。**Phase 4 上线前必须有 prefill 性能基准。**

---

### Phase 5 · 解锁 NZ-only / scale-required 限制（3-5 天）

**目标**：清理 v0.21.0rc 现存 TODO。

**改动文件**：

| 文件 | 行号 | 改动 |
|---|---:|---|
| `vllm_ascend/ops/fused_moe/fused_moe.py` | 116-128 | NZ-only 改条件：MegaMoe → ND；旧 op → NZ |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | 210-230 | BF16 path 直接传 `None` 给 scale（依赖 MegaMoe 接受 Optional）|
| `vllm_ascend/ascend_forward_context.py` | 287-288 | 删 `ep_world_size <= 32` 限制（依赖 MegaMoe 支持大 EP）|
| `vllm_ascend/ascend_forward_context.py` | 288 | 删 `not is_draft_model` 限制（MTP 也可走）|

---

### Phase 6 · 测试矩阵 + 文档（持续）

**E2E 测试用例**：

| 测试 | 平台 | 模型 | 配置 | 预期 |
|---|---|---|---|---|
| `tests/e2e/.../test_megamoe_dsv4_a3.py` | A3 | DSv4 W8A8 | TP8 DP1 EP8 | decode FUSED_MC2 |
| `tests/e2e/.../test_megamoe_glm5_a2.py` | A2 | GLM5 W8A8 | TP8 DP1 EP8 | prefill FUSED_MC2 |
| `tests/e2e/.../test_megamoe_qwen3_moe_bf16.py` | A3 | Qwen3 MoE BF16 | TP4 DP2 EP8 | BF16 None scale |
| `tests/e2e/.../test_megamoe_mtp.py` | A3 | DSv4 + MTP | TP8 EP8 | draft model 也走 FUSED |
| `tests/e2e/.../test_megamoe_large_ep.py` | A3 | DSv4 W8A8 | EP=64 | 解锁 EP>32 |

**性能基准**：每个 Phase 末跑 `vllm bench serve` ×3，对比开关 0 / 1 / 2 / 3（旧 vs MegaMoe），输出 TTFT/TPOT/throughput 对比。

**文档**：

| 文档 | 内容 |
|---|---|
| `docs/source/user_guide/feature_guide/fused_mc2.md` 新增 | MegaMoe / FUSED_MC2 完整启用指南 + 4 档 enable_fused_mc2 含义 |
| `docs/source/user_guide/release_notes.md` | 记录 Phase 2-5 落地的 PR 号 |
| 此文档 `MEGAMOE_ADAPTATION_PLAN.md` | 设计依据保留 |

---

## 四、PR 拆分建议

| PR | 范围 | 行数估计 | 依赖 |
|---|---|---:|---|
| PR-A | csrc/ 加 MegaMoe op binding | +300/-0 | ops-transformer PR #6927 合入 |
| PR-B | `enable_fused_mc2==3` A3 decode 接入 | +120/-20 | PR-A |
| PR-C | A2 路径接入 | +200/-50 | PR-B |
| PR-D | Prefill 大 tokens 路径 | +80/-40 | PR-C |
| PR-E | 清 NZ-only / scale-required TODO | +60/-80 | PR-D |
| PR-F | 文档 + e2e 测试 | +400/-0 | PR-E |

**总改动量预估**：+1160 / -190，6-8 周完成。

---

## 五、风险与回滚

| 风险 | 缓解 |
|---|---|
| MegaMoe op 签名与现 PR 假设不一致 | Phase 0 强制对齐，差异表先行 |
| A2 性能不如小算子链 | A2 接入按 `enable_fused_mc2==3` 显式开关，默认 0 不影响存量用户 |
| Prefill 大 tokens 反而变慢 | Phase 4 上线前必有性能基准，不达标不合 |
| EP > 32 出现 hccl 死锁 | 保留旧 dispatch_ffn_combine 兼容路径，env 切换 |
| 量化 path（W8A8/W4A8/BF16）行为不一致 | Phase 6 测试矩阵覆盖所有量化档 |

---

## 六、依赖与 owner

| 依赖项 | 状态 | owner |
|---|---|---|
| ops-transformer PR #6927 | 待合入 | CANN 团队 |
| MegaMoe DispatchFFNCombine C++ 接口 | 待对齐（Phase 0）| 本人 + CANN 联调 |
| 测试卡：A2 / A3 各 1 台 8 卡 | 待申请 | — |

---

## 七、决策点（先回答这 3 个问题）

1. **是新算子（独立 op）还是 in-place 升级 `dispatch_ffn_combine`？** 决定 PR-A 是「加新 op」还是「改老 op」
2. **A2 上 MegaMoe 默认开 or 默认关？** 决定 `enable_fused_mc2` 默认值和 forward_context 路由
3. **`dispatch_gmm_combine_decode`（开关==2）是否在 MegaMoe 接入后废弃？** 决定 enable_fused_mc2 是否 4 档共存或合并为 2 档

回答完这 3 个再开 Phase 1。

---

*文档版本*：v1 / 2026-06-27
*作者*：FutureSkyFly
*分支*：`megamoe`（自 `releases/v0.21.0rc` @ fe821672 派生）
