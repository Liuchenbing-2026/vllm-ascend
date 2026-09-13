# MegaMoe BF16 接入交付报告

日期：2026-09-14  
代码分支：`codex/megamoe-bf16-local-partial-overlap-20260913`  
代码提交：`21c34b64 perf(megamoe): guard unfavorable bf16 ep shapes`

## 交付结论

本次交付的是 **BF16 MegaMoe 安全门控和可复现验证材料**。当前 Qwen3.6-35B-A3B 的 8 卡 TP8/EP8 配置不会再误触发已知不合适的 MegaMoe 路径，而是自动回退到 AllGather。

BF16 MegaMoe 在当前 EP8/E256 配置下没有取得可交付的 TTFT 收益，也没有通过无损精度验收，因此不能把该配置标记为“性能优化完成”。实际正收益工作点仍需在 EP16、每卡专家数不超过 24 的配置上重新验证。

## 已交付代码

修改文件：

- `vllm_ascend/ascend_forward_context.py`
- `tests/ut/ops/a2/test_megamoe.py`

新增统一 BF16 形状门控：

```text
BF16 MegaMoe 仅在 EP >= 16 且 experts_per_rank <= 24 时继续尝试；
否则 select_moe_comm_method() 返回 ALLGATHER。
```

这条门控覆盖 `mega_moe_local_partial=false` 的普通 EP 路径，也覆盖 local-partial 路径。量化 W8A8/W4A8 的原有选择逻辑保持不变。

## 验证结果

本地验证：

- `python -m py_compile vllm_ascend/ascend_forward_context.py tests/ut/ops/a2/test_megamoe.py`：通过。
- `git diff --check`：通过。
- 本地 pytest 未能启动，原因是当前 Windows 环境没有安装 `vllm`（`ModuleNotFoundError`），不是代码断言失败。

112.29.145.3 容器内烟测：

| 配置 | 选择结果 | 结论 |
|---|---|---|
| BF16、EP8、E256（每卡 32 专家） | `ALLGATHER` | 自动保护 |
| BF16、EP16、E256（每卡 16 专家） | `FUSED_MC2` | 允许进入候选路径 |

远端文件已按最小替换方式 staged 到：

```text
/data1/megamoe_gain_20260905/bf16_e2e_20260907/model_v31_runtime/vllm-ascend-bf16/vllm_ascend/ascend_forward_context.py
```

原文件备份为同目录下的 `ascend_forward_context.py.bak.21c34b64`。替换文件 SHA256 与本地版本一致，容器内 `py_compile` 通过。服务未启动，8001/8002/8100 端口保持空闲，4–7 卡未被本次验证占用。

## 性能和精度结论

此前严格 EP8 BF16 对照中，MegaMoe ON 在 2K 固定请求上出现：

- `text_equal=False`
- `logprobs_equal=False`
- token 首选一致率 31.25%
- logprob 平均绝对差 0.0533，最大差 0.4287

放宽逐字精度门槛后的参考压测仍然劣化：

| 输入 | AllGather OFF 平均 TTFT | MegaMoe ON 平均 TTFT | 变化 |
|---|---:|---:|---:|
| 4K | 1530.5 ms | 1629.4 ms | +6.46% |
| 6K | 1778.3 ms | 1822.3 ms | +2.47% |

对应吞吐下降约 3.74%（4K）和 1.41%（6K）。这些数据不构成精度或性能验收通过。

根因判断：EP8/E256 的每卡专家数为 32，超过当前 MegaMoe 的合适形状；同时 ON 路径存在 dispatch/combine 数值契约差异。前者已通过代码门控处理，后者仍需在满足形状的 EP16 工作点继续定位。两者都是基于已复现证据的结论，不能把它简化为单纯通信耗时问题。

## 复现方法

1. 使用分支 `codex/megamoe-bf16-local-partial-overlap-20260913`，确认提交为 `21c34b64`。
2. 在 A2 环境加载 vLLM-Ascend 和 CANN MegaMoe 依赖。
3. 对 Qwen3.6-35B-A3B 使用 BF16、TP8/EP8、E256，分别设置：
   - OFF：`enable_fused_mc2=0`
   - ON：`enable_fused_mc2=2`
4. 首先发送固定 2K 请求，检查文本、top-logprobs 和重复稳定性；精度门禁通过后再测 4K/6K、并发 32、160 请求的 TTFT 和吞吐。
5. 在 EP8 配置下，日志应出现：

```text
CANN MegaMoe BF16 shape guard: fallback to AllGather
```

6. 若要测试候选 MegaMoe 工作点，使用 EP16/E256（每卡 16 专家），保持模型、镜像、输入、采样参数和 CPU 绑核一致，并执行 OFF/ON 交叉重复测试。

已有详细实验和原始证据见：

- [MEGAMOE_GAIN_CONDITIONS.md](MEGAMOE_GAIN_CONDITIONS.md)
- [ROOT_CAUSE_LOCALIZATION.md](ROOT_CAUSE_LOCALIZATION.md)
- [REPRODUCE_SHAPE_GUARD_TTFT.md](REPRODUCE_SHAPE_GUARD_TTFT.md)

## 后续验收门槛

EP16 候选路径必须同时满足：

- 固定请求和重复请求精度无损；
- native MegaMoe、prepare、dispatch、combine、外层 TP reduction 的完整链路均有计时；
- TTFT 和 full-chain latency 相对 AllGather 稳定下降；
- 至少一轮正向和一轮反向顺序对照，结果差异超过噪声范围；
- 服务退出后端口、进程和 NPU 卡全部释放。

在这些条件完成前，本交付版本的定位是“安全接入和失败路径保护”，不是 BF16 MegaMoe 性能收益版本。