# V31 / V30 四卡 BF16 单算子复测（2026-09-13）

本次使用整网验收时相同的 V31 接入源码和 V30 CANN 内核。OFF 指 CANN 分解算子路径，不是官方 CANN MegaMoe；ON 指当前 local-partial MegaMoe。

采用合成均匀随机 TopK 路由与固定种子的 BF16 权重，双方内容一致。它对齐运行时、版本、形状和数据类型，没有重放整网真实权重或路由分布。未包含 router TopK、共享专家 MLP、共享专家多流重叠、attention 或调度，因此不能据此直接解释或否定整网的吞吐/TTFT 结果。

## 1. 测量边界

- routed_core：OFF 从 routing 开始到 unpermute 结束，含必要中间处理；ON 为预先准备输入后的 MegaMoe 调用。输入准备在计时外，每次重建可能修改的输入并同步。
- routed_with_prepare：在上项基础上纳入 OFF 路由概率掩码和 ON 当前生产 Triton 准备函数；ON 使用 preapply_active_mask=True。
- full_chain：再加固定的共享专家部分结果及最终 BF16 AllReduce；固定结果相加不是共享专家计算。
- 普通计时不启用 profiler；每轮同步后取 host wallclock，每个 iteration 取四 rank 最慢值，再取每个 block 的中位数，最后取两个 OFF / ON block 的均值。顺序 OFF A1 → ON B1 → ON B2 → OFF A2，各 block 预热 10、采样 30。
- 设备时间使用另一次 msprof 运行，只运行 routed_with_prepare。每个 block 预热 10、采样 10；剔除正确性、连续调用及预热记录。

## 2. 不开 profiler 的耗时

正百分比表示 MegaMoe 更慢。各层独立测量，不能用两层之差当作精确组件耗时。

| 范围 | tokens | OFF ms | ON ms | 延迟变化 | 正序 / 反序变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| routed_core | 512 | 1.2280 | 1.6925 | +37.82% | +38.95% / +36.69% |
| routed_core | 4096 | 2.0947 | 2.4754 | +18.18% | +18.71% / +17.65% |
| routed_core | 6144 | 2.7337 | 2.8789 | +5.31% | +5.43% / +5.19% |
| routed_core | 8192 | 3.2150 | 3.3095 | +2.94% | +2.03% / +3.85% |
| routed_with_prepare | 512 | 1.3197 | 1.9211 | +45.57% | +46.03% / +45.10% |
| routed_with_prepare | 4096 | 2.2121 | 2.6914 | +21.67% | +21.25% / +22.08% |
| routed_with_prepare | 6144 | 2.8455 | 3.1542 | +10.85% | +10.40% / +11.30% |
| routed_with_prepare | 8192 | 3.3209 | 3.5534 | +7.00% | +7.64% / +6.36% |
| full_chain | 512 | 1.4495 | 2.0535 | +41.67% | +41.65% / +41.69% |
| full_chain | 4096 | 2.8058 | 3.2754 | +16.73% | +16.91% / +16.56% |
| full_chain | 6144 | 3.6310 | 4.0283 | +10.94% | +11.90% / +9.99% |
| full_chain | 8192 | 4.4442 | 4.7683 | +7.29% | +7.04% / +7.55% |

## 3. 设备侧时间

OFF 为同一 stream 上 routing 开始到 unpermute 结束的区间，包含区间内必要任务及下发空隙；ON 为 MegaMoe task duration。采用四 rank 每次调用最大值及两个 block 中位数的均值。两边都不含输入准备和最终 AllReduce；这是 profiling 诊断，不与普通计时混算。

| tokens | OFF 区间 μs | MegaMoe μs | 延迟变化 |
| --- | ---: | ---: | ---: |
| 512 | 1099.188 | 1136.588 | +3.40% |
| 4096 | 1982.375 | 1948.929 | -1.69% |
| 6144 | 2629.688 | 2412.263 | -8.27% |
| 8192 | 3091.875 | 2801.061 | -9.41% |

四卡最大值由较慢的 rank 1/3 主导，不代表每张卡都加速。8192 tokens 时，rank 0/2 的 MegaMoe 设备时长相对各自 OFF 区间分别慢 9.53% / 8.18%，rank 1/3 则分别快 9.71% / 9.15%。

### 3.1 准备与等待区间

以下来自同一 profiling 时间线，未用普通计时减设备计时。

| rank / tokens | ON 准备 μs | 准备结束至 MegaMoe 开始 μs | OFF 含准备区间 μs | ON 含准备区间 μs |
| --- | ---: | ---: | ---: | ---: |
| 1 / 8192 | 212.025 | 266.375 | 3176.125 | 3232.125 |
| 3 / 8192 | 220.195 | 290.875 | 3205.375 | 3313.000 |

已验证：内核在较慢 rank 上节省的时间被准备和更长的区间空隙抵消。空隙只说明主流没有捕获到设备任务，不能直接归因为某个 Python/C++ 函数，也没有证明整网通过重叠完全隐藏了它。整网收益来源仍需真实路由和同版时间线验证。

分 rank 结果保留在 JSON 中，包括 OFF 五个主要算子的时长之和、区间其他任务和空隙。不能仅凭不同 rank 的时长差异宣称硬件故障。统计前强制验证每个 shape、每个 rank 恰好 61 次 OFF / ON；只取零起始索引 31–40、51–60。OFF 每次区间必须包含两次 GroupedMatmul、一次 Swiglu，并验证单流因果顺序，不把 HCCL 的逻辑/物理重复记录相加。

## 4. 精度与边界

普通计时 4 shapes × 3 scopes × 4 ranks = 48 项，BF16 输出逐位一致；其中 32 项是本卡部分输出、16 项是最终 AllReduce 输出。每项双方各做 20 次连续调用一致性检查，计时后再次检查输出及原始输入未被修改。独立 profiling 的 16 项也完成同样检查。这证明这些合成用例通过，不是全输入整网精度证明。

整网此前四轮吞吐 +1.388% / +1.332%、TTFT -5.173% / -3.728% 是不同测量范围。当前单算子结果不能被写成整网收益已由内核加速解释；共享专家重叠、真实路由分布和整网关键路径的贡献尚未做同版消融归因。

## 5. 固定版本与复现

- vLLM Ascend source: `481b9570b7bf83b483f81d1a5450515e5f2a0e4a`
- CANN kernel source: `23617531fd023f5a90612e6bddb8d21f951af234`
- CANN ND kernel SHA256: `325ec7a0b8060e8070f8156c8a685003d3be4ffd1a3b7a8f54051fcf6154a42b`
- extension SHA256: `3d0fe0d4dd3f86a5917f701ed434f176e8df1be4a95031f3e90500fe55bfb705`
- 容器：`mm_bf16_serve_20260907`，ID `a2d64204a8c4a69733be92c6cc606dd67493691dd2a5ca6ac46451b1c28141ca`，镜像 `sha256:8dd949bff550c8e33211bbf6f0daa899c13eab3a1a140105bb2001509a6b568b`。
- 模型对应维度：hidden=2048、intermediate=512、256 experts、TopK=8、TP/EP=4、BF16 ND；物理卡 0–3。

在原机器保留当前 runtime/package，先确认 0–3 卡、8001/29541 端口、worker 空闲。下列命令仅复现单算子，不启动整网服务。每次指定新输出目录，不复用旧结果。

```bash
set -euo pipefail
TASK=/data1/megamoe_gain_20260905/bf16_e2e_20260907/v31_operator_remeasurement_20260913
GUARD=/data1/megamoe_gain_20260905/bf16_e2e_20260907/user_tp4_v31_overlap/preflight_devices.py
OUT=/data1/megamoe_gain_20260905/bf16_e2e_20260907/v31_operator_recheck_NEW
exec 9>/var/lock/megamoe_q36.lock
flock -n 9
test ! -e "$OUT"
docker exec mm_bf16_serve_20260907 python3 "$GUARD"
docker exec mm_bf16_serve_20260907 bash "$TASK/run_env.sh" \
  python3 "$TASK/bench_v31_operator.py" --validate-only
mkdir -p "$OUT/timing" "$OUT/profile"
docker exec mm_bf16_serve_20260907 bash "$TASK/run_env.sh" \
  timeout --signal=TERM --kill-after=25s 720s \
  torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29541 \
  "$TASK/bench_v31_operator.py" "$OUT/timing" timing
docker exec mm_bf16_serve_20260907 python3 "$GUARD"
docker exec mm_bf16_serve_20260907 bash "$TASK/run_env.sh" \
  timeout --signal=TERM --kill-after=25s 720s \
  msprof --output="$OUT/msprof" --ascendcl=on --runtime-api=on --task-time=on --ai-core=off \
  torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29541 \
  "$TASK/bench_v31_operator.py" "$OUT/profile" profile
docker exec mm_bf16_serve_20260907 python3 "$GUARD"
```

`run_env.sh`、benchmark/reference、源码/二进制哈希清单、原始采样 JSON、解析后的 msprof task 记录、汇总 JSON 和分析脚本随结果包保存。脚本依赖当前机器的固定 runtime；迁移机器需要先按清单验证依赖，不能直接宣称是独立可安装包。

本轮计时及 profiling 已正常退出，postflight 检查通过，0–3 卡及测试端口已释放。没有修改运行时源码或内核，没有重启其他服务。
