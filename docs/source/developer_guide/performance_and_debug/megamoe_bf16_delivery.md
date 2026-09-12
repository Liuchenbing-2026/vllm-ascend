# Qwen3.6 BF16 四卡 MegaMoe 交付与复现

本交付包含两个配套仓库，复现目标是 18 机当前已验证容器中的 Qwen3.6-35B-A3B、TP4/EP4、BF16 场景。历史八卡 W8A8、两卡 BF16 实验不能与这里的收益混用。

## 1. 分支与源码

| 仓库 | 交付分支 | 实际验收源码提交 |
| --- | --- | --- |
| [vLLM Ascend](https://github.com/Liuchenbing-2026/vllm-ascend) | `codex/megamoe-bf16-eval-20260907` | `481b9570b7bf83b483f81d1a5450515e5f2a0e4a` |
| [CANN ops-transformer](https://github.com/Liuchenbing-2026/ops-transformer) | `codex/megamoe-bf16-e2e-20260907` | `23617531fd023f5a90612e6bddb8d21f951af234` |

交付提交还包含历史合并、文档与证据；运行时代码与上述验收提交的对应文件哈希一致。原远端的 `replicated_input` 是早期未验收实验，现已由显式 dispatch / TP4 local-partial 接入取代。双方历史保留，当前复现不启用旧实验。

vLLM 分支包含 local-partial prepare/finalize、Triton 输入准备、容量和配置检查、共享专家重叠兼容、回退路径及对应测试。CANN 分支包含本卡专家路由、确定的 K 遍历、packed ND 权重扩展、unpermute 行范围优化及相关测试。不要只拉取其中一个仓库。

```bash
git clone --branch codex/megamoe-bf16-eval-20260907 \
  https://github.com/Liuchenbing-2026/vllm-ascend.git
git clone --branch codex/megamoe-bf16-e2e-20260907 \
  https://github.com/Liuchenbing-2026/ops-transformer.git
git -C vllm-ascend rev-parse HEAD
git -C ops-transformer rev-parse HEAD
```

## 2. 本机复现所需环境

- 主机：`183.236.60.18`，使用物理 NPU 0、1、2、3。
- 容器：`mm_bf16_serve_20260907`，ID `a2d64204a8c4a69733be92c6cc606dd67493691dd2a5ca6ac46451b1c28141ca`。
- 镜像 ID：`sha256:8dd949bff550c8e33211bbf6f0daa899c13eab3a1a140105bb2001509a6b568b`。
- 当前运行时：vLLM 0.27.1、torch-npu 2.10.0.post4、CANN 9.1、Python 3.12.13。它是用户授权使用的当前容器。
- 模型：`/data2/weights/Qwen3.6-35B-A3B`，1045 个 BF16 张量；模型文件和对应状态清单由准备器检查。
- 服务端口 8001；算子测试进程组端口 29541。
- 源码目录：`/data1/megamoe_gain_20260905/bf16_e2e_20260907/model_v31_runtime/vllm-ascend-bf16`。
- CANN vendor：`/data1/megamoe_gain_20260905/bf16_opt_20260907/package_v30/packages/vendors/mmbf16v30_transformer`。
- 扩展目录：`/data1/megamoe_gain_20260905/bf16_opt_20260907/torch_extension_v26`。

内核对象 SHA256 为 `325ec7a0b8060e8070f8156c8a685003d3be4ffd1a3b7a8f54051fcf6154a42b`；扩展 `npu_mega_moe.so` SHA256 为 `3d0fe0d4dd3f86a5917f701ed434f176e8df1be4a95031f3e90500fe55bfb705`。扩展源码对应 `a97d4ace25f5f8a51ca4584ebab941249022d5c7`，之后的 V30 提交没有改动该扩展。

模型权重和完整 CANN/驱动/镜像属于运行环境，不在 Git 包内。下面的复现脚本针对保留的 18 机部署；换机不能只更改 IP 就认为环境等价。移植时需要恢复相同依赖、设备拓扑和制品，再执行完整精度及性能验收。

## 3. 从分支中的输入包复现整网

[整网输入包](megamoe_bf16_v31_reproduction_inputs.zip) SHA256：`717ef6e37d0944e4570947c1301cdc33918d560c4fd70df410ced47f98025e86`。

包内包含实际运行的共同启动脚本、OFF/ON 配置、现有 `vllm bench serve` 调用、固定精度请求及参考输出、三次重复稳定性检查、设备/端口/worker 检查、进程归属清理、结果校验和完整文件哈希。准备阶段不启动服务。

在 18 机 Linux 宿主机执行，`REPO` 指刚拉取的 vLLM Ascend 仓库。`delivery_repro_01`、`bf16_repro_delivery_A` 和 `bf16_repro_delivery_B` 必须是尚不存在的新目录；不要覆盖旧结果。

```bash
set -euo pipefail
REPO=/absolute/path/to/vllm-ascend
BASE=/data1/megamoe_gain_20260905/bf16_e2e_20260907
INPUTS="$BASE/delivery_repro_01"
A="$BASE/bf16_repro_delivery_A"
B="$BASE/bf16_repro_delivery_B"
ZIP="$REPO/docs/source/developer_guide/performance_and_debug/megamoe_bf16_v31_reproduction_inputs.zip"
printf '%s  %s\n' \
  717ef6e37d0944e4570947c1301cdc33918d560c4fd70df410ced47f98025e86 "$ZIP" | sha256sum -c -
test ! -e "$INPUTS"
mkdir "$INPUTS"
python3 -m zipfile -e "$ZIP" "$INPUTS"
python3 "$INPUTS/prepare_current_machine.py" --work "$A" --order off-on
python3 "$INPUTS/prepare_current_machine.py" --work "$B" --order on-off
bash "$A/run_performance.sh"
bash "$A/collect_results.sh" > "$A/collected_results.log"
bash "$B/run_performance.sh"
bash "$B/collect_results.sh" > "$B/collected_results.log"
```

顺序是 OFF A1 → ON B1 → ON B2 → OFF A2。不要在计时期间同时跑 profiling、编译、格式检查或大型离线分析。脚本发现卡或端口忙时应先处理占用归属，不能绕过门禁。

## 4. 对齐项与验收标准

双方同一运行时、源码、模型、0–3 卡及 TP4/EP4，DP/PP/PCP 均为 1。服务参数固定 max-num-seqs=32、max-model-len=131072、max-num-batched-tokens=8192、显存利用率 0.90、seed=1024、BF16；启用异步调度、chunked prefill、CPU binding、FULL_DECODE_ONLY，关闭 prefix cache。

双方 `multistream_overlap_shared_expert=true`、`fuse_muls_add=true`、`enable_npugraph_ex=true`。差异仅为以下配套开关：

| 开关 | OFF | ON |
| --- | ---: | ---: |
| `MEGAMOE` | 0 | 1 |
| `enable_fused_mc2` 启动值 | 0 | 2 |
| `mega_moe_local_partial` | false | true |

压测采用输入 4096/6144、输出 256、并发 32、各 160 请求、random-range-ratio=0、seed=1024、temperature=0、ignore-eos。实际 input_lens/output_lens 必须逐项相同；每组必须 160 请求成功、总输出 40960、每条输出长度 256、零错误。

每轮先通过固定请求输出及 logprobs 对照和三次重复检查。汇总需 controller/evaluation/stop 全部为 0；仅出现 complete 文件或单个进程退出不足以判定通过。普通精度用例不等价于全输入模型无损证明。

吞吐增幅计算为 `(ON 均值 / OFF 均值 - 1) × 100%`；TTFT 降幅为 `(1 - ON 均值 / OFF 均值) × 100%`。分别报告两次配对、四轮数值与均值。历史验收使用两次配对均超过 0.2% 且最慢 ON 吞吐高于最快 OFF 的实用检查；这不是统计显著性证明，也不是每次都能达到固定百分比的承诺。

## 5. 已测结果与单算子边界

首次四轮整网对照：4K/6K 吞吐分别提高 1.388% / 1.332%，TTFT 分别下降 5.173% / 3.728%。完整原始配对与精度证据见[首次验收证据包](megamoe_bf16_v31_validation_evidence.zip)。

最新同版单算子复测见[单算子报告](megamoe_bf16_v31_operator_report.md)及[单算子输入和证据](megamoe_bf16_v31_operator_remeasurement.zip)。大 token 的四卡瓶颈设备区间有收益，8192 tokens 约快 9.4%；含准备及最终 AllReduce 的独立调用仍慢约 7.3%。不能把整网提升全部归因为单算子计算加速，准备/下发区间、真实路由分布和共享专家重叠的贡献尚未完成消融归因。

2026-09-13 使用分支内原始输入包，在同一容器重放 OFF A1 → ON B1 → ON B2 → OFF A2，结果如下。四轮精度和重复输出检查均通过；八组压测每组 160 请求成功、输出 40960 tokens、无错误；结束后设备、worker 和端口检查通过。

| 输入 | OFF A1 tok/s | ON B1 tok/s | ON B2 tok/s | OFF A2 tok/s | 均值收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 519.6903 | 523.1435 | 524.5024 | 519.0467 | +0.858% |
| 6144 | 425.7645 | 432.0466 | 433.0152 | 426.7263 | +1.475% |

| 输入 | OFF 平均 TTFT ms | ON 平均 TTFT ms | TTFT 变化 | 平均 TPOT 变化 |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 1862.3819 | 1960.1751 | +5.251%，变慢 | -1.704% |
| 6144 | 2353.4160 | 2291.2170 | -2.643%，变快 | -1.299% |

本次吞吐小幅收益得到复现，之前的 4K TTFT 改善没有复现。不能据首次结果承诺 TTFT 稳定提升。两次配对的 4K 吞吐分别提高 0.664% / 1.051%，6K 提高 1.475% / 1.474%。

[交付包重放的完整证据](megamoe_bf16_v31_delivery_replay.zip)包含八份逐请求压测结果、实际命令、四轮服务日志、精度输出、清理回执和可重新计算汇总的脚本。SHA256：`c31544845c899a196e000c7a59ec99853a6772cc6f7b41293620d0e1a9c05eb2`。包内 README 给出汇总命令。

源码闭合检查确认运行目录中 553 个受版本控制的框架文件与交付源码存档完全一致；生成的 ABI 制品单独记录哈希和 `ldd` 解析结果。CPU 单元测试 225 项通过；单算子与边界用例的精度范围见对应报告。这些有限用例不能证明任意输入的数学无损。
