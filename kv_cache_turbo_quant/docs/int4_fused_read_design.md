# int4 融合读：可行性与设计（tq4 容量 3.88x 可用化）

> 日期：2026-10-12。状态：**设计阶段，未实施**。
> 背景：tq4（TurboQuant 4bit）KV 容量 3.88x（int8 的 ~2 倍），但读路径是 Python 分级解压，
> TPOT 劣化 16~146x，不可用。本文分析可用化的候选路径并给出分阶段计划。

## 约束回顾

- A2 FIA（paged attention）硬件 antiquant **只支持 int8**，不支持 int4（官方文档 + 实测）
- 因此"让硬件直接吃 int4 KV"不存在；任何可用方案都必须在软件/kernel 层把 int4 变回高比特表示
- tq4 存储格式：4bit 质心索引 + fp16 范数 + QJL 残差符号位（自定义打包，NZ 排布）

## 候选路径

### 路径 A：dequant-to-scratch + 标准 FIA（不可行，已否决）

每个 decode step 先把 running 序列的 int4 KV 解压成 bf16 scratch，再跑标准 FIA。

否决理由：
- scratch 大小 = 全部 running 序列的完整 KV（c32 × 128k 时即全量 bf16 KV），
  显存直接回到 baseline，容量收益归零
- 流量账：int4 读(0.25x) + bf16 写(1x) + FIA 再读 bf16(1x) > 直接读 bf16(1x)，
  带宽上也是负收益

### 路径 B：自定义 paged-attention kernel，kernel 内解压（唯一可行路径）

FlashDecoding 风格融合 kernel：分页加载 int4 KV block → UB 内解压
（质心 LUT + 逆旋转 + QJL 残差）→ 直接与 Q 做 attention（online softmax）。

收益模型（128k decode，memory-bound）：
- HBM 流量按 int4 计 = bf16 的 ~1/4 → 理论读带宽收益 ~4x（int8 FIA 已实测 2.6x，
  int4 应更高）
- 风险：解压是纯 vector 操作，A2 vector 吞吐是否能在 cube attention 的间隙
  overlap 住，决定净收益。若 vector 成为瓶颈，收益打折但不至于变负
  （int4 HBM 读本身已省 4x）

工作量评估：**周级**。需重写 paged attention 主循环（online softmax、
GQA head packing、NZ 地址、分页 indirection），是 FlashDecoding 级工程。

### 路径 C：混合位宽（妥协备选）

K 保持 int8（FIA 原生）、V 用 int4，或浅层 int8 / 深层 int4。
容量收益介于 2.0x~3.88x，读路径部分保留硬件 antiquant。
可作为路径 B 的回退档，但工程上仍需路径 B 的 kernel 才能读 int4 部分。

## 分阶段计划

| 阶段 | 内容 | 产出 | 量级 |
|---|---|---|---|
| 1 | 解压微基准 kernel：int4 paged 读 + UB 内解压 + 不落盘，测纯 HBM 带宽与 vector 解压吞吐 | 收益判别数据（是否 memory-bound 收益成立） | 1-2 天 |
| 2 | FlashDecoding 融合 kernel（若阶段 1 判别为正） | 可用 int4 读路径 | 周级 |
| 3 | 端到端接入 + 128k sweep + gsm8k | 三方对比更新 | 2-3 天 |

## 判别标准（阶段 1 出数据后）

- int4 paged 读 + 解压的端到端微基准时间 < bf16 FIA 读时间的 60% → 路径 B 值得做
- 若 vector 解压吞吐不足（> 80%），改走路径 C 或放弃 int4 可用化，int8 即终态