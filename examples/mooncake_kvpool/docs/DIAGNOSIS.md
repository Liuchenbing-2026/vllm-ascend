# 根因分析：为什么单机 Mooncake KV 池在 m18 上无法工作

测试日期：2026-07-30/31
硬件：Atlas 800I A2，8× Ascend 910B4-1（64GB HBM/卡），kylin10-6018
软件：vllm-ascend main（vllm 0.25.1）、mooncake-transfer-engine-npu 0.3.12.post1、CANN 9.0.0

## 结论

**Mooncake 在 A2 上的 KV 传输只走 NPU RoCE 网口；本机 RoCE 是双机直连布线，机内卡间没有物理通路，因此单机 KV 池无法工作。**

这与卡间 HCCS 互联无关——HCCS 是健康的，且 KV 池根本不用它。

## 两条独立路径

| 路径 | 载体 | 用途 | 本机状态 |
|---|---|---|---|
| HCCS | 板载互联 | HCCL collective（TP allreduce 等） | ✅ 8 卡全互联，TP=2 推理跑满 5944 tok/s |
| RoCE 网口 | hccn / 光模块 | HCCL P2P、Mooncake ADXL 传输 | ❌ 机内不可达（见下） |

A2 的 Mooncake 传输被官方文档要求设 `HCCL_INTRA_ROCE_ENABLE=1`——变量名直译就是"让机内通信走 RoCE"。HCCS 直传（`ASCEND_ENABLE_USE_FABRIC_MEM`）是 A3 才有的能力。因此在 A2 上，即便 HCCS 全互联，Mooncake 也不会使用它。

## 拓扑证据

```
dev4 光模块: present (HUAWEI OM3538SX101, 51C)     <- 已接线
dev4 link  : UP（自 2026-07-18 起稳定）              <- 链路活跃
dev4 ARP   : 172.16.36.101 at fc:18:03:57:0b:30    <- 对端存在
dev4 LLDP  : System Name    = sealos.hub
             Management IP  = 172.16.36.101
             System Desc    = AscendNPU Linux ...   <- 对端是另一台昇腾机器
dev4 netdetect: 0.0.0.0                             <- 未配置探测地址
```

八张卡的 IP 分属八个不同 /24（172.16.32.102 … 172.16.39.102），每张卡的 LLDP 对端是
另一台机器（.101）的同序号卡。这是**双机直连拓扑**：本机 dev_i ←→ 对端 dev_i，
每对卡独占一个网段。

推论：dev4 的网口只能到达对端机器的 dev4，**物理上到不了本机的 dev5**。
`hccn_tool -i 4 -ping -g address 172.16.37.102` 100% 丢包，与拓扑一致。

### 关于 net_health=Init

八张卡的 `net_health` 均为 `Init`，但这**不代表链路故障**：`netdetect address` 为
`0.0.0.0`，健康检查没有探测目标，状态就停在 Init。配置 netdetect 后会转为 Success，
但那验证的是**跨机**连通性，不改变机内不可达的事实。

## 失败表现

写路径始终成功，只有读失败：

| 指标 | 值 | 说明 |
|---|---|---|
| `External prefix cache hit rate` | 99.7% | 看起来完美 |
| master `Mem Storage` / `Keys` | 12.5GB / 1268 | 数据确实写进池了 |
| 日志 `TRANSFER_FAIL` / `error: -800` | 数千次 | 真相在这里 |

读失败后按 `kv_load_failure_policy=recompute` 回退重算，请求**结果正确但极慢**——
每次都要先等满 10 秒传输超时。这是最危险的故障形态：**监控全绿、命中率漂亮、
性能暴跌 18.5 倍**，只看指标发现不了。

## 排查过程（五种配置全部失败）

| # | 配置 | 结果 |
|---|---|---|
| 1 | TP2 + `HCCL_INTRA_ROCE_ENABLE=1`（官方 A2 配置） | median TTFT 40529ms，1590 次 TRANSFER_FAIL |
| 2 | TP2 + 关闭 RoCE（日志确认 `roce_mode=false`） | median TTFT 120426ms，64 请求仅 1 个成功 |
| 3 | TP2 + `preferred_segment=true` | 同上，1576 次失败 |
| 4 | `protocol=tcp` | `MooncakeBackend does not support protocol 'tcp'`，仅支持 ascend |
| 5 | TP1 单卡（消除跨卡传输） | 64/64 完成但 4210 次传输失败，全靠 recompute 兜底，TTFT 4127ms（比基线冷启动 2190ms 还差） |

第 5 项值得注意：即使 TP=1，Mooncake 仍把 client 与 segment 视为需要经传输引擎通信的实体，
未能走纯本地路径。

## 未验证的部分

1. **关闭 RoCE 后 ADXL 走什么路径**：日志显示 `roce_mode=false` 已生效但传输仍失败
   （error -800）。是"A2 天生没有 HCCS 直传路径"还是"有但需额外配置"，需读
   `ascend_direct_transport.cpp` 源码确认。本文按前者推断，依据是官方文档的 A2/A3 能力分工。
2. **RoCE 端口是否可改配为机内互通**：理论上把八张卡改到同一网段可能让机内直连生效，
   但在点对点直连布线下，dev4 与 dev5 的光纤各自连向对端机器，没有物理路径，
   改 IP 无法解决。

## 可行的替代方向

既然布线是**双机直连且链路健康**，这套拓扑原本就是为跨机 RoCE 设计的。
**跨机 PD 分离**（Prefill 在一台、Decode 在另一台，经 MooncakeConnectorV1 传 KV）
很可能直接可用，且这才是 Mooncake 的主要应用形态。

验证步骤（需要两台机器的操作权限）：
1. 确认对端 `.101` 机器身份
2. 各卡配置 netdetect 为对端同序号卡地址，确认 `net_health` 转 Success
3. `hccn_tool -i N -ping -g address <对端>` 验证跨机可达
4. 按 vllm-ascend PD 分离文档部署 1P1D

## 环境侧发现（与 Mooncake 无关，但值得修）

- **僵尸进程导致 NPU 显存不释放**：容器 PID 1 为 `sleep infinity`，不回收孤儿子进程。
  `kill -9` vLLM 后 `npu-smi` 仍显示 61GB 占用且无进程归属，需 `docker restart` 容器。
  建议容器入口用 `tini` 或 `bash -c 'trap ... wait'`。
- **vLLM 默认绑 0.0.0.0**：测试期间日志出现外部 IP（165.154.36.245）扫描
  `/sse`、`/mcp`、`/get_server_info` 等路径。有公网 IP 的机器应显式 `--host 127.0.0.1`。
