# 实测结果

环境：m18（Atlas 800I A2，Ascend 910B4-1）· Qwen3-14B · vllm-ascend main（vllm 0.25.1）
负载：`vllm bench serve --dataset-name prefix_repetition`，8192 token 前缀 + 256 后缀，
输出 128 token，64 请求，并发 8，seed 42

## A 组：基线 vLLM（引擎内 prefix caching，无 KV 池，TP=2）

| 轮次 | 说明 | Median TTFT | Mean TTFT | P99 TTFT | 吞吐 (tok/s) |
|---|---|---|---|---|---|
| a_cold | 64 个互不重复前缀（零复用） | **2190.42ms** | 4424.95ms | 18315.11ms | 3383.72 |
| a_warm1 | 8 前缀 × 8 复用（首轮，含填充） | 296.07ms | 934.50ms | 5499.79ms | 5538.61 |
| a_warm2 | 8 前缀 × 8 复用（全命中） | **306.74ms** | 328.21ms | 558.23ms | **5943.72** |
| a_restart | 重启引擎后重放同负载 | 294.66ms | 1067.20ms | 6465.98ms | 5492.92 |

**引擎内前缀缓存收益**（a_cold → a_warm2）：

- Median TTFT **7.1x**（2190 → 307ms）
- P99 TTFT **32.8x**（18315 → 558ms）
- 吞吐 **1.76x**（3384 → 5944 tok/s）

a_restart 的 mean TTFT 回升到 1067ms（vs 全命中的 328ms），反映**缓存随进程丢失**后
需要重新填充——这正是 KV 池要解决的问题，也是 KV 池收益的理论上界所在。

## B/C 组：Mooncake KV 池 —— 全部失败（环境阻断）

根因见 [../docs/DIAGNOSIS.md](../docs/DIAGNOSIS.md)：A2 上 Mooncake 传输走 RoCE 网口，
本机为双机直连布线，机内卡间无物理通路。

| 配置 | 成功请求 | Median TTFT | 吞吐 | 传输失败次数 |
|---|---|---|---|---|
| TP2 + `HCCL_INTRA_ROCE_ENABLE=1` | 64/64 | 40529.20ms | 848.80 | 1590 |
| TP2 + 关闭 RoCE | 1/64 | 120426.12ms | 23.42 | 788 |
| TP2 + `preferred_segment=true` | 1/64 | 120415.79ms | 23.42 | 1576 |
| `protocol=tcp` | — | 引擎拒绝启动 | — | `MooncakeBackend does not support protocol 'tcp'` |
| TP1 单卡 | 64/64 | 4126.77ms | 3320.68 | 4210 |

所有轮次的 `External prefix cache hit rate` 均为 99.6–99.7%，master 侧
`Mem Storage 12.5GB / Keys 1268` 正常增长——**写入始终成功，失败的只有读取**。
请求靠 `kv_load_failure_policy=recompute` 兜底完成，结果正确但每次先白等 10 秒超时。

对比基线冷启动（2190ms），最好的一组池配置（TP1，4127ms）仍慢 **1.9 倍**，
最差的一组（TP2 关 RoCE，120426ms）慢 **55 倍**。

## 待补测（环境修复后）

- B 组：DRAM 池命中 TTFT、跨引擎重启的缓存存活性
- C 组：SSD 层命中 TTFT、`rank_N/` 落盘验证、master 重启后 SSD 自动恢复
- 四路径对比：冷 / HBM 命中 / DRAM 池命中 / SSD 命中
