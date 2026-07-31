# mooncake-kvpool-ascend

在 vllm-ascend（昇腾 NPU）上部署 Mooncake KV Cache Pool（含 SSD 分层）的配置、压测脚本与实测结论。

结论先行：**在 Atlas 800I A2 上，单机内的 Mooncake KV 池能否工作，取决于 NPU RoCE 网口的布线拓扑，而非卡间 HCCS 互联。** 详见 [docs/DIAGNOSIS.md](docs/DIAGNOSIS.md)。上线前务必先跑 `scripts/preflight_check.sh`。

## 目录

```
configs/    mooncake.json 变体 + 运行时环境变量
scripts/    前置检查、master 启动、引擎启动、压测
docs/       根因分析与部署手册
results/    实测数据
```

## 快速开始

```bash
export MCSSD_ROOT=/data1/mcssd          # 工作目录（配置/日志/SSD/结果）
export MCSSD_MODEL=/data1/Qwen3-14B     # 模型路径
export MCSSD_TP=2                       # TP 大小
export MCSSD_DEVICES=0,1                # 可见 NPU

# 0. 前置检查——不通过就不要往下走
bash scripts/preflight_check.sh

# 1. 基线（无 KV 池）
bash scripts/serve_launch.sh a
bash scripts/bench_run.sh a_cold 64     # 冷路径：64 个互不重复的前缀
bash scripts/bench_run.sh a_warm 8      # 热路径：8 个前缀被复用 8 次

# 2. DRAM KV 池
bash scripts/master_start.sh dram
bash scripts/serve_launch.sh b
bash scripts/bench_run.sh b_warm 8

# 3. SSD 分层（小 DRAM 池强制下沉）
bash scripts/master_start.sh ssd
bash scripts/serve_launch.sh c
bash scripts/bench_run.sh c_warm 8
```

## 环境要求

| 组件 | 版本 |
|---|---|
| vllm-ascend | main（SSD offload 需 >= v0.21.0rc1） |
| mooncake-transfer-engine-npu | >= 0.3.11.post1，**建议 0.3.12.post1** |
| CANN | A2 >= 8.5.0 / A3 >= 9.0.0 / A5 >= 9.1.0 |
| glibc | >= 2.35（wheel 为 manylinux_2_35） |

```bash
pip install mooncake-transfer-engine-npu==0.3.12.post1 \
  --extra-index-url https://mirrors.aliyun.com/pypi/web/simple
```

0.3.12.post1 相对 0.3.11 修复了三个 SSD 相关问题：master 重启后自动恢复、BatchOffload 竞态导致的 `INVALID_KEY`、`RemoveAll` 不删 SSD 文件产生孤儿文件。

## 关键坑位

1. **`PYTHONHASHSEED` 必须全节点一致**，否则 block hash 不同 → 池永远不命中，且无任何报错。
2. **SSD 配置分工不能写反**：`enable_ssd_offload` / `ssd_offload_path` 只能写 mooncake.json；`MOONCAKE_OFFLOAD_*` 只能用环境变量。写错位置静默失效。
3. **`MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` 默认 2TB 且是 per-rank**，不改的话 master 端显示的配额是物理盘的数倍，监控失真、盘满才暴露。
4. **逐出策略默认 `none`**（写满即失败），生产必须设 `lru`。
5. **`--default_kv_lease_ttl` 必须大于** `ASCEND_CONNECT_TIMEOUT` 和 `ASCEND_TRANSFER_TIMEOUT`，否则传输中途 lease 过期报 `LEASE_EXPIRED`。
6. **`--root_fs_dir` 与 `--enable_offload=true` 严禁同设**（前者是 legacy DFS 路径，冲突）。
7. **容器 PID 1 用 `sleep infinity` 不回收僵尸进程**：SIGKILL 掉 vLLM 后 NPU 显存不释放，需 `docker restart` 容器。
8. **`--host 127.0.0.1`**：vLLM 默认绑 0.0.0.0，在有公网 IP 的机器上会被扫描。

## 失败形态提醒

KV 池的典型故障是**写成功、读失败**：`External prefix cache hit rate` 显示 99.7%、master 里 Keys 正常增长，但每个请求都在等传输超时后走 `kv_load_failure_policy=recompute` 兜底。表现为**监控全绿而性能暴跌**（实测比不开池差 18.5 倍）。只看命中率指标发现不了，必须核对日志中的 `TRANSFER_FAIL` / `error: -800` 计数。
