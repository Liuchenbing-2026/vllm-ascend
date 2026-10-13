# 8000/1 并发256的TQ写路径优化（进行中）

| 项目 | 当前实测状态 |
|---|---|
| 目标 | 同卡、同KV预算下TQ开关整网TTFT/吞吐，兼顾20/20和2000/200 |
| 源码 | perf/tq-prefill-store-20261013；基础f796d8b8e59a5939014d7d982cad546259d63c2a |
| 改动 | 原Ascend C量化算法保持，4-bit pack与scatter融合为一个Triton-Ascend kernel；2/3-bit沿用原路径 |
| 已验证 | NPU全cache逐字节门禁78项通过，含随机/反序/连续slot、尾块、未改槽位、任意BF16位模式、真实量化器 |
| 未验证 | 新writer整网性能和greedy token对照正在后台实跑；未声称超过BF16 |

旧writer在8000/1、并发256、4096 token预算的3秒采集中走ScatterUpdateAiCPU，两个rank平均204/206μs；量化平均116/117μs。BF16原生scatter候选逐字节通过但更慢，因此未采用。调度预算8192的并发64筛选较旧TQ4096吞吐提升8.27%，但仍比相同8192预算BF16慢16.31%，不代替并发256正式数据。

## 环境、依赖与运行

复用容器tq-opt-20261011-model，宿主physical2/5、910B4-1/driver26.0.rc1、TP2、真实Qwen3-30B-A3B BF16权重。镜像vllm-ascend:dspark-a2-028，本地image ID sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14，RepoDigests=[]；公共pull地址未核实。CANN9.1.0、torch2.10.0+cpu、torch_npu2.10.0.post4、triton-ascend3.2.2。框架未改：vLLM2cf0a6915ce544dc493a0990f2ea38d81601128a、Ascend96df623103f921df1a4488170d106f36689acb59。完整基础镜像之外的安装构建依赖与原镜像容器创建步骤沿用[固定前阶段复现交付](https://github.com/Liuchenbing-2026/Model_test/blob/76b6ac7ef3e10c981e2c9bdb4262ce04af5a6d56/项目交付/任务列表/任务23%20Qwen3-30B-A3B-TurboQuant-128K双卡性能对比/README.md)。需准备/ws/.venv、/ws/source、/ws/c128_20261012插件、/root/kvtq_integration、已安装custom OPP、/ws/torch-extensions和/models/Qwen3-30B-A3B；镜像本身不是完整这些依赖。

实际服务完整命令和环境变量在[scripts/serve.sh](scripts/serve.sh)，实际后台监督/资源冲突门禁在[scripts/host.py](scripts/host.py)。与旧TQ和BF16同卡顺序对照，max_num_seqs512、max_model_len8192、max_num_batched_tokens4096、KV25769803776 bytes/card、eager、prefix/async关闭。8000/1并发256、20/20并发16、2000/200并发60，各模式各场景1预热+3正式；精度和profile排除。

```bash
# 容器内，候选文件已部署/ws/diagnosis_20261013/candidate_write/integration
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,5 VLLM_ASCEND_KVTQ_BITS=4
export TORCH_EXTENSIONS_DIR=/ws/torch-extensions
export PYTHONPATH=/ws/diagnosis_20261013/candidate_write/integration:/ws/source/vllm:/ws/source/vllm-ascend
/ws/.venv/bin/python /ws/diagnosis_20261013/scripts/verify_write.py
# 宿主启动的后台正式对照；host.py等待空闲2/5，SSH断开继续
nohup python3 -I -u /data02/tq-opt-20261011/write_ab_20261013/scripts/host.py > /data02/tq-opt-20261011/write_ab_20261013/logs/host.log 2>&1 < /dev/null &
# 实际监督器执行
# docker exec -e TASK_NPU_CARDS=2,5 tq-opt-20261011-model bash -lc 'source /usr/local/Ascend/cann-9.1.0/set_env.sh && exec /ws/.venv/bin/python -u /ws/write_ab_20261013/scripts/run_write_ab.py'
```

量化/读数值不变，新writer要求contiguous原66-byte存储、有效且互不重复的slot，沿用框架num_actual_tokens过滤；每个kernel program写不相交行，逐字节门禁覆盖跨program相邻slot与任意BF16位模式。不同图捕获或CPU设备未支持；本来store模式就要求enforce-eager。未从干净镜像从头复跑，本阶段只有原环境实跑与静态交付检查，不声称CI通过。

完整原始数据、失败台账和候选不采用原因见Model_test任务23中本阶段目录；旧失败、超时、关闭清理问题不删除。本轮没有提交上游PR或改写既有历史。
