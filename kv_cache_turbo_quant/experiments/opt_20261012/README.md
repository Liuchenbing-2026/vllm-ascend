# TQ4 读取继续优化：20/20 不退化、2000/200 提升

| 项目 | 当前结果 | 待验证 |
|---|---|---|
| 20/20、并发16 | 保留上一轮V1原始读取kernel及分配路径 | 本轮3次整网配对指标 |
| 2000/200、并发60 | 筛选中V1读取3.889ms → 分页映射+32行读取2.790ms，下降28.26% | 整网吞吐、TTFT、TPOT |
| 精度 | 新混合读取7/7逐位通过；真实接入15/15；2/3-bit回退各1/1 | 四组真实权重贪心输出比较 |
| 正式压测 | 后台流水线已启动；BF16→上一轮V1→候选，各场景预热1组、正式3组 | 未完成前不宣称新的整网收益 |

## 基线与采用条件

上一轮已交付优化TQ为56.931302 tok/s（20/20/16）、120.062845 tok/s（2000/200/60），固定源码[eb63397b770b0cc5c02b2ad4b17aee8c1c8cf865](https://github.com/Liuchenbing-2026/vllm-ascend/commit/eb63397b770b0cc5c02b2ad4b17aee8c1c8cf865)。本轮在相同宿主、模型、参数下重新测V1与候选；仅短场景吞吐不降、长场景吞吐提升且贪心样本一致时采用，TTFT/TPOT分别展示。历史数值不能替代本轮对照。

4-bit量化及66-byte格式不变，也不增加持久BF16 KV副本。短上下文（最长序列≤页大小）使用V1 kernel；长上下文先生成当前逻辑token→物理token映射，再由32行解量化kernel读取。映射每次调用重新生成，消费当前block table，不缓存物理页而返回过期KV。映射和解量化均为Triton-Ascend，原Ascend C量化算子保持不变。

算子筛选事件区间覆盖10次Python dispatch，9组取中位数；这是筛选计时，包含主机dispatch间隔，不能等同kernel-only profiling或整网吞吐。原始参照、全部7个shape、integer逐位判据未修改；失败候选与原始日志保留。

## 代码与实际环境

| 项目 | 固定版本 / 路径 |
|---|---|
| 任务分支 | `perf/tq-read-throughput-20261012`，基础`eb63397b770b0cc5c02b2ad4b17aee8c1c8cf865`；交付SHA见本分支提交页 |
| 宿主 / 双卡 | `112.29.145.3`，910B4-1，物理卡2/5，driver26.0.rc1 |
| 保留容器 | `tq-opt-20261011-model`，上一轮新建且从源码编译的专用容器；本轮复用、没有重新编译vLLM/Ascend |
| 基础镜像 | `vllm-ascend:dspark-a2-028`，本地image ID`sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`；RepoDigests为空，公开pull地址未验证 |
| vLLM | 0.28.0，固定源码`2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| vLLM-Ascend | 0.28rc1，固定源码`96df623103f921df1a4488170d106f36689acb59` |
| Catlass | `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |
| 原始TQ | 58文件`8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b`；Ascend C量化库及torch glue复用原构建 |
| 其他依赖 | CANN9.1.0、Python3.12.13、torch2.10.0+cpu、torch_npu2.10.0.post4、triton-ascend3.2.2（import版本3.2.0） |
| 模型 | 真实Qwen3-30B-A3B权重，TP2；`/models/Qwen3-30B-A3B`只读 |
| 启动条件 | BF16/eager、prefix/async OFF、maxlen4096、batchedtokens4096、maxseq64、KV每卡24GiB、YaRN4 |
| 本轮根目录 | 宿主`/data02/tq-opt-20261011/opt_20261012` → 容器`/ws/opt_20261012` |

完整镜像/依赖/源码构建、容器创建和权重校验步骤复用[上一轮固定复现归档](https://github.com/Liuchenbing-2026/vllm-ascend/blob/eb63397b770b0cc5c02b2ad4b17aee8c1c8cf865/kv_cache_turbo_quant/experiments/opt_20261011/README.md)，新机器须先完成其中安装和基础构建；现有容器marker检查不能宣称从零复现已通过。本轮新Triton kernel通过独立cache即时编译。

## 实际运行入口

宿主已执行的脱离SSH后台入口：

```bash
nohup python3 -I -u /data02/tq-opt-20261011/opt_20261012/scripts/model_host.py > /data02/tq-opt-20261011/opt_20261012/model-host-controller.log 2>&1 < /dev/null &
```

`model_host.py`等待卡2/5空闲后运行`model_pipeline.py`，每20秒归档实际NPU所有者；出现外部进程时只中止自身流水线。原微测日志在`attempts/`保留。流水线保存每条完整command、原始log和exit；独立run_id不会覆盖前一轮数据。

容器内实际精度和压测命令（CANN环境须先加载）：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,5 TASK_NPU_CARDS=2,5
export VLLM_ASCEND_KVTQ_BITS=4 VLLM_ASCEND_KVTQ=0 VLLM_ASCEND_KVTQ_STORE=0
export TORCH_EXTENSIONS_DIR=/ws/torch-extensions
export TRITON_CACHE_DIR=/ws/opt_20261012/model-triton-cache
export PYTHONPATH=/ws/opt_20261012/candidate/integration:/ws/source/vllm:/ws/source/vllm-ascend
/ws/.venv/bin/python /ws/opt_20261011/verifier/verify.py --op_name paged_read --verify_dir /ws/opt_20261012/verify --output /ws/opt_20261012/verify/verify_result.json --timeout 1800
/ws/.venv/bin/python /ws/opt_20261012/scripts/verify_integration.py
/ws/.venv/bin/python /ws/opt_20261012/scripts/run_model_ab.py --no-profile
```

真正起服入口是`scripts/serve_optimized.sh`，包含完整参数；`original`模式明确指上一轮V1，`optimized`指本轮混合候选，`bf16`关闭TQ。bench_client.py和run_matrix.py复用上一轮固定归档的`/ws/scripts`。每项场景执行同形状全并发预热1组、正式3组；输出tokens/s=正式输出tokens之和/完整HTTP cohort耗时之和，TTFT/TPOT为逐请求均值。正式测试不启动profiler。四组单请求贪心token一致性单独计时并排除，不能代替语义质量评测。

## 问题与当前状态

见`issues.json`、运行时追加的`issues.jsonl`及`evidence/`中的原始失败。int16 Gather不支持、64行UB超限、直接全局查表/选择树性能回退均保留，失败候选不采用。上一阶段O01–O36见固定归档，恢复不等于全部根因已闭环。后台当前PID/status/exit须现场核验，不能把已结束的历史PID写成仍运行。
