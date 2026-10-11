# TQ4 读取继续优化：20/20 不退化、2000/200 提升

本轮已完成并采用：相对上一轮V1，20/20/c16吞吐提高3.44%，2000/200/c60提高21.81%，TTFT和TPOT均下降。相同910B4-1双卡、真实Qwen3-30B-A3B权重、每卡24GiB KV预算，三轮正式对照。

| 场景 | 模式 | 输出tok/s | TTFT(s) | TPOT(ms) | 正式成功 |
|---|---|---:|---:|---:|---:|
| 20/20/c16 | BF16（TQ关闭） | 74.848 | 0.4175 | 202.063 | 48 |
| 20/20/c16 | 上一轮V1 TQ | 56.954 | 0.5175 | 267.306 | 48 |
| 20/20/c16 | 本轮混合TQ | 58.914 | 0.5020 | 258.286 | 48 |
| 2000/200/c60 | BF16（TQ关闭） | 244.012 | 4.1348 | 211.059 | 180 |
| 2000/200/c60 | 上一轮V1 TQ | 119.884 | 6.7503 | 447.941 | 180 |
| 2000/200/c60 | 本轮混合TQ | 146.026 | 6.5712 | 358.687 | 180 |

18组正式cohort、684个请求全部精确完成输入/输出usage，payload逐轮配对；四组贪心样本共440个token IDs与V1一致。读取7/7严格逐位、真实接入15/15、2/3-bit回退各1/1、原glue五组与roundtrip通过。独立复算见`audit.json`，原始数据见`evidence/final/results/model_20261011T170023Z/`。

短场景保留原kernel，3.44%是本次整网实测变化，不能归因为短核本身加速；本轮未重新采集profiling。新TQ整网仍低于关闭TQ的BF16：两场景分别为BF16吞吐的78.71%和59.84%。优化版128K、语义质量测评未测。

## 基线与采用条件

上一轮已交付优化TQ为56.931302 tok/s（20/20/16）、120.062845 tok/s（2000/200/60），固定源码[eb63397b770b0cc5c02b2ad4b17aee8c1c8cf865](https://github.com/Liuchenbing-2026/vllm-ascend/commit/eb63397b770b0cc5c02b2ad4b17aee8c1c8cf865)。本轮在相同宿主、模型、参数下重新测V1与候选；短场景吞吐不降、长场景吞吐提升且贪心样本一致的采用门禁已通过，TTFT/TPOT亦下降并分别展示。历史数值不能替代本轮对照。

4-bit量化及66-byte格式不变，也不增加持久BF16 KV副本。短上下文（最长序列≤页大小）使用V1 kernel；长上下文先生成当前逻辑token→物理token映射，再由32行解量化kernel读取。映射每次调用重新生成，消费当前block table，不缓存物理页而返回过期KV。映射和解量化均为Triton-Ascend，原Ascend C量化算子保持不变。

算子筛选事件区间覆盖10次Python dispatch，9组取中位数；这是筛选计时，包含主机dispatch间隔，不能等同kernel-only profiling或整网吞吐。原始参照、全部7个shape、integer逐位判据未修改；失败候选与原始日志保留。

## 代码与实际环境

| 项目 | 固定版本 / 路径 |
|---|---|
| 任务分支 | `perf/tq-read-throughput-20261012`，基础`eb63397b770b0cc5c02b2ad4b17aee8c1c8cf865`；实测代码交付`2a83edbdd576b1bd4a5cdea7c2c41b22feb6167a`，最终归档SHA见Task23固定链接 |
| 宿主 / 双卡 | `112.29.145.3`，910B4-1，物理卡2/5，driver26.0.rc1 |
| 保留容器 | `tq-opt-20261011-model`，上一轮新建且从源码编译的专用容器；本轮复用、没有重新编译vLLM/Ascend |
| 基础镜像 | `vllm-ascend:dspark-a2-028`，本地image ID`sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`；RepoDigests为空，公开pull地址未验证 |
| vLLM | 0.28.0，固定源码`2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| vLLM-Ascend | 0.28rc1，固定源码`96df623103f921df1a4488170d106f36689acb59` |
| Catlass | `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |
| 原始TQ | 58文件`8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b`；Ascend C量化库复用；torch glue由未改C++源码在新路径JIT重建（见下方实查） |
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

## 已完成状态、问题与限制

见`issues.json`、运行时追加的`issues.jsonl`及`evidence/`中的原始失败。int16 Gather不支持、64行UB超限、直接全局查表/选择树性能回退均保留，失败候选不采用。上一阶段O01–O36见固定归档，恢复不等于全部根因已闭环。本轮pipeline于2026-10-11T17:39:15.499576Z结束，容器内/宿主exit均0；2026-10-11T17:41:39Z实查卡2/5无进程、18377可绑定，保留容器只有sleep，其他卡作业未动。当前无本任务活动压测进程。


## 实际二进制与容量核验

`evidence/final/logs/model_20261011T170023Z/environment.log`记录真实依赖、加载源码目录及framework/native SHA。框架未重编译；原Ascend C opapi库SHA256 `089bafab61cad104a6f5aaff2e08e5a2b01bbd350853c7682d0dc9acd3074071`。torch glue因源文件路径改变发生JIT重建，候选.so SHA256 `a330a1c708c23524e845d44c8271b6b01b476aab94e0eefa5d5cc23e93199294`，build.ninja、mtime及源路径见`evidence/local/native-runtime-inspection.log`。V1与候选C++、store、plugin字节相同，不能说沿用同一torch glue二进制。

固定24GiB KV预算/卡，服务日志BF16 `524288` tokens、V1与候选均`2033536` tokens，容量3.878662109375倍；本轮继续使用原66-byte压缩KV，临时映射和当前attention的BF16工作区不构成持久BF16 KV副本。

## 验收与恢复复跑

本次实际启动和完整命令均保存在`evidence/final/background-launch.command.json`和`evidence/final/logs/`各`.command.json`。`original`始终指上一轮V1，不指最初18tok/s的原始读实现。基线V1源码随`evidence/final/baseline/integration`保存，候选源码随`evidence/final/candidate/integration`保存；生产`code/integration/kvtq_read.py`与已验证候选SHA256均为`228ae934780c27591aff68ebe882535e5a9ce012ddc503a243585bdd5c8bf053`。

独立验收（无需NPU，Python3.9以上）：

```bash
python3 scripts/audit_results.py --snapshot evidence/final --output /tmp/tq-read-audit.json
```

真实硬件回放须先按上一轮固定归档完成镜像、driver、框架源码构建、TQ算子安装、模型只读挂载及`/ws/source/kv_cache_turbo_quant`原58文件准备。以下路径布置只经静态核对，未在干净镜像从头重跑；基础镜像公共pull地址仍未核实。

```bash
# 在容器内，将本轮实验归档复制到 /ws/opt_20261012 后布置已归档依赖。
mkdir -p /ws/scripts /ws/opt_20261011/verifier /ws/opt_20261011/candidate/integration /ws/opt_20261011/verify
cp -a /ws/opt_20261012/tools/shared_bench/. /ws/scripts/
cp -a /ws/opt_20261012/tools/verifier/. /ws/opt_20261011/verifier/
cp -a /ws/opt_20261012/evidence/final/baseline/integration/. /ws/opt_20261011/candidate/integration/
cp -a /ws/opt_20261012/evidence/final/verify/. /ws/opt_20261011/verify/
cp /ws/opt_20261012/evidence/final/source_expected.json /ws/opt_20261011/source_expected.json
# model_pipeline.py重新生成同一混合候选、冻结参考、7/15/1/1验收及三模式3轮压测。
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,5 TASK_NPU_CARDS=2,5
/ws/.venv/bin/python -u /ws/opt_20261012/scripts/model_pipeline.py
```

`tools/verifier`和`tools/shared_bench`是本次实际使用工具的原样副本；模型、算子库及镜像不导出。后台宿主入口和完整服务参数见前述命令及`scripts/serve_optimized.sh`。正式profiler关闭；可选profiling脚本保留但本轮未调用。

本轮问题R01–R14保留，包含两次路径管理失误、编译/UB限制、速度回退候选、状态探针/归档/缓存问题、torch glue复用描述修正、本地审计Python版本问题、tracking-ref推送检查及关闭警告。BF16主动关闭时EngineDeadError及各模式resource_tracker警告根因未隔离，完整请求已先结束，资源释放独立核验；不称所有问题闭环。阶段归档eedf2eb的运行中叙述由本最终版本更新，不改写发布历史。

原始下载210文件逐一SHA验证，发布完整源码、逐请求数据、命令、原始日志及失败记录；Python解释器`.pyc`/`__pycache__`无需复现而不发布，原下载清单保留，发布文件另有`published-evidence-manifest.json`。未采用的K/V合并量化思路仅在本地准备，未测、未部署，不计入本轮收益。
