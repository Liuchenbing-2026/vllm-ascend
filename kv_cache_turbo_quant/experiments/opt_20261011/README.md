# Qwen3-30B-A3B TP2：TQ 读取优化、整网实测与后台脚本

新容器双卡同宿主对照已完成：融合读取保留原4-bit KV格式和容量，20/20/c16整网吞吐提高1.552倍，2000/200/c60提高6.557倍；优化后仍低于BF16。后台脚本正常完成，exit0，684个正式请求全部成功，18个正式组均复核相同payload及完整usage。

| 场景 | BF16 tok/s | 原TQ tok/s | 优化TQ tok/s | 优化/原TQ | 优化/BF16 |
|---|---:|---:|---:|---:|---:|
| 20输入/20输出/c16 | 75.611 | 36.684 | 56.931 | 1.552× | 0.753× |
| 2000输入/200输出/c60 | 245.250 | 18.312 | 120.063 | 6.557× | 0.490× |

| 场景/模式 | TTFT s | TPOT ms | 正式成功请求 |
|---|---:|---:|---:|
| 场景1/bf16 | 0.4120 | 200.109 | 48 |
| 场景1/optimized | 0.5192 | 267.315 | 48 |
| 场景1/original | 0.6133 | 425.122 | 48 |
| 场景2/bf16 | 4.1449 | 209.637 | 180 |
| 场景2/optimized | 6.7374 | 447.390 | 180 |
| 场景2/original | 23.3167 | 3075.676 | 180 |

每场景每模式同负载预热1组、正式3组，逐轮吞吐稳定，正式统计排除启动、smoke、预热、精度及profiling。严格读取NPU7/7、CPU元数据4/4、真实FIA与混合历史门禁15/15、2/3-bit回退各1/1、原glue/roundtrip通过；优化与原TQ四组greedy样本共440个token IDs完全一致。四样本不能替代语义质量benchmark，优化版128K性能尚未测。

同24GiB KV预算：BF16启动日志524288 tokens，原TQ和优化TQ均2033536 tokens，容量3.878662109375倍。两rank映射及实际版本/启动命令见正式日志和run_parameters.json。

完整逐请求、命令、服务/精度日志、独立离线解析汇总见 `evidence/final-snapshot/`。流水线完成时间2026-10-11T02:23:10Z，整网run_id `model_20261011T004955Z`；当前PID已正常退出，不能将历史PID称为仍运行。physical2/5及18377已实查释放，仅自身容器剩sleep，其他作业未动；构建环境和原始profiling继续保留。

八组新profiling视图均通过trace/kernel_details/op_statistic/api_statistic完整性检查。原TQ decode rank1 GatherV3约1.906s/194次、LeftShift0.407s/192次；优化窗口每rank融合读取768次、均值3.39–3.41ms，占kernel sum87.50%/88.48%。窗口完成迭代数不同，不能直接比较累计sum作加速比；正式6.557倍来自独立完整cohort。该窗口是2000/32/c60诊断负载，不自动等同稳态batch60。下一瓶颈仍为历史KV读取/解压与BF16工作区，直接压缩KV attention或分块融合为后续假设，未实现。

## 固定版本与环境

任务分支 `perf/tq-store-read-20261011`，base `d9e527a35100304fe6365ee2d090a08dd75c44d6`。原58文件固定 `8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b`，`source_expected.json`逐文件核验；不修改原量化格式、centroids、旋转、golden或精度容差。

vLLM `2cf0a6915ce544dc493a0990f2ea38d81601128a`，vLLM-Ascend `96df623103f921df1a4488170d106f36689acb59`，Catlass `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。运行源码为导出包，现场无`.git`；不把归档固定版本写成现场git查询。基础镜像完整tag `vllm-ascend:dspark-a2-028`，image ID `sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`；RepoDigests为空，公共获取地址未核实，不导出镜像。CANN9.1.0、Python3.12.13、torch2.10.0+cpu、torch_npu2.10.0.post4、triton-ascend3.2.2（import显示3.2.0）。新宿主driver26.0.rc1，910B4-1；不将旧宿主driver25.5.1的整网吞吐作为本轮因果对照。

新容器 `tq-opt-20261011-model`，ID `55238dba5d6c2c84f296446b4eeacfe4fd952a156e839abfe6486771a8f25186`，有独立PID namespace。宿主 `/data02/tq-opt-20261011`挂载`/ws`，宿主`/root/tq-opt-20261011`挂载`/ws/opt_20261011`，模型只读挂载。原定physical2/3在起跑前3被其他任务占用，改physical2/5；后台每20秒核验进程归属，冲突只终止本任务进程。privileged容器可访问宿主设备；只声明运行环境指定卡，不声明强设备隔离。16份模型权重及config/tokenizer/index在新宿主全量SHA复核通过，模型源HF revision未核实。

实际源码构建及额外依赖完整命令见 `scripts/bootstrap_third.sh`；TQ原算子构建入口是容器 `/ws/build_tq_op.sh`，源码未改，实际构建脚本已保存在 `evidence/background-snapshot/build/build_tq_op.sh`；完整日志待最终回收。`bootstrap.exit`、`tq-op-build.exit`、`model-hash.exit`实查均0。实际运行环境完整清单由后台`collect_environment.py`采集，不把预期依赖当作运行证明。

## 优化及验证边界

融合4-bit物理分页读取、nibble解包、查表与norm缩放，直接写BF16临时张量，保留原FIA和历史attention逻辑。metadata保存最多decode/prefill两个长度计划，跨K/V与层复用，长度或设备变化失效；每次仍读取当前物理block table。FIA采用调度器已知CPU长度列表，避免每层D2H/cumsum同步。并未融合dequant与attention，仍生成BF16临时工作区；query旋转及原quantizer中QJL/gamma未优化。

V1七组真实NPU严格int16 bit比较全部通过；空输入没有kernel，micro profiler三次无耗时结果作为不适用保留，性能仅六个非空组。raw保留复测的全部profile与设备日志已SHA回收。V2连续字节读取候选修正int32 Gather不支持后七组通过；短case变慢0.0135→0.0160ms、长case3.8628→3.5293ms，本轮选择V1以收敛整网验证，V2及初次失败源码/日志保留。

初始原TQ prefill窗口GatherV3约38%、LeftShift约9.8%，每rank1680次aclrtSynchronizeStream；这些是窗口成本，不能外推整网占比。旧TQ采样两次被外部Docker停止，仅prefill有效，decode未完成，不伪造吞吐。完整失败与未解问题O01–O36保存在 `issues.json`。

## 后台入口和完整启动命令

主脚本 `scripts/background_host.py` 在宿主运行，断开SSH继续；容器内 `scripts/background_pipeline.py` 串行执行门禁与完整矩阵，失败保留日志并停止本任务进程，不重置设备、不停止其他容器。以下是实际宿主入口，重复执行前先确认已有PID/锁，不启动两个副本：

```bash
nohup python3 -I /root/tq-opt-20261011/scripts/background_host.py \
  > /root/tq-opt-20261011/logs/background-host.log 2>&1 < /dev/null &
```

后台实际容器执行命令：

```bash
docker exec -e TASK_NPU_CARDS=2,5 tq-opt-20261011-model bash -lc \
 'source /usr/local/Ascend/cann-9.1.0/set_env.sh; exec /ws/.venv/bin/python -u /ws/opt_20261011/scripts/background_pipeline.py'
```

完整服务命令在 `scripts/serve_optimized.sh`：TP2 BF16 eager、prefix caching/async scheduling关闭、max_model_len/batched_tokens4096、maxseq64、每卡KV预算25769803776bytes、gpu-util0.92、YaRN4/original40960、127.0.0.1:18377。formal不打开profiler；profiling另起同参数服务。

独立集成精度命令由门禁实际依次执行（4-bit十五项，2/3-bit各一项），每项必须通过才进入压测：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,5 VLLM_ASCEND_KVTQ_STORE=0 VLLM_ASCEND_KVTQ=0
export PYTHONPATH=/ws/opt_20261011/candidate/integration:/ws/source/vllm:/ws/source/vllm-ascend
export TRITON_CACHE_DIR=/ws/opt_20261011/model-triton-cache TORCH_EXTENSIONS_DIR=/ws/torch-extensions
VLLM_ASCEND_KVTQ_BITS=4 /ws/.venv/bin/python /ws/opt_20261011/scripts/verify_integration.py
VLLM_ASCEND_KVTQ_BITS=2 /ws/.venv/bin/python /ws/opt_20261011/scripts/verify_integration.py
VLLM_ASCEND_KVTQ_BITS=3 /ws/.venv/bin/python /ws/opt_20261011/scripts/verify_integration.py
```

压测实际控制器命令（由pipeline执行，勿与活动pipeline并行）：

```bash
TASK_NPU_CARDS=2,5 /ws/.venv/bin/python -u /ws/opt_20261011/scripts/run_model_ab.py
```

两场景20/20/c16、2000/200/c60；每模式每场景同负载预热一组，正式三组。种子20261010、输入token IDs1000..9999、temperature0、ignore_eos、精确固定输出。逐请求usage全部成功才报告输出吞吐，否则失败组吞吐为空。总体输出吞吐=三组总输出/三组HTTP总时长；TTFT首非空SSE文本，TPOT=(末文本−首文本)/(output−1)。smoke、四组单请求贪心输出对照、预热和profiling均排除。四组greedy样本不是语义质量benchmark。

profiling诊断2000/32/c60，与正式负载分开；decode在所有请求获得首文本后采样，但短输出可能已有请求完成，不能直接称稳态batch60。实际start/stop HTTP时刻记录；请求sleep3秒不等于精确三秒采样。独立非daemon离线解析，验证trace/kernel_details/op_statistic/api_statistic四类视图；多流报告kernel sum/interval union，不把嵌套CPU inclusive之和当wall time。

## 状态、结果与恢复

宿主 `/root/tq-opt-20261011/background-host.pid`、`background-pipeline.pid`、`model-controller.pid`；`background-status.json`为流水线阶段，`status.json`为服务/场景实时阶段。`logs/background-host.log`、`logs/background-launch.log`、`logs/background/*`、`logs/model_*/`保留实际命令与日志。成功后 `results/model_*/summary.json`、`background-results.md`、`profile-summary.json`自动产生，原始逐请求与profiling继续保留。后台只自动生成远端文件，脚本自动生成结果；最终GitHub发布由收尾阶段正常推送并回读，脚本本身不持有GitHub私钥。

恢复第一步：SSH进入宿主读取上述两份status、PID、exit及最新日志；`background.exit=0`才算流水线完成。若失败，先归档日志定位根因；重新运行必须保留旧run_id结果与失败记录，不能删除失败或放宽精度。源码分支的提交页作为交付SHA，不在文件中填写自指提交。CI及上游评审未执行，没有创建PR。

历史启动快照：宿主supervisor PID `1913143`，容器pipeline PID `26681`，开始时间2026-10-11T00:42:18Z；阶段集成门禁。首次attempt因为测试初始化导入顺序失败，14项严格对照已产生零bit差，未宣称十五项通过；首次源码/日志/exit保存在`evidence/background-snapshot/live/attempts/`及first_import_failure.py，重试与根因修复分开。以上PID/阶段是时间快照，接手时必须实时核查。

第二次attempt按正常顺序注册ops后，mixed-history因未建立forward context中断，仍保留14项已零bit比较和完整报错。当前harness使用原生ForwardContext/override_forward_context建立正常执行上下文，再运行同一真实FIA wrapper，不mock、不删用例、不改判据；失败快照与当前阶段见evidence/background-snapshot。

历史门禁快照：2026-10-11T00:49:55Z，4-bit真实读/FIA/mixed-history严格bit门禁15/15、2-bit/3-bit回退各1/1通过；host PID1922823、pipeline PID26881。整网run_id `model_20261011T004955Z`，controller已启动并执行glue前置检查，随后自动formal与profiling。失败attempt仍保留；整网性能及四组greedy输出尚未完成。

原始profiling及完整trace/kernel CSV在宿主`/root/tq-opt-20261011/profiles112/model_20261011T004955Z`保留；当前Git归档上传完整逐请求、服务/精度/命令日志及八视图解析汇总，不宣称全量原始trace已上传。主动关闭后的EngineDeadError/resource_tracker告警仍未根因修复，不影响已完整结束的正式请求；原始日志CR/尾空格保留，源码、脚本、Markdown的diff检查独立通过。
