# Qwen3-30B-A3B · TQ开关短输入场景复测

| 验收项 | 当前记录 | 范围 |
| --- | --- | --- |
| 场景1 | BF16 88.324464 → TQ 43.557618 tok/s，TQ/BF16 0.493155× | 20/20 tokens，并发16；两侧各3轮、各48/48请求完整成功 |
| 场景2 | BF16 295.173177 → TQ 18.857945 tok/s，TQ/BF16 0.063888× | 2000/200 tokens，并发60；两侧各3轮、各180/180请求完整成功 |
| 环境 | 复用此前已从源码编译的runtime；本轮已停止并释放0/1 | .19物理卡0/1，Qwen3-30B-A3B BF16权重，TP2/eager |
| 验证 | 58份TQ源码、全部软件版本和4份native二进制哈希与原实验一致，glue与roundtrip复验通过 | 新目录，不覆盖128K实验；整模型语义精度未测 |

## 章节一 模型与本轮范围

本轮测量同一个Qwen3-30B-A3B模型在短输入场景下的服务性能，只切换KV cache真实4-bit存储。BF16与TQ权重dtype均为BF16；量化的是KV cache。影子模式关闭。本轮不优化算子、不改框架或量化算法。

此前128K实验固定的原TQ源码为`8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b`；远端`liuchenbing-2026`已有`c7d80df37670a4473a021474e09cc471aee78b63`（head_dim256/hybrid改动及新结果）。本轮明确继续使用8ad9ef6版本，未混入新算法实现；已在对话告知。当前实现的[读路径](https://github.com/Liuchenbing-2026/vllm-ascend/blob/8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b/kv_cache_turbo_quant/code/integration/kvtq_store.py#L148)每步解压历史KV再调用BF16 FIA。本轮观察仅适用于这个store v1实现。

## 章节二 性能和精度

### 2.1 计量契约

| 项目 | 两侧一致的实际设置 |
| --- | --- |
| 输入 | 整数token IDs，seed20261010，范围1000..9999；同场景同request ID的payload SHA应一致 |
| 输出 | `ignore_eos=true`，temperature0；成功必须最终usage精确满足20/20或2000/200 |
| 请求数 | 每组等于并发16或60，同时到达；每场景每模式1组相同负载预热、3组正式测量 |
| 场景与顺序 | BF16场景1→场景2，再store4场景1→场景2；每模式先64→16单请求冒烟 |
| 时限 | 每请求3600秒，起服1200秒；保留失败，不记作0吞吐 |
| 输出吞吐 | 请求数×实际要求输出tokens / 完整组HTTP耗时；汇总采用三组总输出/三组总耗时 |
| TTFT / TPOT | 首次非空SSE文本到达 / 首末文本时间跨度除以实际输出tokens−1；TPOT不等于逐token ITL |
| 边界 | 包含排队/prefill/decode，排除payload构造、起服、精度测试和预热；这是重复请求组测试，不是持续稳态压测 |

旧128K服务max_model_len132096、max_num_seqs32；本轮两侧统一改为4096、64，足以覆盖2000+200长度和60活跃请求。不能将本轮结果与旧128K结果归因为只有输入长度变化。保留YaRN factor4/original40960、prefill4096、24GiB/卡KV预算、eager、prefix关闭、async scheduling关闭及OMP1。预热输出保留但不纳入正式平均。

### 2.2 本轮结果

| 场景 | BF16输出 tok/s | TQ输出 tok/s | TQ/BF16 | TQ吞吐下降 |
| --- | --- | --- | --- | --- |
| 20/20，并发16 | 88.324464 | 43.557618 | 0.493155× | 50.68% |
| 2000/200，并发60 | 295.173177 | 18.857945 | 0.063888× | 93.61% |

| 场景 | BF16 TTFT均值(s) | TQ TTFT均值(s) | BF16 TPOT均值(ms) | TQ TPOT均值(ms) |
| --- | --- | --- | --- | --- |
| 场景1 | 0.493656 | 0.701697 | 152.045622 | 338.667549 |
| 场景2 | 4.026710 | 22.596940 | 171.372860 | 2989.183510 |

| 场景/模式 | 三轮整组耗时(s) | 三轮输出吞吐(tok/s) | 成功/失败请求 |
| --- | --- | --- | --- |
| 场景1/bf16 | 3.693/3.539/3.637 | 86.642/90.425/87.989 | 48/0 |
| 场景1/store4 | 7.167/7.581/7.291 | 44.647/42.208/43.889 | 48/0 |
| 场景2/bf16 | 41.028/41.236/39.698 | 292.484/291.007/302.279 | 180/0 |
| 场景2/store4 | 636.604/636.719/635.687 | 18.850/18.847/18.877 | 180/0 |

正式12组、456个请求全部精确完成要求的输入/输出usage，无失败或重试。另有4个同负载预热组及2个冒烟组，均成功但不纳入正式指标。两侧相同场景的全部重复payload SHA一致。汇总源路径/原始SHA及计量边界见[summary.json](summary.json)，原始结果在results目录；同目录的.progress.json是过程快照，不参与最终指标。

吞吐是三组总输出/三组总HTTP耗时，不是三个吞吐值简单平均，也不是TPOT倒数；TTFT/TPOT汇总全部正式成功请求。请求组之间的间隔不在分母中。本轮3次相同合成输入重复，不能推导持续稳态吞吐或真实业务语义质量。

启动容量仍为BF16 524288 → TQ 2033536 tokens，3.87866211倍；两侧两rank初始free60.57/60.58GiB、KV预算24GiB一致。该配置下60条2200-token请求不受BF16总KV容量上限约束；容量倍率不当作吞吐倍率。

日志确认两侧都有60个Running请求且Waiting=0的窗口；TQ真实store4、shadow0、两rank量化调用与Torch扩展加载路径已核验。没有profiling耗时拆解，不能量化解压、通信和调度各自比例。

阶段归档cf01a8c16e949b91b5ebab866fe8ff331fd426f4记录9组正式完成，其snapshot-manifest-partial.json只对应[该固定阶段](https://github.com/Liuchenbing-2026/vllm-ascend/tree/cf01a8c16e949b91b5ebab866fe8ff331fd426f4/kv_cache_turbo_quant/experiments/short_20261010)的字节；最终闭合日志与结果以[snapshot-manifest-final.json](snapshot-manifest-final.json)为准，未删除阶段失败或历史证据。

### 2.3 精度

原5组bit-exact glue与4-bit roundtrip/Q旋转测试本轮在复用二进制上重新执行，均通过，完整日志已读回。没有整模型任务准确率、输出质量或语义一致性评测；合成token输入只测性能。

## 章节三 版本与环境复现

| 组件 | 固定版本与来源 |
| --- | --- |
| 基础镜像 | `vllm-ascend:dspark-a2-028`；本地image ID `sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`，RepoDigests为空，未取得公共拉取地址或registry digest，未导出镜像 |
| runtime | `tq-128k-ab-20261010-runtime`，ID `0354eaafc53831afa0a11835e862a550873d9c1db0c8d54e178747c90b45663f`；本轮重新启动已保留容器，不是新建容器 |
| vLLM | 0.28.0 / `2cf0a6915ce544dc493a0990f2ea38d81601128a`；上轮empty目标editable源构建，本轮未重编 |
| vLLM-Ascend | 0.28.0rc1 / `96df623103f921df1a4488170d106f36689acb59`；复用上轮新编native及C++扩展，未改源码 |
| Catlass | `41bf90da655bba3c66d0acd7e00abe33960ecfd6`，复用上轮编译产物 |
| TQ | 固定8ad9ef6原58份文件，`source_expected.json`逐文件校验；插件路径和4份native哈希再次核验 |
| 软件 | Python3.12.13、CANN9.1.0、torch2.10.0+cpu、torch-npu2.10.0.post4、transformers5.14.1、numpy1.26.4、ml-dtypes0.5.3；全部包版本见runtime_verified.json |
| 附加API依赖 | fastapi0.136.0、starlette1.0.1、fastokens0.2.0、pydantic2.14.0、pydantic-core2.50.0、aiohttp3.14.3；uv0.12.24、setuptools80.10.2、setuptools-scm10.3.4、cmake4.4.2、pybind113.1.0、triton-ascend3.2.2 |
| 权重 | `/data1/models/Qwen3-30B-A3B`只读挂载`/models/Qwen3-30B-A3B`；上轮16分片SHA见固定归档，本轮核对16分片文件名/尺寸/mtime与上轮指纹一致，没有重新计算完整权重SHA，来源revision未验证 |
| 任务分支 | `bench/tq-short-ab-20261010`，基础交付`7ae0083fe4faca6b51e230c9d9206ffc297d1af8`；只新增场景配置、压测组织与本轮证据，阶段cf01a8c16e949b91b5ebab866fe8ff331fd426f4已推送读回；最终固定交付SHA见Model_test任务23追加场景入口 |

从零准备基础镜像、源码、venv、OPP和Torch胶水所需的实际构建命令与历史失败恢复见[上轮固定复现README](https://github.com/Liuchenbing-2026/vllm-ascend/blob/7ae0083fe4faca6b51e230c9d9206ffc297d1af8/kv_cache_turbo_quant/experiments/128k_20261010/README.md)。本轮验证已存在环境的恢复与两场景压测，不宣称干净镜像一次性从零复跑。该栈pip check约束冲突仍存在，未宣称全绿。

容器完整创建/挂载参数见[固定创建脚本](https://github.com/Liuchenbing-2026/vllm-ascend/blob/7ae0083fe4faca6b51e230c9d9206ffc297d1af8/kv_cache_turbo_quant/experiments/128k_20261010/scripts/create_container.py)。`/data1/tq-128k-ab-20261010`挂载`/ws`，driver/add-ons只读，runtime ascend/privileged，ASCEND_VISIBLE_DEVICES与ASCEND_RT_VISIBLE_DEVICES均0,1；privileged不能作为强设备cgroup隔离证明。本轮独立根目录`/ws/short_ab_20261010`，旧`/ws/status.json`和128K结果不改。

## 章节四 实际启动和评测命令

宿主先执行`python3 scripts/preflight.py`与`python3 scripts/start_runtime.py`，只启动预先核对ID的本任务容器，并检查NPU0/1空闲；把本目录必要脚本及source_expected.json复制到`/ws/short_ab_20261010`后运行：

```bash
# 宿主：实际后台启动由launch_short.py执行，完整命令保留在launch.log。
python3 scripts/launch_short.py

# 等价容器内入口（不与已运行controller同时执行）：
source /usr/local/Ascend/cann-9.1.0/set_env.sh
/ws/.venv/bin/python /ws/short_ab_20261010/scripts/run_short.py
```

实际完整服务命令在[scripts/serve_short.sh](scripts/serve_short.sh)，两侧仅参数0/1改变STORE。复用[上轮固定client](https://github.com/Liuchenbing-2026/vllm-ascend/blob/7ae0083fe4faca6b51e230c9d9206ffc297d1af8/kv_cache_turbo_quant/experiments/128k_20261010/scripts/bench_client.py)和[进程管理runner](https://github.com/Liuchenbing-2026/vllm-ascend/blob/7ae0083fe4faca6b51e230c9d9206ffc297d1af8/kv_cache_turbo_quant/experiments/128k_20261010/scripts/run_matrix.py)，本轮没有放宽实际usage验收。

```bash
# 容器内：仅在独立服务已就绪且没有其他压测时单独运行。
bash /ws/short_ab_20261010/scripts/serve_short.sh 0  # 另一轮用1启用store4
/ws/.venv/bin/python /ws/scripts/bench_client.py --mode bf16 --concurrency 16 \
 --input-len 20 --output-len 20 --timeout 3600 --output /ws/short_ab_20261010/s1_bf16.json
/ws/.venv/bin/python /ws/scripts/bench_client.py --mode bf16 --concurrency 60 \
 --input-len 2000 --output-len 200 --timeout 3600 --output /ws/short_ab_20261010/s2_bf16.json

# 独立精度入口；controller实际在性能测量前执行。
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHONPATH=/ws/source/vllm:/ws/source/vllm-ascend:/root/kvtq_integration:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0,1 TORCH_EXTENSIONS_DIR=/ws/torch-extensions VLLM_ASCEND_KVTQ_BITS=4
/ws/.venv/bin/python /root/kvtq_integration/torch_ext/test_glue.py
/ws/.venv/bin/python /root/kvtq_integration/test_store_roundtrip.py
```

独立命令为复现入口；本轮真正执行的各组命令保留在logs对应command.json，正式重复r1/r2/r3、预热warmup与smoke文件分开。没有远端CI验证。

本轮最终汇总的实际命令（在本归档目录执行）：

```bash
python3 scripts/summarize_short.py --root . --run-id short_20261010T150112Z --output summary.json
```

## 章节五 问题台账

| 编号 / 发现时间 | 现象与影响 / 原始证据 | 原因与假设 | 处理与验证 | 当前状态 / 剩余工作 |
| --- | --- | --- | --- | --- |
| S01 / 10-10约22:57 | 初次本地查询误用工作区顶层scripts/remote.py并在非Git根目录查询origin，read-only命令失败；工具原始输出保留于会话 | 私有连接脚本实际位于artifacts/tq-128k-ab-20261010/scripts，仓库位于source/vllm-ascend | 改正确路径后SSH preflight成功，git ls-remote/fetch成功；尚未启动任何GPU工作时发现 | 路径错误已修正；早期失败没有独立文件证据，待补，不删除记录 |
| S02 / 10-10约22:58 | 宿主preflight判venv_python_exists=false；随后容器内判true，runtime-start.log | 该venv Python是指向容器系统路径的链接，宿主exists不能判容器运行时缺失；runtime_snapshot核对链接目标为/usr/local/python3.12.13/bin/python3 | 容器实际Python3.12.13、venv可执行；控制器已经开始精度验证 | 宿主检查口径纠正，不宣称venv丢失；链接目标已读回确认，容器内venv有效 |
| S03 / 10-10约23:00 | runtime中fuser命令不存在，runtime-start.log；不能用该工具确认设备FD | fuser未安装；NPU0/1的npu-smi显示无进程、0% AICore | 保留缺项，不安装/改宿主；npu-smi及容器进程归属核验继续 | fuser设备FD核验未完成；不能宣称完全排除所有潜在设备访问 |
| S04 / 10-10约22:57起 | SSH宿主profile报CANN8.5 set_env Permission denied；各连接原始日志 | 与上轮E02同一宿主profile问题，未修复 | 容器入口显式source实际CANN9.1.0；不改全局profile | 绕过不等于根因修复；精度和两侧服务均成功，根因仍未修复 |
| S05 / 10-10约23:01 | 远端原分支已新增head_dim256/hybrid代码，若直接更新会改变比较对象 | git fetch确认c7d80df及7份integration/op差异；上轮固定8ad9ef6 | 告知用户并固定58份原源码及二进制哈希；新增压测分支与结果目录 | 比较对象固定，未测新版；不把本轮结论外推到c7d80df |
| S06 / 10-10 23:06，23:51复现 | BF16完整6组之后主动清理服务时段出现EngineDeadError和resource_tracker泄漏告警；TQ完整6组后关闭也复现；logs/short_20261010T150112Z/serve_bf16_0.log及serve_store4_0.log | 所有正式请求在关闭前均已成功且最终usage完整；告警在本任务主动服务清理时段出现，未定位关闭协议根因 | 两侧关闭错误均保留，不当作先于请求完成的压测失败；controller最终exit0，12份正式结果完整 | 测量已完成，关闭告警根因未修复 |
| S07 / 10-10约23:15 | SSH wrapper把stderr宿主profile警告拼到stdout，最初host-resources-before.json不是严格JSON；host-resources-before.log保留原始字节 | 连接脚本明确合并stdout+stderr，属于记录格式问题，不是宿主性能数据缺失 | 原始输出单独保留.log，提取首个完整JSON对象为.json并验证；其他发布JSON逐一解析 | 此次记录格式已修正，未改SSH/宿主profile根因；不影响HTTP计量 |
| S08 / 10-10 23:19 | Model_test更新期间另一session增加task3知识记录，rebase在共享index.json冲突；modeltest-index-conflict.json | 两侧修改同一JSON索引；远端TQ记录未变，远端新增/改task3记录，本地仅更新TQ | 保留完整远端索引和所有其他记录，仅替换TQ记录；22条校验通过，正常推送a1663d50c5c3fbb85dd77f5adac077bf5b393440，4文件读回一致并刷新缓存 | 本次冲突已安全解决；并发写入仍需每次fetch/合并，不是后台锁或持续同步 |
| S09 / 10-10 23:09起 | 短输入仍退化：20/20并发16 TQ 43.557618 vs BF16 88.324464 tok/s；2000/200并发60的TQ预热约11分钟 | 场景1实际完整工作量及payload配对已验证；全历史解压/调度耗时比例尚未profiling，不能以事件数补全正式结果 | 不改算法、不缩短输出，保留3轮原始JSON；场景2三轮正式测量均完整成功，最终指标见第二章 | 性能问题未优化；两场景最终汇总已取得，性能问题未通过优化修复 |
| S10 / 10-10约23:35 | 新增S06–S09时以表后段落为插入点，空行使新增行与表头分离；阶段cf01a8c的README保留历史 | 文档生成插入点不当，静态GFM结构检查确认 | 移除同一问题表行之间空行，检查S01–S11连续；不影响原始实验，未执行浏览器渲染验收 | 静态结构已修正，历史提交不重写 |
| S11 / 10-10约23:35 | docker-top-bf16.log实际采集时已经切换到TQ启动，不能作为BF16的宿主PID证据；原文件及runtime-bf16-snapshot.log保留 | 采集请求发出时模式已改变，标签仍沿用原计划；PID4129475属于TQ API | 对照状态时间/两个snapshot与TQ docker top确认；BF16只引用自身容器/procmaps及npu-smi快照，直接宿主PID匹配证据缺项 | 证据命名口径明确；不重测以补造历史PID证据，不把该文件用于BF16归属验收 |
| S12 / 10-10约23:53 | 最终归档辅助脚本已汇总12组后断言结果目录应有18份JSON失败，实际为36份；archive-result-count-failure.json保留异常与文件清单 | glob包含18份测量结果及18份实时progress快照，辅助脚本计数口径错误；实际压测结果不是多跑或缺失 | 显式排除.progress.json，仅统计12正式+4预热+2冒烟，重新核验6个排除组usage/payload和88份快照SHA | 归档辅助计数已修正；未重跑或改测量原始字节，失败记录保留 |
| S13 / 10-10约23:57 | Model_test提交前核对5个暂存路径时集合断言失败；modeltest-path-encoding-failure.json保留转义输出与NUL解码路径 | Git默认core.quotePath把中文路径渲染成八进制转义文本，辅助脚本按文本与Unicode文件名比较；实际5个文件均属本任务 | 检查原始NUL路径确认范围，仅撤销本次5文件暂存；Git调用显式core.quotePath=false，重新进行范围检查、正常提交与推送回读 | 归档发布检查口径已修正；首次断言发生在commit/push之前，无远端覆盖，最终发布结果见Task23固定交付链接 |

原128K失败与未解决记录E01–E28保留在原固定归档，本轮不覆盖、不宣称已修复。其他卡3/4–7存在作业，preflight与host资源快照保留；只能确认抽样时0/1的归属，不能证明整个宿主独占或排除全部干扰。

## 章节六 交付与后续

| 项目 | 状态 |
| --- | --- |
| 正式数据与汇总 | 12正式组/456请求完整成功；usage、payload配对、汇总分母及原始SHA已核验 |
| 脚本与证据 | 独立任务分支正常推送并回读；没有上游PR或强推，固定SHA见Task23入口 |
| Task23与交接 | 同任务追加场景，保留128K原记录；Task23链接本轮固定证据 |
| 资源 | 2026-10-10T15:54:29.071859+00:00仅本任务runtime停止、NPU0/1无进程、18377可绑定；原build仍exited，制品保留 |

最后核对：2026-10-10T15:54:49.998654+00:00。资源释放原始证据：[task-resource-release.log](task-resource-release.log)。本轮性能与归档已完成；缺fuser、宿主profile、关闭协议和TQ性能根因等未全部修复，整模型精度未测。
