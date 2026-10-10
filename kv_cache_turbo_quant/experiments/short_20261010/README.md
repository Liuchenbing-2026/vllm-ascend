# Qwen3-30B-A3B · TQ开关短输入场景复测

| 验收项 | 当前记录 | 范围 |
| --- | --- | --- |
| 场景1 | 20输入 / 20输出 tokens，并发16；正在测量 | BF16与真实store4，各预热1组后正式3组 |
| 场景2 | 2000输入 / 200输出 tokens，并发60；正在测量 | BF16与真实store4，各预热1组后正式3组 |
| 环境 | 复用此前已从源码编译的runtime；没有重新编译 | .19物理卡0/1，Qwen3-30B-A3B BF16权重，TP2/eager |
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

截至2026-10-10 23:09，本轮9份正式组已经完整成功：BF16两场景各3组、TQ场景1三组；TQ场景2在预热。场景1三组总输出/总耗时：BF16=88.324464 tok/s、TQ=43.557618 tok/s；场景2目前BF16=295.173177 tok/s，TQ正式结果未取得。以上为阶段数据，失败、预热和冒烟单列，不采用部分SSE事件推算tokens。最终12份正式组结束后统一复核真实usage及payload匹配。

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
| 权重 | `/data1/models/Qwen3-30B-A3B`只读挂载`/models/Qwen3-30B-A3B`；上轮16分片SHA见固定归档，本轮只复用，来源revision未验证 |
| 任务分支 | `bench/tq-short-ab-20261010`，基础交付`7ae0083fe4faca6b51e230c9d9206ffc297d1af8`；只新增场景配置、压测组织与本轮证据，交付SHA待推送读回 |

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

## 章节五 问题台账

| 编号 / 发现时间 | 现象与影响 / 原始证据 | 原因与假设 | 处理与验证 | 当前状态 / 剩余工作 |
| --- | --- | --- | --- | --- |
| S01 / 10-10约22:57 | 初次本地查询误用工作区顶层scripts/remote.py并在非Git根目录查询origin，read-only命令失败；工具原始输出保留于会话 | 私有连接脚本实际位于artifacts/tq-128k-ab-20261010/scripts，仓库位于source/vllm-ascend | 改正确路径后SSH preflight成功，git ls-remote/fetch成功；尚未启动任何GPU工作时发现 | 路径错误已修正；早期失败没有独立文件证据，待补，不删除记录 |
| S02 / 10-10约22:58 | 宿主preflight判venv_python_exists=false；随后容器内判true，runtime-start.log | 该venv Python是指向容器系统路径的链接，宿主exists不能判容器运行时缺失；runtime_snapshot核对链接目标为/usr/local/python3.12.13/bin/python3 | 容器实际Python3.12.13、venv可执行；控制器已经开始精度验证 | 宿主检查口径纠正，不宣称venv丢失；链接目标已读回确认，容器内venv有效 |
| S03 / 10-10约23:00 | runtime中fuser命令不存在，runtime-start.log；不能用该工具确认设备FD | fuser未安装；NPU0/1的npu-smi显示无进程、0% AICore | 保留缺项，不安装/改宿主；npu-smi及容器进程归属核验继续 | fuser设备FD核验未完成；不能宣称完全排除所有潜在设备访问 |
| S04 / 10-10约22:57起 | SSH宿主profile报CANN8.5 set_env Permission denied；各连接原始日志 | 与上轮E02同一宿主profile问题，未修复 | 容器入口显式source实际CANN9.1.0；不改全局profile | 绕过不等于根因修复；实际服务及精度结果待读回 |
| S05 / 10-10约23:01 | 远端原分支已新增head_dim256/hybrid代码，若直接更新会改变比较对象 | git fetch确认c7d80df及7份integration/op差异；上轮固定8ad9ef6 | 告知用户并固定58份原源码及二进制哈希；新增压测分支与结果目录 | 比较对象固定，未测新版；不把本轮结论外推到c7d80df |

| S06 / 10-10 23:06 | BF16完整6组之后主动清理服务时段出现EngineDeadError和resource_tracker泄漏告警；logs/short_20261010T150112Z/serve_bf16_0.log | 原始请求均成功且最终usage完整；告警在本任务主动服务清理时段出现，未定位关闭协议根因 | 保留关闭错误，不当作先于请求完成的压测失败；下一store4服务正常启动 | 测试已继续，关闭告警根因未修复 |

| S07 / 10-10约23:15 | SSH wrapper把stderr宿主profile警告拼到stdout，最初host-resources-before.json不是严格JSON；host-resources-before.log保留原始字节 | 连接脚本明确合并stdout+stderr，属于记录格式问题，不是宿主性能数据缺失 | 原始输出单独保留.log，提取首个完整JSON对象为.json并验证；其他发布JSON逐一解析 | 此次记录格式已修正，未改SSH/宿主profile根因；不影响HTTP计量 |

原128K失败与未解决记录E01–E28保留在原固定归档，本轮不覆盖、不宣称已修复。其他卡3/4–7存在作业，preflight与host资源快照保留；只能确认抽样时0/1的归属，不能证明整个宿主独占或排除全部干扰。

## 章节六 交付与后续

| 项目 | 状态 |
| --- | --- |
| 正式数据与汇总 | 运行中；完成后验证真实usage、paired payload、失败数、汇总分母与原始SHA |
| 脚本与证据 | 独立任务分支，完成后正常推送并回读；不提上游PR、不强推 |
| Task23与交接 | 本次是同任务追加场景，保留128K原记录，最终链接新固定证据 |
| 资源 | 结束后停止仅本任务runtime并验证卡0/1与18377释放；保留制品 |
