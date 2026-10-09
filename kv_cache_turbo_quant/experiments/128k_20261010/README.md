# TQ 开关 128K 输入 / 1K 输出性能复测

| 项目 | 当前状态 |
| --- | --- |
| 测试范围 | Qwen3-30B-A3B，TP=2，BF16 eager 与 TQ store 4-bit；并发 1/2/4/8/16/32 |
| 工作量 | 输入 131072 tokens、输出 1024 tokens；最终环境128K长冒烟已通过，正式usage持续核验 |
| 设备 | .19：8×Ascend 910B4-1；物理卡0、1。2026-10-10 01:19:37+08:00空闲，正式启动前再次核验；运行中快照仅本任务两个worker占卡 |
| 代码来源 | Liuchenbing-2026/vllm-ascend 的 liuchenbing-2026 分支，目录 kv_cache_turbo_quant；固定 SHA 8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b |
| 实测结果 | BF16 C1/C2/C4/C8已完整完成；TQ C1正式运行中，其他档位继续排程。阶段汇总见下表；不以历史容量收益或影子模式吞吐代替本轮结果 |

## 固定测试条件

新runtime容器 `tq-128k-ab-20261010-runtime`，.19物理卡0、1，2×Ascend910B4-1（64GiB/卡）；Qwen3-30B-A3B BF16权重、TP=2、eager。vLLM和vLLM-Ascend均从固定源码在新任务目录完成构建，实际模块与native库路径已核验。TQ为固定提交的真实store 4-bit，shadow关闭；原58文件和量化/读写算法未改。

每格同时发起N个请求，N=1/2/4/8/16/32；每请求输入131072个固定种子的整数token IDs，输出1024 tokens，temperature=0、ignore_eos=true。两侧相同payload SHA256；成功记录必须实际usage为131072/1024。单格仅一组，不作为稳态吞吐或可信P99。输入为合成token IDs，未测长文本任务准确率。

两侧统一KV预算25769803776 bytes（24.00GiB/卡）、chunked prefill=4096、max_num_seqs=32，关闭prefix cache与async scheduling。原模型上限40960；两侧统一YaRN factor4/original40960，服务总长度132096。这是性能实验配置，不据此认定128K语义质量通过。

## 容量实测

| 服务 | 初始free（两卡） | KV预算/卡 | 日志GPU KV cache size | 132096总长度最大驻留并发 |
| --- | --- | --- | --- | --- |
| BF16 | 60.57 / 60.58 GiB | 24.00 GiB | 524288 tokens | 3.97x |
| TQ store 4-bit | 60.57 / 60.58 GiB | 24.00 GiB | 2033536 tokens | 15.39x |

容量比为3.878662109375。两侧启动日志的manual KV预算和初始可用显存一致；不是按不同gpu_memory_utilization所得容量。容量比与输出吞吐比单独计算。

## 性能结果与计量

当前阶段数据如下（TQ C1仍在运行，未测项尚未完成）：

| 并发 | BF16 输出 tok/s | TQ store 4-bit 输出 tok/s | TQ/BF16 |
| --- | --- | --- | --- |
| 1 | 5.28 | 未测 | — |
| 2 | 8.23 | 未测 | — |
| 4 | 8.79 | 未测 | — |
| 8 | 10.50 | 未测 | — |
| 16 | 未测 | 未测 | — |
| 32 | 未测 | 未测 | — |

| 并发 | BF16 TTFT均值(s) | TQ TTFT均值(s) | BF16 TPOT均值(ms) | TQ TPOT均值(ms) |
| --- | --- | --- | --- | --- |
| 1 | 39.51 | 未测 | 151.05 | 未测 |
| 2 | 61.23 | 未测 | 180.73 | 未测 |
| 4 | 137.48 | 未测 | 198.71 | 未测 |
| 8 | 310.58 | 未测 | 211.81 | 未测 |
| 16 | 未测 | 未测 | 未测 | 未测 |
| 32 | 未测 | 未测 | 未测 | 未测 |

每格一组同时到达的请求，请求数等于并发；未完成组不计算吞吐或收益比。计时从 HTTP 发送到最后响应结束，排除输入构造，包含排队、prefill、decode。

原始JSON保留实际请求usage、完成状态、时延、payload哈希；[selection.json](selection.json)显式选择各格来源，snapshot-manifest JSON记录原路径与文件SHA256，不自动用较新重试覆盖旧失败。

输出吞吐=N×1024/整组耗时，从HTTP发送到最后响应结束，包含排队、prefill、decode，排除输入构造和服务启动。TTFT为首次非空SSE文本到达；TPOT=(最后文本−首次文本)/(实际输出tokens−1)，不是逐token ITL。未完成组不计算完整工作吞吐或收益比；SSE文本事件数不当作已生成token数。

已完成BF16 C1/C2/C4/C8的HTTP时限为1800s，均在时限前完整结束。后续TQ及BF16高并发时限7200s；实际命令与每格timeout保留。计划切换中断的旧BF16 C16不计入正式完成结果，原记录保留。

## 已验证与限制

原5组胶水bit-exact案例全部通过（idx/qjl/norm/gamma）；4-bit roundtrip脚本通过。其报告的重建相对L2误差为MSE-only均值0.0957、最大0.1805，带QJL参考均值0.0878；这些是诊断指标，不是MSE数值，也不是本脚本单独的误差验收门槛。实际store v1不保存或使用QJL残差，不能用带QJL参考代表实际store质量。Q侧旋转等价最大相对差0.00166<0.01。未做整模型任务准确率评估。

源码确认每层每解码步解压完整历史KV，再调用BF16注意力，见固定源码[读取路径](https://github.com/Liuchenbing-2026/vllm-ascend/blob/8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b/kv_cache_turbo_quant/code/integration/kvtq_store.py#L148)。当前长上下文解码慢与该路径一致，但未采集正式运行的kernel耗时分解，不能量化各阶段开销占比。

基础镜像 `vllm-ascend:dspark-a2-028` 为宿主既有本地镜像，完整image ID与实际版本见下文及[environment.json](environment.json)。RepoDigests为空，无可提供的registry digest或拉取地址；未导出镜像。模型配置/tokenizer/index已记录哈希，16个权重文件仅记录尺寸、未验全量weight哈希及来源revision。依赖元数据仍有已披露冲突，不能说pip check全绿。

构建、实际启动、压测、独立精度命令及所有问题台账见下文。整合复现脚本经过静态核对；本轮实际编译、依赖修正、精度与服务均分步执行，不声明在干净镜像从零一次性复跑通过。

## 问题台账

失败记录保留；恢复不等于根因修复。

| 编号 / 发现时间 | 现象、影响与证据 | 已确认原因 / 假设 | 处理、验证与状态 |
| --- | --- | --- | --- |
| E01 / 2026-10-10 01:19:37+08:00 | .18 普通账号执行 npu-smi：DrvMngGetConsoleLogLevel failed ret=4，dcmi initialize ret=-8005；Docker socket permission denied。无法由该账号完成占卡检查。 | Docker 权限不足已确认；npu-smi 具体失败原因未确定，不能把错误解释为无卡或空闲。 | 改查 .19，成功取得八卡无进程快照。任务绕过，.18 根因未修复；原始完整登录日志待补。 |
| E02 / 2026-10-10 01:19:37+08:00 | .19 登录 profile 尝试 source cann-8.5.0/set_env.sh，Permission denied。 | 该宿主环境脚本不可读；新容器 CANN 版本待独立核验。 | 新容器显式 source 实际 CANN 环境，宿主 profile 不修改。未修复。 |
| E03 / 2026-10-10 01:23+08:00 | .19 没有 rg；sudo -n fuser 提示需要密码，未取得设备句柄清单。 | 缺少 rg、sudo 需要认证已确认。 | 远端搜索改用 Python；Docker/NPU 只读结果已取得，设备句柄检查待补。不认定零利用率等于空闲。 |
| E04 / 2026-10-10，读取固定源码时 | docs/reproduce.md §7 记载 store 读路径约 10s/step，历史压测超时；真实 store 性能不可由 shadow 结果替代。 | 固定源码 kvtq_store.py 在每层每解码步解压完整历史 KV，成本随上下文增加；本轮耗时尚未测量。 | 按用户要求测试真实 store 4-bit，先做小样本验证，再跑要求的负载；保留超时、失败、OOM，不伪造吞吐。待验证。 |
| E05 / 2026-10-10，读取固定源码时 | install_plugin.sh 的 cp 源路径是旧扁平目录，但提交文件位于 code/integration；直接运行会找不到文件。 | 归档后的目录结构与旧脚本默认路径不一致。 | 部署时将固定源码复制到插件约定的 /root/kvtq_integration 扁平目录；不改算子或插件逻辑。待验证。 |
| E06 / 2026-10-10，容器创建前 | 普通账号不能在 /data1 根目录创建任务目录；容器尚未创建。原始证据 logs/create-container.log。 | /data1 目录写权限不足。 | 使用 Docker 自动创建精确的任务挂载目录，只调整该新目录归属，未更改 /data1 或其他目录。待验证。 |
| E07 / 2026-10-10 01:30:29+08:00 | 新容器首次 NPU tensor 初始化 aclInit=507899，Get device cnt failed 0x7020010、drv devId invalid；不能启动模型。 | 初版仅挂载 driver/lib64 两目录，较已验证容器缺完整 driver 和 add-ons；具体根因待对照验证，不解释为有人占卡。 | 保留并停止本任务失败容器，使用完整 driver/add-ons 只读挂载重新创建，仍仅映射卡 0、1。待验证。 |

E07 更新：完整 driver 挂载、SYS_ADMIN/关闭 seccomp、UID/GID=1000、仅运行时可见卡变量四类尝试仍失败，完整原始日志保留在 logs/npu-*.log；标准 privileged 容器配合 ASCEND_VISIBLE_DEVICES=0,1 后两卡 tensor 分配/拷回均成功。已确认容器设备/权限配置相关，精确是哪一项驱动访问限制尚未隔离。实际性能在新 runtime 容器运行，构建容器保留。privileged 允许访问宿主设备节点，运行时可见设备固定为物理 0、1，不将其表述为设备 cgroup 的强隔离。

| 编号 / 发现时间 | 现象、影响与证据 | 已确认原因 / 假设 | 处理、验证与状态 |
| --- | --- | --- | --- |
| E08 / 2026-10-10，SSH 认证核验 | 用户所述新 key 属于 wangzhao-11a，但本机 ssh -T git@github.com 实际返回 Hi Liuchenbing-2026。仓库读取已经成功。 | 本机默认 key 的实际认证身份与给定描述不同；是否是其他环境的 key 待核查。 | 核查远端主机现有 key 身份，不更改全局 SSH 配置、不复制私钥；推送按实际认证身份验证。未解决。 |
| E09 / 2026-10-10 01:51:42+08:00 | torch 胶水扩展已生成，但 golden 精度入口缺少 ml_dtypes，ModuleNotFoundError；尚未执行 bit-exact 判定。原始证据 logs/precision_glue_precheck.log。 | 原镜像/复现指南没有提供该 golden 外部依赖。 | 安装固定 ml-dtypes==0.5.3 并记录安装位置/命令后重跑；不弱化精度断言。待验证。 |

E08 更新：.19 默认 SSH key 实测身份为 CCH-gif，本机为 Liuchenbing-2026；均不是给定描述中的 wangzhao-11a。尚未定位该 key，本机已有仓库读取权限，不改变其他用户的 SSH 配置。

## 复现与验收

基础镜像完整 tag、registry digest / 本地 image ID、额外依赖、框架 SHA、模型配置哈希、实际构建和启动命令、压测与精度命令将在执行时更新。静态检查、编译完成、冒烟、正式性能与精度分别记录。

E09 更新：安装 ml-dtypes==0.5.3 至 /ws/precision-deps 后，原始五组 bit-exact 测试全部通过，idx/qjl/norm/gamma 均一致；4-bit roundtrip 通过。证据 precision_glue_installed_deps.log、precision_roundtrip_precheck.log。早于依赖安装完成的重试也失败，失败日志保留；该依赖问题已处理，不代表整模型质量评测通过。

| 编号 / 发现时间 | 现象、影响与证据 | 已确认原因 / 假设 | 处理、验证与状态 |
| --- | --- | --- | --- |
| E10 / 2026-10-10，框架源构建 | vLLM editable 源构建成功；Ascend 原生算子构建后，CMake find_package(Torch) 找不到 TorchConfig.cmake，整体安装退出 1。证据 logs/build-failure-context.log 和原始 bootstrap.log。 | setup.py 传入的 CMAKE_PREFIX_PATH 仅含 pybind11；系统 torch/share/cmake 中 TorchConfig.cmake 实际存在，未进入查找路径。 | 在构建脚本显式导出 torch.utils.cmake_prefix_path，保留失败日志并从现有源构建目录恢复；没有替换预编译框架二进制。待验证。 |

E06 更新：任务根目录已由 Docker 创建并限于该目录 chown；其容器内部创建的 artifacts 子目录归 root，宿主普通账号写 manifest 失败。改由本任务容器写入该目录，保留日志 manifest-and-resume-status.log；不扩张宿主目录权限。

E10 原因更正：额外 CMAKE_PREFIX_PATH 后仍失败，原始恢复日志保留为 framework-resume-autoload-failed.log。继续核验发现 build 容器 import torch 会自动加载 torch_npu，向 stdout 输出 DrvMngGetConsoleLogLevel failed (ret=4)，污染 CMake 读取的 torch.utils.cmake_prefix_path（和版本字符串）。设置固定源码安装文档要求的 TORCH_DEVICE_BACKEND_AUTOLOAD=0 后，Torch prefix/version 输出干净；在同一源码构建目录再恢复。不能把第一次路径补齐当作修复完成。

| 编号 / 发现时间 | 现象、影响与证据 | 已确认原因 / 假设 | 处理、验证与状态 |
| --- | --- | --- | --- |
| E11 / 2026-10-10，启动前静态核对 | 初拟 YaRN 原始长度 32768 × factor 4 仅支持 131072，总工作 131072+1024=132096 会超过派生限制。未将该配置用于实测。 | vLLM v0.28.0 model.py 对 yarn 使用 original_max_position_embeddings×factor 推导长度；该模型原 config.max_position_embeddings=40960。 | 两侧统一使用 original_max_position_embeddings=40960、factor=4.0，派生上限163840，服务max_model_len=132096；待服务日志和精确长度冒烟核验。只评性能，不据此宣称长文本质量通过。 |

E10 更新：framework-resume.log 记录 Ascend C++/原生扩展构建与 editable 安装成功，bootstrap.exit=0。构建阶段关闭自动后端加载消除了 CMake 输入污染；最后的源码路径核验暴露了另一个环境问题 E12，下述检查完成前不启动正式对比。

| 编号 / 发现时间 | 现象、影响与证据 | 已确认原因 / 假设 | 处理、验证与状态 |
| --- | --- | --- | --- |
| E12 / 2026-10-10 02:08:53+08:00 | 共享系统依赖 venv 中，vllm metadata 显示新构建0.28.0+empty，但 import vllm 实际命中基础镜像 /usr/local/.../site-packages/vllm；源码未优先。尚未压测。 | system-site-packages 中原有真实目录先于 editable finder 被查到；版本标签不能证明实际源码。 | 服务与核验显式前置 /ws/source/vllm:/ws/source/vllm-ascend 到 PYTHONPATH，并断言模块实际路径；不删除基础镜像的原有包。待验证。 |

E12 更新：显式源码路径后 import vllm=/ws/source/vllm/vllm/__init__.py（__version__=0.28.0）、vllm_ascend=/ws/source/vllm-ascend/vllm_ascend/__init__.py。该核验命令末尾错误探测了不存在的 vllm_ascend._C；实际生成的扩展名称为 vllm_ascend.vllm_ascend_C，错误探测日志保留，后续使用正确名称核验。未据错误模块名判断编译失败。

| 编号 / 发现时间 | 现象、影响与证据 | 已确认原因 / 假设 | 处理、验证与状态 |
| --- | --- | --- | --- |
| E13 / 2026-10-10，依赖核验 | pip check 报 vLLM API FastAPI/Starlette 版本不足、缺 fastokens，及镜像内未用于此任务的 profiler/vision 包依赖冲突。证据 package-check-and-binaries.log、pip-check.log。 | 固定 vLLM 要求 FastAPI>=0.133,<0.137、Starlette>=1.0.1，但固定 Ascend requirements.txt 要求 FastAPI<0.124，两方元数据约束无共同交集。 | 为实际API安装 fastapi=0.136.0、starlette=1.0.1、fastokens=0.2.0，记录 Ascend 的元数据约束偏离，不谎称 pip check 全绿。ml-dtypes=0.5.3 安装到正式venv，最终测试使用原numpy1.26.4；早期precision-deps/numpy2.5.3结果单独保留，不用于服务。实际服务兼容性待验证；无关profiler/vision依赖未处理。 |

E13 更新：runtime 容器中首次调用 uv 未找到，证据 serving-dependencies-install.log；uv 此前只安装在 build 容器系统路径。改由 build 容器向共享任务 venv 安装 uv==0.12.24 及服务依赖，后续 runtime 从 /ws/.venv/bin 访问；不误称两容器的系统层同步。

E13 实装版本更正：uv 为共享 venv 实际解析/安装 numpy==2.5.3、pydantic==2.14.0、pydantic-core==2.50.0（完整列表见 serving-dependencies-install-build.log），最终服务/精度使用这些实际版本，不能继续沿用拟保持numpy1.26.4的描述。原镜像numpy1.26.4仍在系统层，但venv包优先；最终精度将重新核验。

| 编号 / 发现时间 | 现象、影响与证据 | 已确认原因 / 假设 | 处理、验证与状态 |
| --- | --- | --- | --- |
| E14 / 2026-10-10，首次服务初始化 | NumPy2.5.3超出镜像SciPy所要求<2.3，出现实际导入警告；默认插件自动加载 msserviceprofiler、arctic_inference 等无关插件，正式性能范围不明确。初始启动日志 matrix_20261009T181235Z/serve_bf16_0.log（尚未发请求）。 | 依赖解析选到了新NumPy；vLLM默认加载所有已安装general plugins。是否已实际采集profiling数据尚未确认，不能宣称已发生profiling开销。 | 停止本任务首次初始化，固定numpy2.2.6并重新核验精度；两侧统一VLLM_PLUGINS白名单，仅Ascend平台/模型/loader/connector及本项目TQ。依据服务线程警告统一OMP_NUM_THREADS=1。保留原启动记录，正式请求尚未开始。待验证。 |

E14 更新：numpy2.2.6固定安装成功；首次setup停止保留，第二次controller于修订配置重启。服务白名单为 ascend,ascend_kv_connector,ascend_model,ascend_model_loader,kv_cache_turbo_quant_shadow；正式两侧OMP_NUM_THREADS=1。首次精度通过记录仍保留，第二次将验证最终numpy环境。

E14 范围更正：停止前旧配置的 BF16 短请求冒烟已成功（64→16），128K→1 冒烟已经启动，后被配置收口中断；正式六档矩阵仍未开始。先前“尚未发请求”的记录不准确，不能删除历史偏差；保存原smoke、progress、启动日志，不把配置中断当TQ/BF16性能超时。最终pip核验显示Triton要求numpy==1.26.4，正式环境据此固定1.26.4；opencv vision包的numpy>=2约束未处理，Qwen文本测试不使用该包。

| 编号 / 发现时间 | 现象、影响与证据 | 已确认原因 / 假设 | 处理、验证与状态 |
| --- | --- | --- | --- |
| E15 / 2026-10-10 02:17:09+08:00 | 第二次controller启动报端口18377 Address already in use。原启动停止后，旧API等待在途128K请求结束，bench_client子进程仍在。证据 matrix-attempt2.log、port-restart-diagnosis.log。 | 首次手工SIGTERM只停止controller/API，没有结束由controller启动的client；API优雅关闭未结束。 | 容器内按cmdline核验本任务PID后停止遗留client/API，验证端口可绑定；controller补SIGTERM处理和client进程组finally清理。其他容器/服务未动。待第三次启动验证。 |

E15 更新：核验PID身份后停止本任务遗留API/client，日志 setup-orphan-cleanup.log 证实端口18377可绑定。第三次controller启用子进程组清理及SIGTERM处理重新启动；不删除前两次记录。

执行记录：整理阶段报告的长内联Python命令因stdin编码报SyntaxError，未写入；后续改用文件补丁写入。首次补丁的上下文不匹配也未修改文件，随后改用实际完整行定位；不影响远端测试。

## 实际配置与构建证据（阶段记录）

- 新建runtime容器 `tq-128k-ab-20261010-runtime`、build容器 `tq-128k-ab-20261010`；共享目录 `/data1/tq-128k-ab-20261010`，容器内 `/ws`。失败容器与日志保留，仅本任务进程被重启。
- 基础镜像 `vllm-ascend:dspark-a2-028`，本地image ID `sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`，arm64，**RepoDigests为空，既有本地镜像，没有可宣称的registry digest或公共拉取地址**。未执行docker commit/save/upload。官方0.26镜像RootFS不同，未用它替代本轮镜像、未验证可等价复现。
- vLLM固定源码 `v0.28.0` / `2cf0a6915ce544dc493a0990f2ea38d81601128a`，empty目标完整执行editable源构建；实际module版本0.28.0，package metadata0.28.0+empty。Ascend固定源码 `v0.28.0.rc1` / `96df623103f921df1a4488170d106f36689acb59`，完整原生算子及C++扩展新编译。Catlass固定 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`（基础镜像中的源码，不复用旧.so）。两框架本地git克隆未修改，detached HEAD；未为这些未改仓库新建分支。
- TQ固定原始58文件未修改。根据code/op新编译OPP/customize和torch胶水；根据code/integration配置插件扁平路径，未改shadow/store算法。
- TP=2，BF16模型权重，eager，chunked prefill=4096，max_num_seqs=32，prefix cache=false，async scheduling=false，OMP=1；模型只读挂载 `/models/Qwen3-30B-A3B`。
- 两侧显式KV预算 `25769803776` bytes =24.00 GiB/卡。日志确认该设置跳过自动KV内存profiling，gpu_memory_utilization=0.92不替代这个手动预算。BF16日志容量524288 tokens，132096总长度最大驻留并发3.97x；TQ容量待自身日志核验。
- 两侧长上下文覆盖 `rope_type=yarn,factor=4.0,original_max_position_embeddings=40960`，max_model_len=132096。原模型config上限40960；这是本轮性能配置，不代表已评长上下文语义质量。
- 固定种子20261010生成整数token IDs（1000..9999），每并发一组同时到达的N个请求，N=并发1/2/4/8/16/32；ignore_eos=true、temperature=0、max_tokens=1024。payload在计时前构造；两侧逐请求SHA256须相同，按实际usage核验131072/1024。
- 正式请求timeout=1800s，起服timeout=1200s；两侧先64→16短冒烟、131072→1长冒烟。未完成组吞吐/比值为空，失败与超时单列；不将部分tokens作为完整1024输出。TTFT从首次非空SSE文本到达计量；TPOT=(最后文本−首次文本)/(output_tokens−1)，不是逐token ITL。
- 吞吐与时延包含排队、prefill、decode，只排除payload构造和服务启动。当前每格单组测量，不宣称稳态吞吐、可信P99或重复性已评。
- 原5组bit-exact判据全部通过（最终numpy1.26.4环境再次通过），4-bit roundtrip及Q侧旋转等价检查通过；未做整模型准确率/128K语义质量测评。

源档案hash见source-archives.sha256；模型配置/tokenizer/权重索引hash见model-manifest.json。16个权重文件仅记录尺寸，未计算全量weight hash，本地已有模型revision来源未核验。实际依赖与native binary hashes见environment.json（最终numpy版本清单待刷新）及相关安装日志。

E16 / 2026-10-10，结果拉回：scp读取首个完整结果时报Connection closed，结果未拉回；客户端随后空目录生成了“未测”临时汇总，不能当远端实测状态。未覆盖远端数据。具体SFTP失败原因未确定；改用经过现有SSH连接读取精确结果文件，检查JSON和本地/远端文件hash后再汇总。监控命令还曾漏写docker exec -i，导致stdin脚本未执行、仅返回profile警告；已使用-i传入固定monitor脚本，未影响压测。

E16 更新：通过SSH读取C1结果成功，本地与远端SHA256均为e76b60ffd625391761d50e8c17798f021af88860bf0f6c1f2a7078d6896e9694，summarize_results.py已验证usage及吞吐分母。SFTP根因未修复。

## 完整入口与命令

以下均以宿主任务目录 `/data1/tq-128k-ab-20261010`、两个上述容器和已存在的只读权重为前提。`scripts/create_container.py`包含实际runtime创建参数与空闲检查；`logs/actual-container-configs.log`记录两个实际容器配置。镜像需已存在于该宿主，不能把本地image ID当registry digest拉取。

实际build容器创建参数：

```bash
docker run -d --name tq-128k-ab-20261010 --network host --shm-size 32g --user 0 \
  --runtime ascend --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -v /data1/tq-128k-ab-20261010:/ws \
  -v /data1/models/Qwen3-30B-A3B:/models/Qwen3-30B-A3B:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  vllm-ascend:dspark-a2-028 sleep infinity
python3 scripts/create_container.py
```

源码：本轮在本机克隆上述固定提交，以git archive生成三份source.tar.gz后传入宿主共享根目录。`scripts/prepare_source_archives.sh`提供按完整SHA重新取得同样源档案的替代命令（静态核对，未在本轮宿主从头执行）；它要求GitHub读取权限，不复制任何私钥。脚本不会覆盖已有checkout。新宿主先创建任务挂载目录，只调整该目录归属；不改变/data1或其他任务权限。

本轮实际构建先运行原bootstrap，因E10失败后在同一新源码目录使用恢复入口，编译结果有完整日志：

```bash
docker exec -d tq-128k-ab-20261010 bash -c \
  'bash /ws/bootstrap.sh; printf "%s\n" "$?" >/ws/bootstrap.exit'
docker exec -d tq-128k-ab-20261010 bash -c \
  'bash /ws/scripts/resume_framework_build.sh; printf "%s\n" "$?" >/ws/bootstrap.exit'
docker exec -d tq-128k-ab-20261010-runtime bash -c \
  'bash /ws/build_tq_op.sh; printf "%s\n" "$?" >/ws/tq-op-build.exit'
```

归档的 `scripts/bootstrap.sh` 已整合实际必要的环境修正、固定依赖和源码路径断言，供恢复：

```bash
bash scripts/prepare_source_archives.sh /data1/tq-128k-ab-20261010
# 将本目录scripts复制至宿主任务目录scripts/，容器通过/ws/scripts访问
docker exec tq-128k-ab-20261010 bash /ws/scripts/bootstrap.sh
docker exec tq-128k-ab-20261010-runtime bash /ws/scripts/build_tq_op.sh
```

这个整合脚本只做了静态检查，**没有再次在干净镜像从头运行**；本轮源构建、原生编译、依赖修正和实际精度/服务验证分别提供日志，不把分步成功说成从零脚本验收。

实际最终额外依赖通过共享venv安装（完整实际解析版本另见environment.json），没有改框架仓库源码：

```bash
docker exec tq-128k-ab-20261010-runtime bash -c \
  'TORCH_DEVICE_BACKEND_AUTOLOAD=0 /ws/.venv/bin/uv pip install --python /ws/.venv/bin/python numpy==1.26.4'
# API依赖、ml-dtypes、uv的安装在build容器执行，安装到同一个/ws/.venv
# uv==0.12.24 ml-dtypes==0.5.3 fastokens==0.2.0 fastapi==0.136.0 starlette==1.0.1
```

精度与独立压测入口（容器内，先显式source CANN并设置源码路径）：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/ws/.venv/bin:$PATH
export PYTHONPATH=/ws/source/vllm:/ws/source/vllm-ascend:/root/kvtq_integration:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0,1 TORCH_EXTENSIONS_DIR=/ws/torch-extensions VLLM_ASCEND_KVTQ_BITS=4
/ws/.venv/bin/python /root/kvtq_integration/torch_ext/test_glue.py
/ws/.venv/bin/python /root/kvtq_integration/test_store_roundtrip.py
bash /ws/scripts/serve.sh 0  # BF16；另一次使用1为真实TQ store 4-bit
# 服务ready后在另一个容器exec中运行，六档并发逐一改--concurrency
/ws/.venv/bin/python /ws/scripts/bench_client.py --mode bf16 --concurrency 1 \
  --input-len 131072 --output-len 1024 --timeout 1800 --output /ws/results/bf16_c1.json
```

本轮自动矩阵实际启动命令（attempt1/2日志保留；最终attempt3）：

```bash
docker exec -d tq-128k-ab-20261010-runtime bash -c \
 'source /usr/local/Ascend/cann-9.1.0/set_env.sh; \
  export PATH=/ws/.venv/bin:$PATH PYTHONPATH=/ws/source/vllm:/ws/source/vllm-ascend:/root/kvtq_integration; \
  /ws/.venv/bin/python /ws/scripts/run_matrix.py >/ws/logs/matrix-attempt3.log 2>&1; \
  printf "%s\n" "$?" >/ws/matrix-attempt3.exit'
# 每个cohort实际完整参数还保存为logs/<run_id>/*.command.json
python3 scripts/summarize_results.py --results results --selection selection.json --output /tmp/tq-result-summary.md
```

汇总核验每个成功请求的实际长度、成功/失败数、吞吐分母，两侧payload SHA256逐一相同才计算比值。完整矩阵与最终结果尚在运行；目前有证据的完成范围是精度、最终起服、短/128K长冒烟、BF16已完成的正式cohort，不宣称TQ正式性能已通过。

## 请求时限调整与继续运行

BF16 C4完整组耗时466.11s，原1800s请求时限可能提前截断高并发。已完成BF16 C1/C2/C4与正在运行的C8保留原计时/timeout=1800s；在C8完整结束后，将后续请求时限设为7200s。两侧模型、源码、binary、KV预算、请求输入/输出、采样与计时边界不变。已完成请求均早于原时限结束，时限调整不改变它们的吞吐分母；失败或被计划切换中断的记录仍保留。

`continue_after_c8.py`等C8完成后，停止原controller的后续调度，按顺序执行TQ C1/C2/C4/C8、BF16 C16/C32、TQ C16/C32，每个请求7200s；这样能先取得已完成BF16对应的TQ数据。若切换时原C16已开始，保留其中断progress并重新完整测量，不将中断记录计入性能。

```bash
docker exec -d tq-128k-ab-20261010-runtime bash -c \
 '/ws/.venv/bin/python /ws/scripts/continue_after_c8.py \
  >/ws/logs/deadline-continuation-reordered.log 2>&1; \
  printf "%s\n" "$?" >/ws/deadline-continuation-reordered.exit'
# 继续runner的完整参数保存为logs/*_concurrency.command.json
# 如重新跑完整矩阵，可使用以下入口（本轮未以这个一次性命令从头重跑）：
/ws/.venv/bin/python /ws/scripts/run_matrix.py --request-timeout 7200
```

复用精度明确引用最终环境的已通过日志，并核验torch/torch-npu/numpy/ml-dtypes版本和所有已记录native artifact哈希。当前environment.json已更新为numpy1.26.4，含torch胶水.so哈希；两rank实际/proc maps证明加载的是本轮源码目录中的新Ascend C++/kernels/custom_transformer binary，见active-native-paths.log。早先awk转义命令未取得有效路径，错误记录保留，后改Python读取成功。

静态检查还发现第一次增加runner参数时把mode循环缩进写错，未上传或执行；修正后py_compile和--help通过。继续运行尚在执行，自动cleanup修复没有单独做信号回归测试，不能写成已经完整验证。调整等待顺序只重启未占卡的watcher，benchmark controller/client/server未变；原watcher退出记录保留。

| 编号 / 发现时间 | 现象、影响与证据 | 已确认原因 / 假设 | 处理、验证与状态 |
| --- | --- | --- | --- |
| E17 / 2026-10-10 02:52+08:00 | C8已8/8完成，但planned transition后socket.bind检查端口18377报98，continuation停止。原controller已按SIGTERM退出130；无本任务活API/client，只有已退出worker僵尸。ss工具缺失，后续现场检查两种bind都成功。证据transition-port-diagnosis.log、transition-port-bind-comparison.log。 | 最初端口状态未捕获，TIME_WAIT为与现象一致的推断，不能说现场根因已证明；preflight未设置SO_REUSEADDR已确认。 | 对另一个自选临时端口做真实TCP关闭对照：非reuse bind失败98、reuse bind成功，tcp-time-wait-preflight-regression.log。controller/watcher加SO_REUSEADDR；C8数据保留，手动从已记录的transition继续。恢复状态待实际起服核验。 |
| E18 / 2026-10-10，恢复入口 | continue_after_c8 --after-transition报AttributeError: list has no after_transition，未启动TQ。证据deadline-continuation-recovered.log。 | /proc扫描中的args列表覆盖argparse Namespace变量，已确认。 | 扫描变量改process_args，保留原失败日志，以同一恢复入口重跑；尚未据py_compile宣称端到端通过。 |

E17/E18 更新：修订恢复入口实际启动TQ服务，短64→16与长131072→1冒烟均成功，正式store4 C1开始。原controller按计划终止130，C8已经8/8完成；SIGTERM对子进程的清理及后续起服在此次真实切换中完成，不再把这项仅写成静态核验。现场原端口的TIME_WAIT原因仍为推断，真实TCP对照与修订恢复起服成功分别记录。

源码证据：运行中的旧controller在磁盘文件更新后，其异常栈会显示新文件行文，不能凭该行文推断旧RAM代码。已从本任务Git中恢复旧实际controller blob `29f8ad67fd4168a96840f08d77c6c14c3fb1d27b`，归档为scripts/controller_attempt3_actual.py；新phase使用带显式modes/concurrencies/request-timeout参数的runner。

本轮容量同预算配对核验：BF16=524288 tokens，TQ=2033536 tokens，TQ/BF16=3.878662109375；同为24.00 GiB/卡，初始free分别60.57/60.58 GiB。132096总长度最大驻留并发3.97x→15.39x。仅为容量，不当作性能加速比；TQ正式吞吐在运行。

03:18阶段核验：E03设备句柄在正式起服前已补查，未发现占用；E05插件路径部署已实际通过短/长冒烟；E06任务目录写入已验证。E08认证身份仍不符给定描述，本机实际Liuchenbing-2026身份向任务分支正常推送并回读成功，未声称找到wangzhao-11a的key。后续本地读取证据曾将pulled/artifacts/environment.json和尚未拉回的服务日志写成错误路径，命令只读失败、未影响测试；改用rg列举实际文件和SSH快照拉回33个证据文件，逐文件SHA256核验。E16的SFTP根因仍未处理。

归档修订：初版额外新增REPORT.md，并重复保存canonical结果，与archive-organize的一份README约定不符。已将实测条件、计量、精度边界和阶段表合并至本README；由selection.json直接选取原run目录结果，保留Git中的旧版本，不再交付详版副本。原日志/ws/logs/<run_id>归档映射为server_evidence/<run_id>；快照manifest记录原相对路径与SHA256。

当前恢复入口实际命令（E17/E18修订后）：

```bash
docker exec -d tq-128k-ab-20261010-runtime bash -c \
 '/ws/.venv/bin/python /ws/scripts/continue_after_c8.py --after-transition \
  >/ws/logs/deadline-continuation-recovered2.log 2>&1; \
  printf "%s\n" "$?" >/ws/deadline-continuation-recovered2.exit'
```

当前TQ低并发run为matrix_20261009T185712Z；每phase的完整参数与来源文件hash见server_evidence/<run_id>/run_parameters.json。`summary.json`还记录被选原始结果source_path与source_sha256；selection仅列正式cohort，不选冒烟或中断progress。
