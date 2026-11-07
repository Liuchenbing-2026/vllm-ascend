# DSV4 FP8 MegaMoE / Ascend 950DT

| 项目 | 状态 |
| --- | --- |
| routed FP8 算子 | 已实现原生类型检查、JIT 模板分流、AIV0 NZ 加载及双缓冲事件；保留 FP4 路径 |
| vLLM 接入 | DSV4 block-FP8 → MXFP8 → MegaMoE；共享专家融合；使用已有配置开关 |
| 主机验证 | 52 项 adapter/打包测试、6 项真实权重转换测试通过；固定基线补丁重复重放及 `format.sh ci` 通过 |
| 设备验证 | 未在 950DT 编译、执行、测精度或性能；不属于硬件验收通过的版本 |
| 适用范围 | TP1、线性 EP8、BF16 输入、SwiGLU；不支持 EPLB、DBO 或权重热更新 |

## 固定版本与实现

- 集成基础：`Liuchenbing-2026/vllm-ascend`，`dev/v032-megamoe-dsl080`，`80f1a251c30bf7057036677e855515c39f373815`。
- 任务分支：`v032-megamoe-dsl080_dsv4`。交付版本以该文件所属提交及 `git rev-parse HEAD` 为准。
- 算子基础：[DeepGEMM-Ascend 3a35326271b069b8b5bf1a754435acd6cc91575b](https://github.com/deepseek-ai/DeepGEMM-Ascend/tree/3a35326271b069b8b5bf1a754435acd6cc91575b)。
- 原生构建子模块：DeepJIT `65f513952b8771408a19202f65e36398522db6db`、magic_enum `b233b96e49d371bad00300f59b5ba581100b8745`。
- vLLM 目标版本沿用仓库 `.github/vllm-release-tag.commit`：`v0.32.0`。本机未安装 vLLM，未验证该完整运行栈。
- 实际主机测试：macOS、Python 3.12、torch 2.8.0 CPU、pytest 9.2.0；没有使用基础镜像。
- 950DT 镜像完整 tag/digest、驱动、CANN、HCCL、编译器实际版本：**未采集**，不能从旧任务推断。
- 仓库依赖要求包括 torch 2.10.0、torch_npu 2.10.0.post7；设备构建还需要匹配的 CANN SDK、GCC 13+、elfutils headers。实际安装版本以目标环境为准。

`patches/routed-fp8.patch` 包含完整原生修改和算子测试，并非只放开 Python 类型判断。
`build.py` 将干净、固定提交的源码复制到独立构建目录，应用补丁后编译原生绑定与设备头文件。
源 checkout 不被修改。打包清单 `ascend_build.json` 记录基础 SHA、补丁 SHA-256、子模块和实际依赖版本。
不需要另建 DeepGEMM fork；这份依赖修改随当前任务分支交付。

FP8 权重采用 E4M3、每 32 个 K 元素一个 UE8M0 scale，两个 scale 字节装进一个 int16。
原生 FP4 的 K/2 存储保留；FP8 使用完整 K，物理 NZ 布局为 `[K/32, N, 32]`。
新增 MTE2→MTE3 完成事件与反向复用事件，保持原 L1B/AIC 握手；退出时回收全部 FP8 UB credits。
权重类型作为 JIT 模板参数进入代码缓存键，没有改变内核参数中 `layout::Weights` 的 ABI。

DSV4 `expert_dtype="fp8"` 的 FP32 block scales 经已有 block-FP8 scheme 逐专家反量化、重新量化为 MXFP8。
不能把任意 FP32 scale 截取指数后直接当成同一权重。新 FP8 路径的共享专家也执行重新量化。
这会引入重新量化误差，必须在模型精度评测中验证；原有 FP4 共享转换行为保持不变。
外部旧版 DeepGEMM 缺少原生能力标记时明确报错，不静默执行错误布局。

## 已执行的主机检查

在仓库根目录、具有上述 CPU torch/pytest 的 Python 环境运行：

```bash
python tools/deepgemm/check_host.py
```

该入口运行实际 adapter、builder 和 loader 的测试，仅对不可用的 vLLM/NPU 导入边界设置桩。
52 项通过不代表真实 NPU kernel 或完整 vLLM 单测通过。量化配置的新增标准 UT 已编写，完整 vLLM 环境执行待补。

将算子补丁应用到固定基线后运行实际权重转换测试：

```bash
git clone https://github.com/deepseek-ai/DeepGEMM-Ascend.git /tmp/dsv4-deepgemm
cd /tmp/dsv4-deepgemm
git checkout 3a35326271b069b8b5bf1a754435acd6cc91575b
git switch -c test/dsv4-fp8
# VLLM_ASCEND_SRC 指向当前任务分支的绝对目录
git apply --check "$VLLM_ASCEND_SRC/tools/deepgemm/patches/routed-fp8.patch"
git apply "$VLLM_ASCEND_SRC/tools/deepgemm/patches/routed-fp8.patch"
python -m pytest --noconftest -q tests/test_mega_moe_host.py
```

6 项通过，覆盖 FP8 NZ 的独立字节解码、scale 保留、FP4 指数补偿回归、格式不匹配拒绝及旧原生模块拒绝。
补丁还从固定基线独立重放两次，逐文件与开发源码比对一致，原始源码保持不变。
额外 22 项地址算例通过，但没有模拟真实 DMA 或同步硬件。

## 950DT 构建与验收命令（未执行）

以下是待执行入口，不是已成功部署的环境配方。先确认八卡资源使用权和容器实际版本。
不使用已暂停 CANNBench 任务的环境、令牌或额度。

在匹配 CANN/torch_npu 的 950DT 容器中，从本任务分支构建：

```bash
python -m pip install --no-build-isolation -e .
python -c 'from vllm_ascend.ops.fused_moe.deepgemm_loader import load_deepgemm; d=load_deepgemm(); assert d.SymmBuffer.supports_mma_type("fp8xfp8"); print(d.__file__)'
```

A5 构建沿用仓库的 SOC 检测与原生扩展配置；不能用关闭自定义算子的纯 Python 安装代替。
如果环境安装了旧外部 NPU `deep_gemm`，需先按同一补丁重新构建它，或由环境所有者移除冲突安装后使用内置包。

独立算子精度测试在上述已应用补丁的 DeepGEMM checkout 中执行；还需要其基线使用的 DeepEP、TileLang 及原仓库测试依赖。
这些基线依赖尚未在当前环境安装，因此版本和实际运行日志待补。

```bash
git submodule update --init --recursive third-party/deep-jit third-party/magic_enum
python -m pip install --no-build-isolation --no-deps -e .
python tests/test_mega_moe.py --weight-format fp8 --num-processes 8 \
  --hidden 7168 --intermediate-hidden 2304 --num-experts 256 --num-topk 6 \
  --num-tokens 0 1 17 129 513 --num-shared-experts 0 1 2 \
  --num-correctness-tests 10 --skip-prof
python tests/test_mega_moe.py --weight-format fp8 --num-processes 8 \
  --hidden 7168 --intermediate-hidden 2304 --num-experts 256 --num-topk 6 \
  --num-tokens 17 129 --empty-rank 0 --num-correctness-tests 10 --skip-prof
# 同形状 FP4 回归，保持原有精度阈值
python tests/test_mega_moe.py --weight-format fp4 --num-processes 8 \
  --hidden 7168 --intermediate-hidden 2304 --num-experts 256 --num-topk 6 \
  --num-tokens 1 129 --num-correctness-tests 10 --skip-prof
```

测试基线是独立 dispatch、原始 ND 权重 grouped GEMM、SwiGLU、combine，不复用新 NZ 加载路径。
单卡模式或 `--skip-baseline` 会跳过参考精度检查，不能用其 PASS 代替精度验收。
精度通过后移除 `--skip-prof` 采集算子性能，保留日志于私有目录；不把历史 FP4 性能作为 FP8 结果。

整网启动模板（`MODEL` 必须指向真实 FP8 checkpoint，模型版本与配置需要归档）：

```bash
vllm serve "$MODEL" --quantization deepseek_v4_fp8 --dtype bfloat16 \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --enforce-eager --max-num-batched-tokens 1024 --max-model-len 4096 \
  --additional-config '{"deepgemm_mega_moe":true,"enable_fused_mc2":0}'
# 独立服务压测入口；不替代模型精度评估
vllm bench serve --backend vllm --model "$MODEL" --dataset-name random \
  --random-input-len 128 --random-output-len 128 --num-prompts 32
```

完整 checkpoint 加载、参考模型精度、图模式、长时间压力和性能尚未测；本次未提供已验证的整网分数。

## 问题台账（发现日期均为 2026-11-07）

| 问题与影响 | 原始证据/原因 | 处理及当前状态 |
| --- | --- | --- |
| 早期把生成器当实现进展，产生沟通偏差 | 仅有 `apply_w8a8.py` 与地址检查，没有完整编译或集成 | 已明确纠正；本次提交真实源码和构建接入；设备验证仍未完成 |
| 下载失败 | curl DNS、SSL_ERROR_SYSCALL；SSH 连接关闭；连接器 Unknown tool | 用现有本地代理恢复 Git/HTTPS；原网络故障根因未确认，连接器未修复 |
| 初稿修改 Weights ABI | 参数布局与 launcher 打包不一致 | 放弃该字段，改用 JIT 模板参数；原生成器已废弃，不能再应用 |
| 同步自动审计不覆盖封装 | 扫描结果 event_count=0；案例数据库重建报 unable to open database file | 人工检查事件方向/编号/复用与退出；不宣称自动审计通过，设备同步验证待补 |
| Python 缓存写权限 | 初次 py_compile 写系统缓存失败 | 使用任务目录缓存恢复；未更改系统权限 |
| 全量 format 检查 | 首次 DNS 失败；代理重试后格式类检查通过，但缺 gitleaks/wget、shellcheck | 复用现有工具后完整 `format.sh ci` 本地通过；先前失败记录保留，不代表远端 CI 通过 |
| 硬件环境缺失 | 当前仅本机 CPU 环境，未指定 950DT 机器/容器 | 没有设备编译、正确性、性能结论；不占用旧任务服务器 |

完整 PR 尚未创建，发布前必须提交用户审核。release-note 文案应说明“实验性 DSV4 MXFP8 MegaMoE”；有 PR 编号后再按仓库模板添加编号片段，不伪造编号。
