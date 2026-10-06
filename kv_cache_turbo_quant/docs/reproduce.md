# kv_cache_turbo_quant 新环境完整复现指南

目标：在一台**全新**的昇腾机器上，从零开始复现本任务的全部验收结论 —— 算子 bit-exact 精度、vllm-ascend 接入可用性、写路径开销(口径 A)、KV 容量收益(口径 B/C)、量化精度(口径 D)。

预计耗时：环境就绪后约 1 小时(算子编译 ~10 min,bench 两组各 ~15 min)。

## 0. 环境前提

| 项 | 要求 | 本次验证值 |
|---|---|---|
| 硬件 | Atlas 800T A2 (910B4),>=2 张空闲 NPU | 物理卡 2、3 |
| 驱动/CANN | CANN 9.1.0+ | `/usr/local/Ascend/cann-9.1.0` |
| 容器镜像 | vllm-ascend v0.28.0rc(vllm 0.28.0 + torch_npu 2.10.0.post4 + python3.12) | 容器 `vllm-ascend-v0.28.0rc-test` |
| 模型 | Qwen3-30B-A3B(MoE, 48 层, 4 KV heads, head_dim 128) | `/home/models/Qwen3-30B-A3B` |

已验证的其他模型(冒烟级):Qwen3-1.7B、Qwen3-8B、GLM-4-9B-0414。

### 0.1 拉起容器(若无现成容器)

```bash
IMG=<vllm-ascend:v0.28.0rc 镜像>   # 用本机已有的 vllm-ascend v0.28.0rc 镜像
docker run -d --name kvtq-repro \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --shm-size 16g --network host \
  $IMG sleep infinity
```

### 0.2 部署代码包

```bash
# 宿主机:将本提交目录拷入容器(后续命令默认容器名 kvtq-repro)
docker cp 04_tasks/01_community-task-2026/tasklist/09-kv_cache_turbo_quant/liuchenbing-2026 \
  kvtq-repro:/root/kvtq_integration
# 容器内目录约定:/root/kvtq_integration 下为 README.md docs/ code/ results/
```

注意:`code/bench/*.sh` 中默认路径为 `/root/kvtq_integration`、模型 `/home/models/Qwen3-30B-A3B`、设备 `2,3`、端口 `8377`;如环境不同,改脚本头部参数即可(均带 usage 注释)。

## 1. 编译安装 OPP 算子(~10 min)

```bash
docker exec -i kvtq-repro bash -s <<'EOS'
set -e
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /root/kvtq_integration/code/op
msopgen gen -i KvCacheTurboQuant.json -c ai_core-ascend910b -lan cpp -out gen
cp op_host/kv_cache_turbo_quant.cpp   gen/op_host/
cp op_kernel/kv_cache_turbo_quant.cpp gen/op_kernel/
cp op_kernel/kv_cache_turbo_quant_tiling.h gen/op_kernel/
python3 - <<'PYEOF'
import json
p = "gen/CMakePresets.json"
d = json.load(open(p))
for preset in d["configurePresets"]:
    cv = preset.get("cacheVariables", {})
    cv["ASCEND_COMPUTE_UNIT"] = {"type": "STRING", "value": "ascend910b"}
    cv["ASCEND_CANN_PACKAGE_PATH"] = {"type": "PATH", "value": "/usr/local/Ascend/cann-9.1.0"}
json.dump(d, open(p, "w"), indent=2)
PYEOF
cd gen && rm -rf build_out && bash build.sh 2>&1 | tail -5
RUNFILE=$(ls build_out/*.run | head -1)
$RUNFILE --quiet --install-path=/usr/local/Ascend/cann-9.1.0/opp
find /usr/local/Ascend/cann-9.1.0/opp/vendors/customize -name "*.so" | head -3
EOS
```

成功标志:`opp/vendors/customize/op_api/lib/libcust_opapi.so` 存在。

## 2. vendor 加载白名单(关键,漏做则 aclnn 报 161001)

```bash
docker exec kvtq-repro bash -c \
  "sed -i 's/^load_priority=batch_invariant/load_priority=batch_invariant,customize/' \
   /usr/local/Ascend/cann-9.1.0/opp/vendors/config.ini; \
   grep load_priority /usr/local/Ascend/cann-9.1.0/opp/vendors/config.ini"
# 预期输出:load_priority=batch_invariant,customize
# (若原文件 load_priority 为空或不同,把 customize 追加到逗号分隔列表末尾)
```

## 3. torch 胶水扩展 + 精度单测(口径 D 第一层)

```bash
# 编译(有缓存,第二次秒过);要点:需链接 -lnnopbase(aclCreateTensor 在 libnnopbase.so)
docker exec kvtq-repro bash -c \
  "cd /root/kvtq_integration/code/integration/torch_ext && python3 build_ext.py"
# 预期:BUILD OK

# 与任务附件 golden.py 逐位比对:bits 2/3/4 x N=1~1024 共 5 组,全部 bit-exact
docker exec kvtq-repro bash -c \
  "cd /root/kvtq_integration/code/integration/torch_ext && python3 test_glue.py"
# 预期:5 PASS
```

## 4. 安装 vLLM 插件(EngineCore 子进程内生效的关键)

```bash
docker exec -i kvtq-repro bash /root/kvtq_integration/code/bench/install_plugin.sh
# 预期打印:kv_cache_turbo_quant_shadow -> kvtq_vllm_plugin:register
```

说明:vllm-ascend 的 EngineCore 是 spawn 子进程,不继承父进程 monkeypatch,必须走 `vllm.general_plugins` entry point;开关通过启动 shell 的环境变量传入:

| 环境变量 | 作用 |
|---|---|
| `VLLM_ASCEND_KVTQ=1` | 影子模式:量化+丢弃,cache 仍 BF16,输出与基线一致 |
| `VLLM_ASCEND_KVTQ_STORE=1` | store 模式:KV 真实压缩落盘(66 B/向量) |
| `VLLM_ASCEND_KVTQ_BITS=2/3/4` | mse_bits,默认 3 |
| `VLLM_ASCEND_KVTQ_DEBUG=1` | 打印重构误差 |

## 5. 冒烟验证(口径:接入可用性)

```bash
docker exec kvtq-repro bash -c \
  "cd /root/kvtq_integration/code/integration && python3 run_shadow.py --model /home/models/Qwen3-1.7B --enforce-eager"
```

预期:3 条 prompt 正常生成,日志有 `[KVTQ]` 量化调用计数;同参数不设 `VLLM_ASCEND_KVTQ` 再跑一次,输出逐 token 一致。

## 6. 口径 A:写路径开销(影子模式 A/B,各 ~15 min)

```bash
# 基线
docker exec -d kvtq-repro bash /root/kvtq_integration/code/bench/serve_kvtq.sh 0
docker exec kvtq-repro bash -c "until curl -s -m2 127.0.0.1:8377/v1/models >/dev/null; do sleep 5; done; echo READY"
docker exec kvtq-repro bash /root/kvtq_integration/code/bench/bench_grid.sh baseline
bash code/bench/stop_serve.sh   # 停服务并确认 npu-smi 卡 2/3 无进程、8377 释放

# 影子量化
docker exec -d kvtq-repro bash /root/kvtq_integration/code/bench/serve_kvtq.sh 1
docker exec kvtq-repro bash -c "until curl -s -m2 127.0.0.1:8377/v1/models >/dev/null; do sleep 5; done; echo READY"
docker exec kvtq-repro bash /root/kvtq_integration/code/bench/bench_grid.sh shadow
bash code/bench/stop_serve.sh

# 聚合对比(把新 JSON 拷到 results/ 后运行)
python3 results/aggregate.py
```

bench 参数:random 数据集,input 16384/32768,output 1024,并发 1/4/16/32,`--ignore-eos`;两组服务除 `VLLM_ASCEND_KVTQ=0/1` 外完全相同(TP=2、max_model_len 40960、gpu_mem_util 0.92、关 prefix caching)。

**预期结果**(本次实测,`results/` 下 16 个 JSON 与 grid_*.log 为原始数据):

| input | 并发 | dTTFT | dTPOT | d总吞吐 |
|---|---|---|---|---|
| 16384 | 1  | +5.1% | +19.1% | -15.3% |
| 16384 | 4  | +4.1% | +25.6% | -18.3% |
| 16384 | 16 | +4.2% | +11.2% | -9.3%  |
| 16384 | 32 | +4.0% | +7.7%  | -7.8%  |
| 32768 | 1  | +3.0% | +20.4% | -14.9% |
| 32768 | 4  | +3.0% | +21.4% | -14.6% |
| 32768 | 16 | +0.1% | +6.9%  | -5.4%  |
| 32768 | 32 | +5.8% | +6.5%  | -5.7%  |

规律:TTFT 开销稳定 +3~6%(prefill 逐 chunk 量化);TPOT 开销随并发从 ~+20% 收敛到 +5~8%(被 decode 计算摊薄)。

## 7. 口径 B/C:KV 容量收益(store 模式真实落盘)

```bash
# BF16 基线(eager,与 store 同执行方式保证可比)
docker exec -d kvtq-repro bash /root/kvtq_integration/code/bench/serve_base_eager.sh \
  /home/models/Qwen3-30B-A3B qwen3-30b 2 2,3 8377 40960
docker exec kvtq-repro grep -a "KV cache size" /root/kvtq_integration/bench/serve_base_eager_qwen3-30b.log
bash code/bench/stop_serve.sh

# store 模式(自动带 VLLM_ASCEND_KVTQ_STORE=1;必须 eager + 关 prefix caching,脚本已带)
docker exec -d kvtq-repro bash /root/kvtq_integration/code/bench/serve_tq.sh \
  /home/models/Qwen3-30B-A3B qwen3-30b 2 2,3 8377 40960 1
docker exec kvtq-repro grep -a "KV cache size" /root/kvtq_integration/bench/serve_tq_qwen3-30b.log
# 发一条 chat 请求确认生成通顺,然后:
bash code/bench/stop_serve.sh
```

**预期**(两侧 Available KV cache memory 均 23.72 GiB/worker):

| 模式 | GPU KV cache size | 40960 上下文最大并发 |
|---|---|---|
| BF16 eager | 518,144 tokens | 12.65x |
| store(4-bit packed, 66 B/向量) | 2,009,728 tokens | 49.07x |

容量比 **3.88x**,与 256B->66B 打包压缩率一致;等容量换算省 ~19 GiB/卡(25.69 -> 6.6 GiB)。基线在 32k/c32 场景出现 3 波排队(TTFT 210s)、c48 达 367s,见 `results/cells_base_eager.log`;同场景 store 模式 KV 低水位全驻留。

注意:store 读路径是 torch 暂存实现(decode ~10s/step),**不能**用于吞吐对照(`results/cells_store.log` 中请求超时即此原因);容量收益看启动日志即可判定。

## 8. 口径 D 第二层:量化-重构精度

```bash
docker exec kvtq-repro bash -c \
  "cd /root/kvtq_integration/code/integration && python3 test_store_roundtrip.py"
# 预期 PASS;4-bit 重构 rel err 均值 ~0.096 / 最大 ~0.18(mse_bits=3 时逐 token 最大 0.19~0.27,符合 TurboQuant 预期量级)
```

## 9. 已知坑清单

1. `opp/vendors/config.ini` 的 `load_priority` 必须含 `customize`,否则 aclnn 161001(operator package not installed / SoC verification failed)。
2. 胶水链接需 `-lnnopbase`(aclCreateTensor/aclDestroyTensor 不在 libascendcl)。
3. 多返回值算子的 torch 实现必须返回 `std::tuple`,不能返回 `std::vector<at::Tensor>`。
4. torch cpp_extension 仅含 TORCH_LIBRARY 时要加空 `PYBIND11_MODULE`。
5. EngineCore spawn 子进程不继承父进程 monkeypatch,必须 `vllm.general_plugins` entry point;env 标志在启动 shell export。
6. `import vllm_ascend.attention.attention_v1` 前先 `import vllm_ascend.ops`,否则循环导入。
7. store 模式限制:`--enforce-eager`、`--no-enable-prefix-caching`;prefill 续写 chunk 走批量 cache 解压。
8. 停服务必须连 `VLLM::Worker_TP` 残留进程一起清(`code/bench/stop_serve.sh`),并用 npu-smi + 端口 + 进程三项验证。