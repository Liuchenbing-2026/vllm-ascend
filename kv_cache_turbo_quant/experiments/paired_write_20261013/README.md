# 合并K/V量化与双cache写入候选（验证排队中）

| 项目 | 状态 |
|---|---|
| 分支/基础 | perf/tq-paired-write-20261013 / 4a88742bfc663b9863dd1703201d6c2326c9e91e |
| 改动 | 仅bits4把K/V在head维拼接成一个原Ascend C量化调用，Triton一次写两份原66B/cache；norm、centroid、矩阵、量化bits不变 |
| 已验证 | 新module Ruff/format和Python编译、git diff检查通过；单cache前阶段78项通过 |
| 待验证 | paired90项NPU全cache位一致与原两个量化调用对照、整网greedy及TTFT/TPOT/吞吐；不能说paired已通过或有收益 |
| 后台 | 等/write_ab_20261013/host.exit=0后再占同physical2/5，无交叉压测 |

未改vLLM2cf0a6915ce544dc493a0990f2ea38d81601128a和Ascend96df623103f921df1a4488170d106f36689acb59；沿用已验证镜像vllm-ascend:dspark-a2-028，image ID sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14，RepoDigests=[]。原容器tq-opt-20261011-model、910B4-1/driver26.0.rc1、CANN9.1.0、torch2.10.0+cpu、torch_npu2.10.0.post4、triton-ascend3.2.2。完整基础镜像之外的源码构建、模型权重与依赖准备沿用[前阶段可复现Markdown](../prefill_store_20261013/README.md)，同24GiB KV/card、真实BF16 Qwen3-30B-A3B、TP2、eager、prefix/async关闭、maxseq512/budget4096/len8192。

后台脚本/scripts/host.py等前一轮完整退出，保存等待状态，资源空闲后才执行容器/scripts/run_write_ab.py；该入口先执行90项paired精度门禁，任何失败立即保留日志并停止自己的流水线，不进入性能轮。正式对照与前阶段同8000/1C256、20/20C16、2000/200C60，各模式1预热+3正式，模型greedy440 IDs独立核对。实际启动命令和全部环境变量保存在[scripts/serve.sh](scripts/serve.sh)。

```bash
# 宿主：恢复依赖、候选和scripts到对应挂载后启动；已有launch时不得重复启动
nohup python3 -I -u /data02/tq-opt-20261011/paired_write_20261013/scripts/host.py > /data02/tq-opt-20261011/paired_write_20261013/logs/host.log 2>&1 < /dev/null &
# 独立NPU精度（需空闲卡；当前排队，不代表已经执行）
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,5 VLLM_ASCEND_KVTQ_BITS=4 TORCH_EXTENSIONS_DIR=/ws/torch-extensions
export PYTHONPATH=/ws/paired_write_20261013/candidate_write/integration:/ws/source/vllm:/ws/source/vllm-ascend
/ws/.venv/bin/python /ws/paired_write_20261013/scripts/verify_pair.py
```

90项是72组完整K/V字节与CPU独立pack/scatter oracle、18组真实原两个Ascend C量化调用严格对照；包括1/2/4 heads、尾块、随机/反序/连续slot、未修改槽位及任意BF16位模式。未放宽精度、golden或输出token数量。bits2/3仍走原分开写入；本轮未新建Meta/图路径，本来store要求eager。候选额外张量只存当前step拼接K/V和量化结果，不保持BF16历史cache。

OPP/框架三个kernel SO仍严格哈希一致；C++glue可能因不同等价源路径重新构建，当前与历史SHA分开记录，C++源码字节固定51fdaeb74f26aa2c9aece180955651dab6e858de943cee9ab105a8022066d6b5，paired精度必须重跑，不能以旧78项替代新90项。未从干净镜像从头执行此候选、未声明CI/语义质量或整网收益通过。
