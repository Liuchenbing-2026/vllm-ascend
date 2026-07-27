# Kimi-K3

## 1 Introduction

Kimi K3 is a native multimodal Mixture-of-Experts (MoE) model. Its language backbone combines Kimi Delta Attention (KDA) with periodic Gated Multi-head Latent Attention (MLA), and uses Stable LatentMoE for expert computation. The model also integrates a MoonViT vision encoder and supports text, image understanding, reasoning, and tool calling.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, multi-node data parallel and PD separation deployment, functional verification, and accuracy and performance evaluation.

This document is validated and written based on **vLLM-Ascend 0.23.0**. The current model (Kimi-K3) is first supported in this version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

Download the [Eco-Tech/Kimi-K3-w4a8](https://www.modelscope.cn/models/Eco-Tech/Kimi-K3-w4a8) ModelSlim W4A8 quantized weight from ModelScope. Deploying this weight requires at least 4 Atlas 800 A3 (64G × 16) nodes or 8 Atlas 800 A2 (64G × 8) nodes.

The local implementation supports Kimi K3 ModelSlim quantization through `--quantization ascend`. For a checkpoint that already contains a `compressed-tensors` quantization configuration, omit `--quantization ascend` and let vLLM discover the quantization method from the checkpoint.

The checkpoint directory must contain the model configuration, tokenizer, image processor, and model weight files required by the published Kimi K3 package.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Kimi K3 is validated on Atlas 800 A3 (64G × 16) and Atlas 800 A2 (64G × 8). Select the image that matches the target hardware and host operating system, and start it on each node, referring to [using docker](../../installation.md#set-up-using-docker).

| Target environment | Image |
| --- | --- |
| A3 Ubuntu | `quay.io/ascend/vllm-ascend:kimi-k3-a3` |
| A3 openEuler | `quay.io/ascend/vllm-ascend:kimi-k3-a3-openeuler` |
| A2 Ubuntu | [vLLM-Ascend-Kimi-k3-a2-ubuntu.tar.gz](http://xql-model.obs.cn-east-3.myhuaweicloud.com/images/vllm-atlas-day0/vLLM-Ascend-Kimi-k3-a2-ubuntu.tar.gz) |

Run the following command on each node:

```{code-block} bash
# Ubuntu:
export IMAGE=quay.io/ascend/vllm-ascend:kimi-k3-a3
# openEuler:
# export IMAGE=quay.io/ascend/vllm-ascend:kimi-k3-a3-openeuler
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --privileged=true \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

For Atlas 800 A2, download and load the validated image archive on every node:

```{code-block} bash
wget -O vLLM-Ascend-Kimi-k3-a2-ubuntu.tar.gz \
    http://xql-model.obs.cn-east-3.myhuaweicloud.com/images/vllm-atlas-day0/vLLM-Ascend-Kimi-k3-a2-ubuntu.tar.gz

docker load -i vLLM-Ascend-Kimi-k3-a2-ubuntu.tar.gz

export IMAGE=quay.io/ascend/vllm-ascend:vLLM-Ascend-Kimi-k3-a2-ubuntu
export CONTAINER=kimi3_a2_ubuntu

docker run -itd --privileged \
    --name $CONTAINER \
    --net=host \
    --shm-size 500g \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /home:/home \
    -v /data1:/data1 \
    -v /data2:/data2 \
    -v /data3:/data3 \
    -v /opt:/opt \
    -v /mnt:/mnt \
    --entrypoint /bin/bash \
    $IMAGE -lc 'sleep infinity'
```

The A2 image contains the validated vLLM and vLLM-Ascend build for the 896-expert model. Do not replace the image's vLLM-Ascend installation with an older source revision that can select the unsupported MC2 path for this expert count.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

Kimi K3 configuration, multimodal processing, reasoning parsing, and tool parsing are registered by vLLM-Ascend. Use a vLLM and vLLM-Ascend source revision that matches the validated version in this document.

## 5 Online Service Deployment

### 5.1 Four-Node P/D Co-Located Deployment

The validated P/D co-located deployment runs Prefill and Decode in the same service and uses four Atlas 800 A3 (64G × 16) nodes. vLLM data parallelism spans the four nodes, each node runs one DP rank, and tensor parallelism uses all 16 NPUs in the node. The resulting topology is DP4/TP16/EP64.

Before starting the service:

- Replace the model path, local IP address, network interface, service port, and DP RPC port with values from the target environment.
- `NIC_NAME` must be the interface that owns `LOCAL_IP`.
- Start Node 0 first. The `NODE0_IP` configured on Nodes 1 through 3 must equal `LOCAL_IP` on Node 0.
- Assign `--data-parallel-start-rank` values `1`, `2`, and `3` to Nodes 1, 2, and 3 respectively.

:::::{tab-set}
:sync-group: mixed-deployment

::::{tab-item} Node 0
:sync: node-0

```shell
# Values that must be adapted to the target environment.
export MODEL_PATH=<KIMI_K3_MODEL_PATH>
export LOCAL_IP=<NODE0_LOCAL_IP>
export NIC_NAME=<NODE0_NIC_NAME>
export PORT=<SERVICE_PORT>
export RPC_PORT=<DP_RPC_PORT>

export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=800
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve $MODEL_PATH \
    --served-model-name kimi-k3 \
    --port $PORT \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 16 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-address $LOCAL_IP \
    --data-parallel-rpc-port $RPC_PORT \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 131027 \
    --max-num-batched-tokens 24576 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}' \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true}' \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt '{"vision_chunk": 40}' \
    --enable-auto-tool-choice \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3
```

::::
::::{tab-item} Nodes 1-3
:sync: worker-nodes

Run this command on every worker node. Set `LOCAL_IP` and `NIC_NAME` to the current node and set `DP_START_RANK` to `1`, `2`, or `3`.

```shell
# Values that must be adapted to the target environment.
export MODEL_PATH=<KIMI_K3_MODEL_PATH>
export LOCAL_IP=<WORKER_LOCAL_IP>
export NODE0_IP=<NODE0_LOCAL_IP>
export NIC_NAME=<WORKER_NIC_NAME>
export PORT=<SERVICE_PORT>
export RPC_PORT=<DP_RPC_PORT>
export DP_START_RANK=<1_OR_2_OR_3>

export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=800
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve $MODEL_PATH \
    --headless \
    --served-model-name kimi-k3 \
    --port $PORT \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 16 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank $DP_START_RANK \
    --data-parallel-address $NODE0_IP \
    --data-parallel-rpc-port $RPC_PORT \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 131027 \
    --max-num-batched-tokens 24576 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}' \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true}' \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt '{"vision_chunk": 40}' \
    --enable-auto-tool-choice \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3
```

::::
:::::

The following values differ between the master and worker nodes:

| Setting | Node 0 | Nodes 1-3 | Description |
| --- | --- | --- | --- |
| `LOCAL_IP` | Node 0 IP | Current worker IP | Each node uses its own IP address. |
| `NODE0_IP` | Not required | Node 0 IP | Workers use this address to join the DP group. |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `7200` | Not set | Only the master waits for all engines to become ready. |
| `--headless` | Omitted | Enabled | Workers do not expose the API endpoint. |
| `--data-parallel-address` | `$LOCAL_IP` | `$NODE0_IP` | Always resolves to Node 0. |
| `--data-parallel-start-rank` | `0` by default | `1`, `2`, or `3` | Every node must own a unique DP rank. |

Key deployment parameters:

| Parameter | Description |
| --- | --- |
| `--tensor-parallel-size 16` | Uses all 16 NPUs in one A3 node for tensor parallelism. |
| `--data-parallel-size 4` | Creates four global DP ranks across four nodes. |
| `--data-parallel-size-local 1` | Runs one DP rank on the current node. |
| `--data-parallel-start-rank` | Selects the global starting DP rank for a worker node. |
| `--data-parallel-rpc-port` | Must be identical and reachable on every node. |
| `--enable-expert-parallel` | Enables expert parallelism for the MoE layers. |
| `--max-model-len 131027` | Sets the maximum combined input and output length. |
| `--max-num-seqs 16` | Sets the maximum active sequences for each DP group. |
| `--max-num-batched-tokens 24576` | Controls the scheduler token budget. |
| `--enable-prefix-caching` | Enables automatic prefix caching. |
| `--compilation-config` | Uses `FULL_DECODE_ONLY` ACL Graph replay. |
| `--additional-config` | Enables Ascend CPU binding and FlashComm1. |
| `HCCL_IF_IP` and socket interface variables | Bind HCCL, Gloo, and TP communication to the selected interface. |

If a worker exits immediately, confirm that Node 0 is already running, `--data-parallel-address` resolves to Node 0, and every worker uses a unique `--data-parallel-start-rank`.

Verify the service through Node 0:

```shell
curl http://<NODE0_LOCAL_IP>:<SERVICE_PORT>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi-k3",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

The service should return HTTP 200 and a `choices` field containing generated text.

### 5.2 Eight-Node A2 P/D Co-Located Deployment

The validated Atlas 800 A2 P/D co-located deployment runs Prefill and Decode in the same service and uses eight nodes with eight 64 GB NPUs per node. Each node runs one data-parallel rank and uses all eight local NPUs for tensor parallelism. Expert parallelism spans all 64 NPUs, resulting in a DP8/TP8/EP64 topology.

This deployment uses the A2 image archive from [Docker Image Installation](#41-docker-image-installation). Keep the vLLM-Ascend installation included in that image because it contains the validated AllGather fallback for the 896-expert Kimi K3 model on A2.

The validated A2 image reports vLLM `0.23.0` and vLLM-Ascend commit `0c33eb3fd146030fbd1d4a0f65bec6cf114a3ab2`.

Before starting the service:

- Make the model directory visible at the same path in all eight containers.
- Assign a unique data-parallel rank from `0` through `7` to each node.
- Set `COMM_IP` to the communication address on the current node.
- Set `HCCL_IFNAME` to the interface that owns `COMM_IP`.
- Set `MASTER_IP` to the Rank 0 communication address on every node.
- Use the same service port and data-parallel RPC port on every node.
- Start all eight ranks in the same deployment window. Rank 0 hosts the data-parallel coordinator.

Create `/opt/kimi-k3/node.env` in every `kimi3_a2_ubuntu` container. Replace the rank, address, interface, and model path for the current node:

```shell
RANK="<CURRENT_NODE_DP_RANK>"
COMM_IP="<CURRENT_NODE_COMMUNICATION_IP>"
HCCL_IFNAME="<CURRENT_NODE_COMMUNICATION_INTERFACE>"
MASTER_IP="<RANK0_COMMUNICATION_IP>"
NO_PROXY_EXTRA="<MANAGEMENT_AND_COMMUNICATION_NETWORKS>"

VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend
MODEL_PATH="<KIMI_K3_MODEL_PATH>"
SERVED_MODEL_NAME=kimi-k3

DP_SIZE=8
TP_SIZE=8
DP_RPC_PORT=39830
SERVICE_PORT=8730

MAX_MODEL_LEN=262144
GPU_MEMORY_UTILIZATION=0.90
SAFETENSORS_PREFETCH_NUM_THREADS=16
```

Create `/opt/kimi-k3/start_vllm.sh` in every container:

```shell
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/node.env"

VLLM_ASCEND_DIR=${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}
CUSTOM_OP_ENV="${VLLM_ASCEND_DIR}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

if [[ ! -d "${VLLM_ASCEND_DIR}/.git" ]]; then
    echo "Missing vLLM-Ascend repository: ${VLLM_ASCEND_DIR}" >&2
    exit 1
fi
if [[ ! -f "${CUSTOM_OP_ENV}" ]]; then
    echo "Missing Kimi-K3 custom operator environment: ${CUSTOM_OP_ENV}" >&2
    exit 1
fi

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "${CUSTOM_OP_ENV}"
set -u

unset ftp_proxy http_proxy https_proxy FTP_PROXY HTTP_PROXY HTTPS_PROXY
unset HCCL_OP_EXPANSION_MODE
export no_proxy="localhost,local,.local,${NO_PROXY_EXTRA}"

export PYTHONPATH="${VLLM_ASCEND_DIR}:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PHYSICAL_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HOST_IP="${COMM_IP}"
export HCCL_IF_IP="${COMM_IP}"
export GLOO_SOCKET_IFNAME="${HCCL_IFNAME}"
export TP_SOCKET_IFNAME="${HCCL_IFNAME}"
export HCCL_SOCKET_IFNAME="${HCCL_IFNAME}"
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_BUFFSIZE=256
export HCCL_INTRA_ROCE_ENABLE=1

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=INFO
export TASK_QUEUE_ENABLE=1
export TIKTOKEN_CACHE_DIR=/root/.cache/tiktoken-k3

mkdir -p "${SCRIPT_DIR}/logs"
echo $$ > "${SCRIPT_DIR}/vllm.pid"

exec python -m vllm.entrypoints.cli.main serve \
    "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${SERVICE_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --tokenizer-mode kimi_k3 \
    --language-model-only \
    --mm-encoder-tp-mode data \
    --skip-mm-profiling \
    --limit-mm-per-prompt '{"vision_chunk":0}' \
    --data-parallel-size "${DP_SIZE}" \
    --data-parallel-rank "${RANK}" \
    --data-parallel-address "${MASTER_IP}" \
    --data-parallel-rpc-port "${DP_RPC_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --quantization ascend \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enable-prefix-caching \
    --safetensors-load-strategy prefetch \
    --safetensors-prefetch-num-threads "${SAFETENSORS_PREFETCH_NUM_THREADS}" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,32]}' \
    --enable-auto-tool-choice \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --additional-config '{"enable_flashcomm1":false,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true}'
```

Set permissions and verify the script on every node:

```shell
docker exec kimi3_a2_ubuntu chmod 600 /opt/kimi-k3/node.env
docker exec kimi3_a2_ubuntu chmod 755 /opt/kimi-k3/start_vllm.sh
docker exec kimi3_a2_ubuntu bash -n /opt/kimi-k3/start_vllm.sh
```

Start the service on all eight nodes:

```shell
docker exec -d kimi3_a2_ubuntu bash -lc \
    'exec /opt/kimi-k3/start_vllm.sh > /opt/kimi-k3/logs/service.log 2>&1'
```

The first startup loads the model weights, compiles the graph, profiles the KV cache, and captures five decode graphs. Monitor the service log until it contains `Application startup complete`:

```shell
docker exec kimi3_a2_ubuntu tail -n 200 -f /opt/kimi-k3/logs/service.log
```

Verify every node locally. `--noproxy '*'` prevents a host proxy from intercepting the loopback request:

```shell
curl --noproxy '*' http://127.0.0.1:8730/v1/models
```

Every node should return HTTP 200 and report `kimi-k3` with `max_model_len` set to `262144`.

Run a text-generation smoke test against any data-parallel endpoint:

```shell
curl --noproxy '*' http://127.0.0.1:8730/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi-k3",
        "messages": [{
            "role": "user",
            "content": "Hello, introduce yourself in one sentence."
        }],
        "temperature": 0,
        "max_tokens": 128
    }'
```

The service should return HTTP 200 with readable generated text. This request is a functional smoke test and does not replace dataset-based accuracy evaluation.

Key A2 deployment parameters:

| Parameter | Description |
| --- | --- |
| `--data-parallel-size 8` | Creates eight global data-parallel ranks across eight A2 nodes. |
| `--data-parallel-rank` | Assigns the current node a unique rank from `0` through `7`. |
| `--tensor-parallel-size 8` | Uses all eight NPUs on the current A2 node. |
| `--enable-expert-parallel` | Distributes 896 experts across the 64-NPU EP group. |
| `--max-model-len 262144` | Enables a 256K-token maximum context length. |
| `--compilation-config` | Captures decode graphs for batch sizes `1`, `2`, `4`, `8`, and `32`. |
| `enable_flashcomm1=false` | Uses the validated non-FlashComm1 A2 path. |
| `enable_npugraph_ex=true` | Enables the validated Ascend graph compilation path. |
| `HCCL_IF_IP` and socket interface variables | Bind HCCL, Gloo, and TP communication to the communication network. |

### 5.3 Sixteen-Node PD Separation Deployment

The validated PD separation topology uses 16 Atlas 800 A3 (64G × 16) nodes: eight Prefill nodes and eight Decode nodes. Both sides use DP8/TP16/PP1. Prefill nodes additionally use a memcache-backed KV pool.

Refer to [PD Disaggregation with Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general service workflow and [KV Pool](../../user_guide/feature_guide/kv_pool.md) for memcache pool concepts.

#### 5.3.1 Start the memcache MetaService

Start one MetaService instance before the Prefill engines:

```shell
export MMC_META_CONFIG_PATH=<PATH_TO_MMC_META_CONF>
python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

`mmc-meta.conf` configures MetaService and `mmc-local.conf` is loaded by every Prefill inference process. Run `pip show memcache_hybrid` to locate the installed package, copy the example files from `memcache_hybrid/config/`, and adapt them to the target environment.

#### 5.3.2 Create the engine templates

:::::{tab-set}
:sync-group: pd-templates

::::{tab-item} Prefill
:sync: prefill

```shell
KV_PORT=36000

unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

nic_name=<PREFILL_NIC_NAME>
local_ip=<PREFILL_LOCAL_IP>

export HCCL_IF_IP=${local_ip}
export GLOO_SOCKET_IFNAME=${nic_name}
export TP_SOCKET_IFNAME=${nic_name}
export HCCL_SOCKET_IFNAME=${nic_name}
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export VLLM_TORCH_PROFILER_DIR="./vllm_profile"
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export MMC_LOCAL_CONFIG_PATH=<PATH_TO_MMC_LOCAL_CONF>
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1

vllm serve <KIMI_K3_MODEL_PATH> \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name kimi-k3 \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --enforce-eager \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --mm-encoder-tp-mode data \
    --skip-mm-profiling \
    --safetensors_load_strategy prefetch \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --additional-config '{"recompute_scheduler_enable":false}' \
    --limit-mm-per-prompt '{"vision_chunk": 0}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": true}' \
    --kv-transfer-config \
    '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_producer",
      "kv_connector_extra_config": {
        "connectors": [
          {
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "'"$KV_PORT"'",
            "kv_connector_extra_config": {
              "prefill": {"dp_size": 8, "tp_size": 16},
              "decode": {"dp_size": 8, "tp_size": 16}
            }
          },
          {
            "kv_connector": "AscendStoreConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
              "backend": "memcache",
              "lookup_rpc_port": "0"
            }
          }
        ]
      }
    }'
```

::::
::::{tab-item} Decode
:sync: decode

```shell
KV_PORT=36200

unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

nic_name=<DECODE_NIC_NAME>
local_ip=<DECODE_LOCAL_IP>

export HCCL_IF_IP=${local_ip}
export GLOO_SOCKET_IFNAME=${nic_name}
export TP_SOCKET_IFNAME=${nic_name}
export HCCL_SOCKET_IFNAME=${nic_name}
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_TORCH_PROFILER_DIR="./vllm_profile"
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve <KIMI_K3_MODEL_PATH> \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name kimi-k3 \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --mm-encoder-tp-mode data \
    --skip-mm-profiling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --safetensors_load_strategy prefetch \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --additional-config '{"recompute_scheduler_enable":false}' \
    --limit-mm-per-prompt '{"vision_chunk":0}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": true}' \
    --kv-transfer-config \
    '{
      "kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "'"$KV_PORT"'",
      "kv_connector_extra_config": {
        "prefill": {"dp_size": 8, "tp_size": 16},
        "decode": {"dp_size": 8, "tp_size": 16}
      }
    }'
```

::::
:::::

#### 5.3.3 Start the engines

Deploy `launch_online_dp.py` and the corresponding engine template on every node. The following example starts one local DP rank in a DP8/TP16/PP1 group:

```shell
python launch_online_dp.py \
    --dp-size 8 \
    --tp-size 16 \
    --pp-size 1 \
    --dp-size-local 1 \
    --dp-rank-start <LOCAL_DP_RANK> \
    --dp-address <PD_MASTER_IP> \
    --dp-rpc-port <DP_RPC_PORT> \
    --vllm-start-port <VLLM_START_PORT>
```

Use ranks `0` through `7` for each eight-node side. Configure independent master addresses, RPC ports, and vLLM port ranges for the Prefill and Decode groups.

After the engines start, configure and start the load-balancing proxy as described in [PD Disaggregation with Mooncake](../features/pd_disaggregation_mooncake_multi_node.md#start-the-service).

Key PD settings:

| Setting | Value | Description |
| --- | --- | --- |
| Topology | 8P8D | Eight Prefill and eight Decode nodes. |
| `--dp-size` | `8` | Eight DP ranks on each side. |
| `--tp-size` | `16` | Uses all 16 NPUs in a node. |
| `--pp-size` | `1` | One pipeline stage per engine. |
| `--dp-size-local` | `1` | One DP rank per node. |
| `KV_PORT` | `36000` for P, `36200` for D | Separates producer and consumer KV traffic. |
| `MMC_LOCAL_CONFIG_PATH` | Prefill only | Connects the producer to the memcache KV pool. |
| `recompute_scheduler_enable` | `false` | Matches the validated Prefill and Decode configuration. |

## 6 Functional Verification

After the mixed or PD service is ready, send a multimodal request to the API endpoint:

```shell
curl http://<SERVICE_IP>:<SERVICE_PORT>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi-k3",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "<IMAGE_URL_OR_DATA_URL>"}
                },
                {
                    "type": "text",
                    "text": "Describe the image."
                }
            ]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

The service should return HTTP 200 and a `choices` field containing the image description. The current implementation supports image inputs but does not support video inputs.

## 7 Accuracy Evaluation

The following evaluation procedure was validated with the four-node DP4/TP16/EP64 service.

### 7.1 Prepare the Evaluation Environment

The validation toolkit requires Python 3.12 or later:

```shell
conda create -n kvv python=3.12
conda activate kvv
cd <KIMI_K3_EVALUATION_TOOLKIT>
pip install -e .
```

Prepare these datasets:

| Task | Description | Dataset |
| --- | --- | --- |
| MMMU Pro Vision | Ten-option multimodal visual question answering. | [MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| OCRBench | OCR and text-recognition evaluation. | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) |
| ToolCall/KVVV | Tool-calling evaluation. | `toolcall_benchmark/` in the evaluation toolkit |

All recorded evaluations use Thinking mode, preserve the reasoning output, set `reasoning_effort=max`, `temperature=1.0`, `top_p=1.0`, and run one epoch.

| Benchmark | Max output tokens | Max connections |
| --- | ---: | ---: |
| OCRBench | 8192 | 16 |
| MMMU Pro | 96000 | 16 |
| ToolCall/KVVV | 32768 | 16 |

### 7.2 Check the Service

```shell
conda activate kvv
cd <KIMI_K3_EVALUATION_TOOLKIT>

export KIMI_BASE_URL="http://<SERVICE_IP>:<SERVICE_PORT>/v1"
export KIMI_API_KEY="EMPTY"
export no_proxy="localhost,127.0.0.1,<SERVICE_IP>"
export NO_PROXY="$no_proxy"
export INSPECT_LOG_DIR=<INSPECT_LOG_DIRECTORY>

curl --noproxy <SERVICE_IP> \
    http://<SERVICE_IP>:<SERVICE_PORT>/v1/models

python verify_params_k3.py \
    --model "kimi-k3" \
    --think-mode "opensource" \
    --base-url "$KIMI_BASE_URL" \
    --api-key "$KIMI_API_KEY" \
    --all
```

All parameter checks must pass before running the benchmarks.

### 7.3 Run OCRBench

```shell
python eval.py ocrbench \
    --model "opensource/kimi-k3" \
    --max-tokens 8192 \
    --thinking \
    --think-mode "opensource" \
    --thinking-effort max \
    --stream \
    --max-connections 16 \
    --temperature 1.0 \
    --top-p 1.0
```

### 7.4 Run MMMU Pro

```shell
python eval.py mmmu \
    --model "opensource/kimi-k3" \
    --max-tokens 96000 \
    --thinking \
    --think-mode "opensource" \
    --thinking-effort max \
    --stream \
    --max-connections 16 \
    --temperature 1.0 \
    --top-p 1.0
```

### 7.5 Run ToolCall/KVVV

ToolCall uses the JSONL data in `toolcall_benchmark/`. For the long-context validation, restart the four-node service with the following master-node values:

| Parameter | Standard co-located deployment | ToolCall validation |
| --- | ---: | ---: |
| `--max-num-seqs` | 16 | 4 |
| `--max-model-len` | 131027 | 286720 |
| `--max-num-batched-tokens` | 24576 | 8192 |
| `--gpu-memory-utilization` | 0.9 | 0.97 |

All other options match Section 5.1. Worker nodes also use these values and retain `--headless` plus their unique DP ranks.

Run the benchmark:

```shell
python eval.py kimi_toolcall \
    --model "opensource/kimi-k3" \
    --max-tokens 32768 \
    --thinking \
    --think-mode "opensource" \
    --thinking-effort max \
    --stream \
    --max-connections 16 \
    --temperature 1.0 \
    --top-p 1.0 \
    --dataset toolcall_benchmark/toolcall_thinking_samples.jsonl
```

### 7.6 Inspect and Resume Evaluations

```shell
inspect view
inspect view start --log-dir <INSPECT_LOG_DIRECTORY>
inspect eval-retry logs/<EVALUATION_LOG>.eval
```

The evaluation toolkit retries rate-limit and network failures with exponential backoff. Non-network failures, including invalid model output, are recorded in the logs without retrying.

## 8 Performance Evaluation

The following performance procedure uses the four-node DP4/TP16/EP64 service and AISBench.

### 8.1 Install AISBench

Run AISBench in a separate environment or container on the master node so the load generator does not affect the serving processes:

```shell
git clone https://github.com/AISBench/benchmark
cd benchmark
pip3 install -e ./ --use-pep517
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
pip3 install -r requirements/hf_vl_dependency.txt
```

### 8.2 Performance Service Configuration

Change these values from the standard Section 5.1 deployment on all four nodes:

| Parameter | Standard deployment | Performance test |
| --- | ---: | ---: |
| `--max-model-len` | 131027 | 250000 |
| `--max-num-batched-tokens` | 24576 | 8192 |
| `--gpu-memory-utilization` | 0.9 | 0.95 |
| `torch_profiler_with_stack` | `false` | `true` |
| `--limit-mm-per-prompt` | `{"vision_chunk": 40}` | `{"vision_chunk": 2}` |

The master-node `vllm serve` command is:

```shell
vllm serve <KIMI_K3_MODEL_PATH> \
    --served-model-name kimi-k3 \
    --port <SERVICE_PORT> \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 16 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-address <NODE0_LOCAL_IP> \
    --data-parallel-rpc-port <DP_RPC_PORT> \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 250000 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": true}' \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true}' \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt '{"vision_chunk": 2}' \
    --enable-auto-tool-choice \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3
```

Worker nodes use the same performance values and the worker-specific arguments from Section 5.1.

### 8.3 Configure the Load Generator

Before running `aisbench_test.py`, create its dataset directory and configure the validation helper:

```shell
mkdir -p <DATASET_DIRECTORY>
```

```python
DATASET_PATH = "<DATASET_DIRECTORY>"
WORK_PATH = "<AISBENCH_BENCHMARK_DIRECTORY>"
MODEL_NAME = "kimi-k3"
MODEL_PATH = "<KIMI_K3_MODEL_PATH>"
HOST_IP = "<SERVICE_IP>"
HOST_PORT = "<SERVICE_PORT>"
DEFAULT_PERFORMANCE_TEST = "default_perf"
OUTPUT_DIR = "./outputs/default"

# Set the serving endpoints when collecting per-DP prefix-cache metrics.
# PD deployments should list every relevant endpoint.
POD_INFO = []
```

Disable proxies before the test:

```shell
env | grep -i proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

### 8.4 Run the Tests

8K input, 1K output, and no prefix-cache hit:

```shell
python3 aisbench_test.py \
    --input_len 8192 \
    --output_len 1024 \
    --data_num 16 \
    --concurrency 4 \
    --request_rate 0 \
    --repeat_rate 0 \
    --prefix_test
```

128K input, 1K output, and a 99% prefix-cache hit rate:

```shell
python3 aisbench_test.py \
    --input_len 131024 \
    --output_len 1024 \
    --data_num 16 \
    --concurrency 4 \
    --request_rate 0 \
    --dataset_type prefix_cache \
    --repeat_rate 0.99 \
    --prefix_test
```

`request_rate=0` sends requests as quickly as the configured concurrency permits. `repeat_rate=0.99` makes 99% of requests reuse the same prefix.

### 8.5 Enabled Optimizations

| Feature | Description |
| --- | --- |
| Chunked Prefill | Splits long prefill inputs into chunks to reduce per-step memory peaks. |
| Asynchronous scheduling | Decouples scheduling and execution. |
| Prefix Cache | Reuses KV state for repeated prefixes. |
| DP + TP + EP | Combines data, tensor, and expert parallelism for the MoE model. |
| ACL Graph | Uses `FULL_DECODE_ONLY` replay to reduce decode scheduling overhead. |
| KDA + MLA cache management | Manages the heterogeneous recurrent and KV states. |
| FlashComm1 | Enables communication optimization. |
| CPU Binding | Reduces cross-core scheduling overhead. |

## 9 Performance Tuning

Use the validated deployment values above as a baseline. Adjust `max-model-len`, `max-num-seqs`, `max-num-batched-tokens`, and `gpu-memory-utilization` together for the target workload.

Refer to the [performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md) and the [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for additional guidance.

## 10 FAQ

For common environment, installation, and general parameter issues, refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).

- **Q: Which multimodal inputs are supported by the current Kimi K3 implementation?**

  A: The current local processor accepts image inputs. Video inputs are not supported.

- **Q: Which server options are required for Kimi K3 reasoning and tool calling?**

  A: Configure `--enable-auto-tool-choice`, `--reasoning-parser kimi_k3`, and `--tool-call-parser kimi_k3` together.

- **Q: When should `--quantization ascend` be used?**

  A: Use it for ModelSlim native quantized weights. If the checkpoint declares a `compressed-tensors` configuration, omit the option and allow vLLM to load the checkpoint configuration.

- **Q: How should TP size be selected?**

  A: TP size must divide the checkpoint's attention-head count. It also affects KDA state layout and expert placement, so validate memory capacity and communication performance together.
