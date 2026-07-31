#!/bin/bash
# Mooncake KV Pool 运行时环境变量（A2 / Atlas 800I A2）
# 用法: source configs/env_common.sh

MCSSD_ROOT=${MCSSD_ROOT:-/data1/mcssd}

export ASCEND_RT_VISIBLE_DEVICES=${MCSSD_DEVICES:-0,1}
export PYTHONHASHSEED=0            # 必须；全节点一致，否则 block hash 不同 -> 池永不命中且无报错
export ACL_OP_INIT_MODE=1

# --- 传输路径（按硬件三选一）---
# A2: 走 RoCE 网口。前提是机内卡间 RoCE 可达（先跑 scripts/preflight_check.sh）。
export HCCL_INTRA_ROCE_ENABLE=1
# A3: 走 HCCS fabric memory。注意 global_segment_size 与 offload buffer 均需 1GB 对齐，
#     且都计入 fabric mem 配额。依赖不满足时降级为 ASCEND_BUFFER_POOL=4:8。
# export ASCEND_ENABLE_USE_FABRIC_MEM=1
# A5 (UBOE):
# export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'

# --- 超时组 ---
# 约束: ASCEND_TRANSFER_TIMEOUT > RDMA 重传时间 x 7；
#       master 的 --default_kv_lease_ttl 必须大于下面两个值，否则报 LEASE_EXPIRED。
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000     # 约 500ms x Decode 总卡数
export ASCEND_TRANSFER_TIMEOUT=10000

# --- SSD 层配额（全部 per-rank，只能用环境变量，写进 mooncake.json 无效）---
# 按 “单盘容量 / 该盘上的 rank 数” 设置。默认 2TB 是 per-rank 值，不改会让 master
# 显示的总配额达到物理盘的数倍，监控失真、盘满才暴露。
MCSSD_SSD_QUOTA_GB=${MCSSD_SSD_QUOTA_GB:-60}
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((MCSSD_SSD_QUOTA_GB*1024*1024*1024))
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((MCSSD_SSD_QUOTA_GB*1024*1024*1024))
export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru   # 默认 none = 写满即失败
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824   # 1GB；A3 下必须 1GB 对齐

# 默认 bucket 后端（重启后 SSD 数据可恢复）。
# 切勿用 offset_allocator_storage_backend：初始化时会截断数据文件，重启即丢。
# export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
# export MOONCAKE_OFFLOAD_USE_URING=true   # 可选：切 io_uring（O_DIRECT，4KB 对齐）

export MCSSD_SSD_PATH=${MCSSD_SSD_PATH:-$MCSSD_ROOT/offload}
