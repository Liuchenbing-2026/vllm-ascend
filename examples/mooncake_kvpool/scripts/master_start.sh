#!/bin/bash
# 启动 mooncake_master。用法: master_start.sh dram|ssd
set -e
MODE=${1:?usage: master_start.sh dram|ssd}
MCSSD_ROOT=${MCSSD_ROOT:-/data1/mcssd}
mkdir -p "$MCSSD_ROOT/logs"
LOG=$MCSSD_ROOT/logs/master_${MODE}.log

# --default_kv_lease_ttl 必须 > ASCEND_CONNECT_TIMEOUT / ASCEND_TRANSFER_TIMEOUT（默认各 10000ms）
BASE="--port 50088 \
  --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.1 \
  --default_kv_lease_ttl 11000 \
  --metrics_port 9003"

case $MODE in
  dram)
    exec mooncake_master $BASE >>"$LOG" 2>&1
    ;;
  ssd)
    # --client_ttl 默认 10s，SSD 场景过短会触发 SEGMENT_NOT_FOUND
    # --allocation_strategy=ssd_free_ratio_first 避免流量打到 SSD 已满的节点
    # 严禁同时设置 --root_fs_dir（legacy DFS 路径，与 SSD offload 冲突）
    exec mooncake_master $BASE \
      --enable_offload=true \
      --client_ttl=120 \
      --allocation_strategy=ssd_free_ratio_first \
      --offloading_queue_limit=500000 \
      --offload_cap_ratio=0.8 \
      >>"$LOG" 2>&1
    ;;
  *)
    echo "usage: master_start.sh dram|ssd" >&2; exit 1 ;;
esac
