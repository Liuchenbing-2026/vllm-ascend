#!/bin/bash
# Start the GLM-5.3-Flash serving container.
#
# THE ONE THING THAT MATTERS: --ulimit memlock=-1
#   Docker's default max-locked-memory is 64 KB. --privileged does NOT raise it.
#   Without it, every pinned-host allocation past the caching allocator's fast path fails with
#     allocate_host_memory_slowpath ... aclrtMallocHostWithCfg, error code is 207001
#     rtsMallocHost execution failed, reason=driver error:out of memory
#   and the driver *also* prints
#     Not_Supported(EE1016): ... operation not permitted when a stream is capturing
#   which is a red herring -- the allocation failed on the rlimit, not on the capture mode.
#   Symptom: graph capture dies, and it only shows up once the block tables are big enough
#   (i.e. as soon as you turn prefix caching on).
set -e
IMG=${IMG:-quay.io/atlas-ci/vllm-atlas-temp:glm-5.3-flash-0902-1-910b-openeuler-33646519920-1-arm64-temp}
C=${C:-glm53s}
docker rm -f "$C" >/dev/null 2>&1 || true
docker run -itd --name "$C" \
  --net=host --ipc=host --privileged --shm-size=200g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65535 \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /data02:/data02 -v /data01:/data01 \
  --entrypoint /bin/bash "$IMG" -lc "sleep infinity"
sleep 3
docker exec "$C" bash -lc 'echo "memlock: $(ulimit -l)   (must be: unlimited)"'
# --prefix-caching-hash-algo xxhash needs this; the image does not ship it.
docker exec -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
            -e PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn "$C" \
  bash -lc 'pip install -q xxhash && python3 -c "import xxhash;print(\"xxhash\", xxhash.VERSION)"'
