#!/bin/bash
# Mooncake KV Pool 前置检查（只读，不修改任何配置）
#
# 存在意义：Mooncake 在 A2 上的 KV 传输走 NPU RoCE 网口，而卡间 HCCS 互联健康
# 并不代表 RoCE 可用。这两条路径独立，TP 推理跑得好 != KV 池能工作。
# 本脚本在部署前判定 RoCE 路径是否具备条件，避免把数小时浪费在“引擎起得来、
# 池写得进、读全失败”的软失败上。

FAIL=0
warn() { echo "  [WARN] $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=1; }
ok()   { echo "  [ OK ] $*"; }

echo "=== 1. 软件版本 ==="
python3 -c "import vllm_ascend; print('  vllm-ascend', getattr(vllm_ascend,'__version__','?'))" 2>/dev/null || bad "vllm_ascend 不可导入"
mc=$(pip list 2>/dev/null | grep -i 'mooncake-transfer-engine-npu' | awk '{print $2}')
if [ -z "$mc" ]; then
  bad "未安装 mooncake-transfer-engine-npu（注意 -npu 后缀，GPU 版没有 ascend 协议）"
else
  ok "mooncake-transfer-engine-npu $mc"
  case "$mc" in
    0.3.1[2-9]*|0.3.[2-9]*|0.[4-9]*) ;;
    *) warn "建议 >= 0.3.12.post1（含 SSD auto-recovery / INVALID_KEY 竞态修复）" ;;
  esac
fi
g=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')
[ -n "$g" ] && ok "glibc $g（wheel 需 >= 2.35）"
python3 -c "from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector" 2>/dev/null \
  && ok "AscendStoreConnector 可导入" || bad "AscendStoreConnector 导入失败，检查 vllm-ascend 版本"

echo
echo "=== 2. 卡间计算互联（HCCS）——用于 TP，不用于 KV 池 ==="
topo=$(npu-smi info -t topo 2>/dev/null)
if echo "$topo" | grep -q HCCS; then
  n=$(echo "$topo" | grep -c HCCS)
  ok "检测到 HCCS 互联（$n 行）"
  echo "$topo" | grep -qE '\bPIX\b|\bPXB\b|\bSYS\b' && warn "存在非 HCCS 降级路径，TP 性能可能受影响"
else
  warn "未检测到 HCCS，卡间走 PCIe，TP 通信会慢"
fi

echo
echo "=== 3. NPU RoCE 网口——Mooncake KV 传输的实际路径（A2 关键项）==="
DEVS=${MCSSD_CHECK_DEVICES:-$(ls /dev/davinci[0-9]* 2>/dev/null | grep -oE '[0-9]+$' | sort -n | tr '\n' ' ')}
for d in $DEVS; do
  ip=$(hccn_tool -i $d -ip -g 2>/dev/null | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1)
  lk=$(hccn_tool -i $d -link -g 2>/dev/null | grep -oE 'UP|DOWN')
  hl=$(hccn_tool -i $d -net_health -g 2>/dev/null | grep -oE 'Success|Init|Fail[a-z]*|Warning')
  nd=$(hccn_tool -i $d -netdetect -g 2>/dev/null | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}')
  op=$(hccn_tool -i $d -optical -g 2>/dev/null | grep -oE 'present|absent' | head -1)
  peer=$(hccn_tool -i $d -lldp -g 2>/dev/null | grep -A1 'Management Address' | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1)
  printf "  dev%-2s ip=%-15s link=%-4s health=%-8s netdetect=%-15s optical=%-7s lldp_peer=%s\n" \
    "$d" "${ip:-NONE}" "${lk:-?}" "${hl:-?}" "${nd:-0.0.0.0}" "${op:-?}" "${peer:-NONE}"
done

echo
echo "  -- 判定 --"
h_ok=0; h_init=0
for d in $DEVS; do
  hl=$(hccn_tool -i $d -net_health -g 2>/dev/null | grep -oE 'Success|Init')
  [ "$hl" = "Success" ] && h_ok=$((h_ok+1))
  [ "$hl" = "Init" ] && h_init=$((h_init+1))
done
[ $h_init -gt 0 ] && warn "$h_init 张卡 net_health=Init。多因 netdetect 未配（探测地址为 0.0.0.0），不一定代表链路故障"
[ $h_ok -gt 0 ] && ok "$h_ok 张卡 net_health=Success"

# 关键判定：机内两卡能否互通。这是单机 KV 池的硬性前提。
set -- $DEVS
d1=$1; d2=$2
if [ -n "$d2" ]; then
  ip2=$(hccn_tool -i $d2 -ip -g 2>/dev/null | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1)
  net1=$(hccn_tool -i $d1 -ip -g 2>/dev/null | grep -oE '([0-9]{1,3}\.){3}' | head -1)
  net2=$(echo "$ip2" | grep -oE '([0-9]{1,3}\.){3}')
  echo "  机内互通测试: dev$d1 -> dev$d2 ($ip2)"
  if [ "$net1" != "$net2" ]; then
    warn "两卡不在同一网段（$net1 x vs $net2 x），机内直连不可达；这是双机直连拓扑的典型特征"
  fi
  r=$(hccn_tool -i $d1 -ping -g address $ip2 2>/dev/null | grep -oE '[0-9]+% packet loss')
  if echo "$r" | grep -q '^0%'; then
    ok "机内卡间 RoCE 可达（$r）—— 单机 KV 池具备条件"
  else
    bad "机内卡间 RoCE 不可达（${r:-无响应}）—— 单机 Mooncake KV 池无法工作"
    echo "         A2 的 Mooncake 传输只走 RoCE 网口（HCCS 直传是 A3 的能力），"
    echo "         机内无 RoCE 通路时，KV 池会表现为“写成功、读全部失败、回退重算”。"
    echo "         若 lldp_peer 指向另一台机器，说明是双机直连布线，"
    echo "         此时应改用跨机 PD 分离而非单机 KV 池。"
  fi
fi

echo
echo "=== 4. 运行时配置 ==="
[ "$PYTHONHASHSEED" = "0" ] && ok "PYTHONHASHSEED=0" || bad "PYTHONHASHSEED 未设为 0（不一致会导致池永不命中且无报错）"
if [ -n "$MCSSD_SSD_PATH" ]; then
  if [ -d "$MCSSD_SSD_PATH" ] && [ -w "$MCSSD_SSD_PATH" ]; then
    ok "SSD 路径可写: $MCSSD_SSD_PATH ($(df -h "$MCSSD_SSD_PATH" 2>/dev/null | tail -1 | awk '{print $4}') 可用)"
  else
    bad "SSD 路径不存在或不可写: $MCSSD_SSD_PATH"
  fi
fi

echo
if [ $FAIL -eq 0 ]; then
  echo "===> 前置检查通过"
else
  echo "===> 前置检查未通过，修复后再部署（强行部署会得到“监控全绿但性能暴跌”的软失败）"
fi
exit $FAIL
