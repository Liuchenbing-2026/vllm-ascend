#!/bin/bash
# /data01/nt-work/wait_and_launch2.sh <leg> <out> [deadline_minutes]
#
# Same as wait_and_launch.sh, with one change: the container is left *stopped*
# while waiting and started only when the cards actually free up. The previous
# run left it up for 80 minutes holding nothing, which on a shared machine is
# just noise for whoever reads `docker ps`.
#
# Cards 4-7 belong to whoever is on them. Poll, never pre-empt.
LEG="${1:?leg}"
OUT="${2:?out}"
DEADLINE_MIN="${3:-360}"
IDS="4 5 6 7"
MAX_USED_MB=8000
STAMP=/data01/nt-work/logs/${OUT}.waitstamp

mkdir -p /data01/nt-work/logs
: > "$STAMP"

used_mb() {
  npu-smi info 2>/dev/null | awk -v want="$1" '
    /^\| *[0-9]+ +[0-9A-Za-z-]+ +\|/ { id = $2 }
    id == want && match($0, /[0-9]+ *\/ *65536/) {
      s = substr($0, RSTART, RLENGTH); sub(/ *\/.*/, "", s); print s; exit
    }'
}

cards_ready() {
  local out i u
  out=$(npu-smi info 2>/dev/null) || return 1
  [ -n "$out" ] || return 1
  for i in $IDS; do
    printf '%s\n' "$out" | grep -q "No running processes found in NPU $i" || return 1
    u=$(used_mb "$i"); [ -n "$u" ] || return 1
    [ "$u" -le "$MAX_USED_MB" ] || return 1
  done
  return 0
}

end=$(( $(date +%s) + DEADLINE_MIN * 60 ))
n=0
while [ "$(date +%s)" -lt "$end" ]; do
  if cards_ready; then
    sleep 20                      # re-check: a neighbour may be mid-restart
    if cards_ready; then
      echo "$(date -Is) cards free after ${n} polls -- starting container" >> "$STAMP"
      docker start nt-dspark-mrv2 >> "$STAMP" 2>&1
      sleep 5
      # One last look: starting the container takes a few seconds, and that is
      # long enough for somebody else to claim the set.
      if ! cards_ready; then
        echo "$(date -Is) lost the race while starting -- stopping again, keep waiting" >> "$STAMP"
        docker stop nt-dspark-mrv2 >/dev/null 2>&1
      else
        echo "$(date -Is) launching $LEG" >> "$STAMP"
        rm -f "/data01/nt-work/logs/$OUT"
        docker exec -d nt-dspark-mrv2 bash -lc "setsid bash /nt/$LEG"
        echo "LAUNCHED" >> "$STAMP"
        exit 0
      fi
    fi
  fi
  n=$((n+1))
  [ $((n % 10)) -eq 1 ] && echo "$(date -Is) poll $n: $(for i in $IDS; do printf '%s=%s ' "$i" "$(used_mb "$i")"; done)" >> "$STAMP"
  sleep 60
done
echo "$(date -Is) DEADLINE after ${n} polls, never launched" >> "$STAMP"
exit 9
