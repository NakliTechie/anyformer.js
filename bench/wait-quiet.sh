#!/bin/sh
# Wait until the machine is quiet enough to benchmark: no rsync running and GPU utilization under 10% in each of 8
# samples 15 s apart (2 minutes). Exits 0 when quiet, 3 when --max-minutes pass first. Prints one line per check.
#   sh bench/wait-quiet.sh [max-minutes]   (default 100)
MAX=${1:-100}
START=$(date +%s)
gpu() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
while :; do
  ok=1; worst=0
  for i in 1 2 3 4 5 6 7 8; do
    if pgrep -x rsync >/dev/null; then ok=0; why=rsync; break; fi
    g=$(gpu); [ "${g:-100}" -gt "$worst" ] && worst=${g:-100}
    if [ "${g:-100}" -ge 10 ]; then ok=0; why="gpu ${g}%"; break; fi
    sleep 15
  done
  now=$(TZ=Asia/Kolkata date +%H:%M)
  if [ $ok = 1 ]; then echo "$now quiet (GPU max ${worst}% over 2 min, no rsync)"; exit 0; fi
  echo "$now not quiet: $why"
  [ $(( $(date +%s) - START )) -ge $(( MAX * 60 )) ] && exit 3
  sleep 60
done
