#!/bin/sh
# Overnight B3: wait for a quiet machine, then the full speed rerun with GPU utilization logged every 30 s.
# Exit 3 when the machine never went quiet within the wait; the caller re-arms.
cd "$(dirname "$0")/.." || exit 1
LOG=${LOG:-/tmp/anyformer-night}
mkdir -p "$LOG"
sh bench/wait-quiet.sh "${WAIT:-100}" >> "$LOG/quiet.log" || exit 3
OUT=plan/bench/$(TZ=Asia/Kolkata date +%F)-speed-quiet.json
( while :; do echo "$(TZ=Asia/Kolkata date +%H:%M:%S) $(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1) rsync=$(pgrep -x rsync | wc -l | tr -d ' ')"; sleep 30; done ) > "$LOG/gpu-b3.log" 2>&1 &
SAMPLER=$!
node bench/speed.mjs --runs 3 --n 256 --out "$OUT" > "$LOG/b3.out" 2>&1
CODE=$?
kill $SAMPLER 2>/dev/null
echo "b3 exit $CODE -> $OUT"
exit $CODE
