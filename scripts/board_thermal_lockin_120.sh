#!/bin/bash
# Lock-in power/temperature campaign with 120 s ON / 120 s OFF half-cycles.
# The 90 s runs (board_thermal_lockin.sh) show die temperature still rising at
# the end of the ON window; 120 s gives the thermal response room to plateau.
# Power settles within ~5 s, so it is unaffected either way.
#
# Usage (on the board, in /root):  ./board_thermal_lockin_120.sh [cycles] [xclbin] [iters_on] [host]
#   ON length is set by iters_on (events per host invocation x iterations):
#     AIE-PL hybrid image: 400 iters ~ 92 s  ->  use 520 for ~120 s
#     PL-only image      : 220 iters ~ 94 s  ->  use 280 for ~120 s
#   Phase stamps go to mod_phase.txt in the same on_start/on_end format the
#   folding scripts (plot_lockin_*.py, analyze_board_lockin.py) already parse.
CYCLES=${1:-200}
XCLBIN=${2:-aie_stream_retrained.xclbin}
ITERS=${3:-520}
HOST=${4:-host_aie_timed}
OFF_S=${OFF_S:-120}
echo "run_start $(date +%s) cycles=$CYCLES xclbin=$XCLBIN iters=$ITERS host=$HOST off_s=$OFF_S" >> mod_phase.txt
sleep 60
i=0
while [ $i -lt "$CYCLES" ]; do
  echo "on_start $(date +%s) cycle=$i" >> mod_phase.txt
  ./"$HOST" "$XCLBIN" eval_bkg.bin 2000 "$ITERS" >> mod_load.log 2>&1 &
  PID=$!
  n=0
  while kill -0 $PID 2>/dev/null && [ $n -lt 200 ]; do sleep 2; n=$((n+1)); done   # 400 s watchdog
  if kill -0 $PID 2>/dev/null; then
    kill -9 $PID; wait $PID 2>/dev/null
    echo "on_end $(date +%s) cycle=$i rc=TIMEOUT" >> mod_phase.txt
    break
  fi
  wait $PID
  echo "on_end $(date +%s) cycle=$i rc=$?" >> mod_phase.txt
  sleep "$OFF_S"
  i=$((i+1))
done
sleep 30
echo "run_end $(date +%s)" >> mod_phase.txt
