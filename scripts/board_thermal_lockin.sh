#!/bin/sh
# Lock-in style ON/OFF modulation run: N cycles of (load ~90s ON + 90s OFF).
# Folding the cycles in analysis cancels slow thermal/ambient drift.
# Usage (on the board): ./thermal_lockin.sh [cycles] [xclbin] [iters_on] [host]
# The xclbin MUST match the active BOOT.BIN image — a mismatch wedges the
# AIE partition (gated-tile spin) and needs a reboot to clear.
# iters_on sets the ON-window length: ~90 s means 400 iters on the AIE image
# (0.223 s/iter) but only 51 on the all-PL image (1.756 s/iter).
CYCLES=${1:-16}
XCLBIN=${2:-aie_stream_maskfix.xclbin}
ITERS=${3:-400}
HOST=${4:-host_aie_timed}
cd /root
rm -f mod_log.csv mod_phase.txt mod_load.log
echo "run_start $(date +%s) cycles=$CYCLES xclbin=$XCLBIN iters=$ITERS host=$HOST" >> mod_phase.txt
python3 /root/power_sampler.py /root/mod_log.csv &
SAMPLER=$!
sleep 60
i=0
while [ $i -lt "$CYCLES" ]; do
  echo "on_start $(date +%s) cycle=$i" >> mod_phase.txt
  ./"$HOST" "$XCLBIN" eval_bkg.bin 2000 "$ITERS" >> mod_load.log 2>&1 &
  PID=$!
  n=0
  while kill -0 $PID 2>/dev/null && [ $n -lt 150 ]; do sleep 2; n=$((n+1)); done
  if kill -0 $PID 2>/dev/null; then
    # host is spinning (wedged AIE partition) - kill it and end the campaign;
    # completed cycles remain usable, recovery needs a reboot anyway
    kill -9 $PID; wait $PID 2>/dev/null
    echo "on_end $(date +%s) cycle=$i rc=TIMEOUT" >> mod_phase.txt
    break
  fi
  wait $PID
  echo "on_end $(date +%s) cycle=$i rc=$?" >> mod_phase.txt
  sleep 90
  i=$((i+1))
done
sleep 30
kill $SAMPLER
echo "run_end $(date +%s)" >> mod_phase.txt
