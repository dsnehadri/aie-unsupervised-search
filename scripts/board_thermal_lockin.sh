#!/bin/sh
# Lock-in style ON/OFF modulation run: 16 cycles of (load ~90s ON + 90s OFF).
# Folding the cycles in analysis cancels slow thermal/ambient drift.
cd /root
rm -f mod_log.csv mod_phase.txt mod_load.log
echo "run_start $(date +%s)" >> mod_phase.txt
python3 /root/power_sampler.py /root/mod_log.csv &
SAMPLER=$!
sleep 60
i=0
while [ $i -lt 16 ]; do
  echo "on_start $(date +%s) cycle=$i" >> mod_phase.txt
  ./host_aie_timed aie_stream_maskfix.xclbin eval_bkg.bin 2000 400 >> mod_load.log 2>&1
  echo "on_end $(date +%s) cycle=$i rc=$?" >> mod_phase.txt
  sleep 90
  i=$((i+1))
done
sleep 30
kill $SAMPLER
echo "run_end $(date +%s)" >> mod_phase.txt
