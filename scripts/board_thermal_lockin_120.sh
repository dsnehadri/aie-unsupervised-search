#!/bin/sh
# Lock-in power/temperature campaign, 120 s ON / 120 s OFF.
#
# The ON window is now TIME-based. It used to be set by an iteration count,
# which meant the window silently changed length whenever the design's speed
# changed: 920 iterations gave 108 s on the hybrid, and would give ~32 s on the
# build with the embedding on the array. The script now calibrates the host
# once and picks the iteration count that lands on ON_S seconds.
#
# Usage (on the board, in /root):  ./board_thermal_lockin_120.sh [cycles] [xclbin] [host]
#   ON_S / OFF_S override the half-cycle lengths.
# Phase stamps go to mod_phase.txt in the on_start/on_end format the folding
# scripts already parse; they use the stamps, not the nominal length.
CYCLES=${1:-200}
XCLBIN=${2:-aie_stream.xclbin}
# host_aie_timed opens the AI Engine graph and dies on a fabric-only image
# ("Can not get id for Graph aie_graph"); plhost_timed is the fabric load. Pick
# by xclbin name unless told otherwise.
case "$XCLBIN" in
  pl_stream*) DEFHOST=plhost_timed ;;
  *)          DEFHOST=host_aie_timed ;;
esac
HOST=${3:-$DEFHOST}
ON_S=${ON_S:-120}
OFF_S=${OFF_S:-120}
NEV=${NEV:-2000}
cd /root
rm -f mod_log.csv mod_phase.txt mod_load.log

now() { cut -d' ' -f1 /proc/uptime; }
run_iters() {   # $1 = iterations, prints elapsed seconds
  t0=$(now)
  ./"$HOST" "$XCLBIN" eval_bkg.bin "$NEV" "$1" >> mod_load.log 2>&1
  t1=$(now)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}'
}

# --- calibrate: two points remove the fixed host startup cost ---------------
e1=$(run_iters 20)
e2=$(run_iters 60)
ITERS=$(awk -v e1="$e1" -v e2="$e2" -v on="$ON_S" 'BEGIN{
  per=(e2-e1)/40; if (per<=0) per=e2/60;
  n=int(on/per + 0.5); if (n<1) n=1; print n}')
# a host that failed fast gives no scaling with the iteration count; refuse
# rather than run a campaign whose ON window is meaningless
BAD=$(awk -v e1="$e1" -v e2="$e2" 'BEGIN{print ((e2-e1) < 0.2*e1 || e2 < 0.5) ? 1 : 0}')
if [ "$BAD" = "1" ]; then
  echo "ABORT: $HOST does not scale with iterations (20 -> ${e1}s, 60 -> ${e2}s)."
  echo "       It probably failed on this image. Check: ./$HOST $XCLBIN eval_bkg.bin 2000 5"
  exit 1
fi
PER=$(awk -v e1="$e1" -v e2="$e2" 'BEGIN{printf "%.4f", (e2-e1)/40}')
echo "calib 20 iters=${e1}s 60 iters=${e2}s -> ${PER}s/iter -> iters=$ITERS for ${ON_S}s ON"
echo "calib per_iter=$PER iters=$ITERS on_s=$ON_S" >> mod_phase.txt

echo "run_start $(date +%s) cycles=$CYCLES xclbin=$XCLBIN iters=$ITERS host=$HOST on_s=$ON_S off_s=$OFF_S" >> mod_phase.txt
python3 /root/power_sampler.py /root/mod_log.csv &
SAMPLER=$!
sleep 60
i=0
while [ $i -lt "$CYCLES" ]; do
  echo "on_start $(date +%s) cycle=$i" >> mod_phase.txt
  ./"$HOST" "$XCLBIN" eval_bkg.bin "$NEV" "$ITERS" >> mod_load.log 2>&1 &
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
kill $SAMPLER 2>/dev/null
echo "run_end $(date +%s)" >> mod_phase.txt
