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

# --- calibrate ---------------------------------------------------------------
# The ON window must be the SAME on every design, or a figure that shades one
# "compute" band for two traces misrepresents the slower one. Three things used
# to push it over ON_S:
#   1. the two-point fit removes the host's fixed startup cost F from the
#      per-iteration slope, but F is still spent, so the window ran ON_S + F;
#   2. calibration ran BEFORE the power sampler, so it measured an idle machine
#      and the campaign then ran ~2% slower with the sampler competing for CPU.
#      That error scales with the iteration count, which is why the hybrid at
#      3478 iterations overshot by 4.1 s and the fabric at 483 by only 2.1 s;
#   3. on a fast design the 20/60-iteration points are only ~1.8 and ~3.1 s, so
#      /proc/uptime's 10 ms granularity alone is ~1.4% on the slope.
# Fixes: start the sampler first, size the calibration points in TIME rather
# than iterations, and finish with a trial window that measures what actually
# happens and rescales. Result lands within the +-1 s the integer date stamp
# allows.
python3 /root/power_sampler.py /root/mod_log.csv &
SAMPLER=$!
sleep 5

e0=$(run_iters 20)
# choose calibration points worth roughly 15 s and 45 s of load
N1=$(awk -v e="$e0" 'BEGIN{n=int(20*15/(e>0.05?e:0.05)); if(n<20)n=20; if(n>200000)n=200000; print n}')
N2=$(awk -v n1="$N1" 'BEGIN{print n1*3}')
e1=$(run_iters "$N1")
e2=$(run_iters "$N2")
# a host that failed fast gives no scaling with the iteration count; refuse
# rather than run a campaign whose ON window is meaningless
BAD=$(awk -v e1="$e1" -v e2="$e2" 'BEGIN{print ((e2-e1) < 0.2*e1 || e2 < 0.5) ? 1 : 0}')
if [ "$BAD" = "1" ]; then
  echo "ABORT: $HOST does not scale with iterations ($N1 -> ${e1}s, $N2 -> ${e2}s)."
  echo "       It probably failed on this image. Check: ./$HOST $XCLBIN eval_bkg.bin 2000 5"
  kill $SAMPLER 2>/dev/null
  exit 1
fi
# per-iteration slope and fixed startup cost, both with the sampler running
PER=$(awk -v e1="$e1" -v e2="$e2" -v n1="$N1" 'BEGIN{printf "%.6f", (e2-e1)/(2*n1)}')
FIX=$(awk -v e1="$e1" -v p="$PER" -v n1="$N1" 'BEGIN{f=e1-n1*p; if(f<0)f=0; printf "%.3f", f}')
# LAG is the mean delay of the on_end stamp behind the host's real exit: the
# watchdog polls every second, so ~0.5 s.
LAG=0.5
ITERS=$(awk -v on="$ON_S" -v f="$FIX" -v p="$PER" -v l="$LAG" 'BEGIN{
  n=int((on-f-l)/p + 0.5); if (n<1) n=1; print n}')
echo "calib $N1 iters=${e1}s $N2 iters=${e2}s -> ${PER}s/iter fixed=${FIX}s -> iters=$ITERS"

# --- trial window: measure what that iteration count really produces ---------
# Timed exactly the way a campaign cycle is, watchdog and all, so the
# correction absorbs the polling lag and any residual rate error.
t0=$(now)
./"$HOST" "$XCLBIN" eval_bkg.bin "$NEV" "$ITERS" >> mod_load.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do sleep 1; done
wait $PID
t1=$(now)
W=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
ITERS=$(awk -v it="$ITERS" -v w="$W" -v on="$ON_S" -v f="$FIX" -v l="$LAG" 'BEGIN{
  num=on-f-l; den=w-f-l; if (den<=0) {print it; exit}
  n=int(it*num/den + 0.5); if (n<1) n=1; print n}')
echo "trial: $W s -> corrected iters=$ITERS for ${ON_S}s ON"
echo "calib per_iter=$PER fixed=$FIX trial=$W iters=$ITERS on_s=$ON_S" >> mod_phase.txt
sleep 30

echo "run_start $(date +%s) cycles=$CYCLES xclbin=$XCLBIN iters=$ITERS host=$HOST on_s=$ON_S off_s=$OFF_S" >> mod_phase.txt
sleep 60
i=0
while [ $i -lt "$CYCLES" ]; do
  echo "on_start $(date +%s) cycle=$i" >> mod_phase.txt
  ./"$HOST" "$XCLBIN" eval_bkg.bin "$NEV" "$ITERS" >> mod_load.log 2>&1 &
  PID=$!
  n=0
  while kill -0 $PID 2>/dev/null && [ $n -lt 400 ]; do sleep 1; n=$((n+1)); done   # 400 s watchdog
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
