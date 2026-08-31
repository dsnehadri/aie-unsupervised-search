#!/usr/bin/env bash
# Preflight + deploy + launch for an N-cycle lock-in power/temp run on the VCK190.
#
#   ./scripts/board_lockin_launch.sh              # 64 cycles (~3.4 h), auto-discover board
#   ./scripts/board_lockin_launch.sh 16           # shorter run
#   ./scripts/board_lockin_launch.sh 64 fe80::... # skip discovery
#
# Refuses to start if the board is busy or if the active BOOT.BIN has no known
# matching xclbin (a mismatched graph open wedges the AIE partition — reboot to
# clear). Writes the coordination lock into Claude's shared memory so a parallel
# session holds deploys; DELETE that lock if you abort the run by hand.
#
# Afterwards: scp mod_log.csv + mod_phase.txt off /root and run
#   python3 scripts/analyze_board_lockin.py mod_log.csv mod_phase.txt
set -euo pipefail
cd "$(dirname "$0")/.."

CYCLES=${1:-64}
ADDR=${2:-}
IF=eno2
SSH_OPTS=(-6 -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
LOCK="$HOME/.claude/projects/-home-snehadri-repos-aie-unsupervised-search/memory/thermal-run-in-progress.md"

# -- find the board (link-local address changes every boot) --------------------
if [ -z "$ADDR" ]; then
  ping6 -c2 "ff02::1%$IF" >/dev/null 2>&1 || true; sleep 2
  for a in $(ip -6 neigh show dev "$IF" | awk '$NF!="FAILED"{print $1}'); do
    h=$(ssh "${SSH_OPTS[@]}" "root@${a}%$IF" hostname </dev/null 2>/dev/null || true)
    if [ "$h" = "versal-rootfs-common-20252" ]; then ADDR=$a; break; fi
  done
fi
[ -n "$ADDR" ] || { echo "ERROR: board not found on $IF"; exit 1; }
R="root@${ADDR}%$IF"
echo "board: $ADDR"

# -- gate 1: refuse if anything is already running ----------------------------
busy=$(ssh "${SSH_OPTS[@]}" "$R" \
  'ps | grep -E "host_aie|plhost|power_sampler" | grep -v grep' </dev/null || true)
if [ -n "$busy" ]; then
  echo "ERROR: board is busy — not launching:"; echo "$busy"; exit 1
fi
if [ -f "$LOCK" ]; then
  echo "ERROR: coordination lock already present ($LOCK) — another run may be active"; exit 1
fi

# -- gate 2: image <-> xclbin pairing -----------------------------------------
img=$(ssh "${SSH_OPTS[@]}" "$R" 'cd /run/media/mmcblk0p1 && A=$(md5sum BOOT.BIN | cut -d" " -f1)
  for f in BOOT.BIN.*; do [ "$(md5sum "$f" | cut -d" " -f1)" = "$A" ] && echo "${f#BOOT.BIN.}"; done; true' \
  </dev/null | head -1 || true)
echo "active image: ${img:-UNKNOWN}"
# ITERS is chosen so each ON window is ~90 s at that image's per-iteration time.
case "$img" in
  aie_maskfix)      XCLBIN=aie_stream_maskfix.xclbin; HOST=host_aie_timed; ITERS=400 ;;
  aie_pipe_fast)    XCLBIN=aie_stream_fast.xclbin;    HOST=host_aie_timed; ITERS=400 ;;
  # NOTE: "plstream_batched" is a MISNOMER — it is byte-identical to
  # plstream_fast (the 1,139 ev/s optimized-kernels step). The true-batched
  # dataflow design is plstream_batched2 (4,869 ev/s). Prefer batched2.
  plstream_batched)  XCLBIN=pl_stream_batched.xclbin;  HOST=plhost_timed; ITERS=51 ;;
  plstream_batched2) XCLBIN=pl_stream_batched2.xclbin; HOST=plhost_timed; ITERS=220 ;;
  *) echo "ERROR: no known xclbin pairing for image '$img' — refusing" \
        "(a mismatch wedges the AIE array; add the pairing here once verified)"; exit 1 ;;
esac
ssh "${SSH_OPTS[@]}" "$R" "test -f /root/$XCLBIN && test -x /root/$HOST" </dev/null \
  || { echo "ERROR: /root/$XCLBIN or /root/$HOST missing on board"; exit 1; }
echo "xclbin: $XCLBIN  host: $HOST  iters/ON: $ITERS"

# -- deploy + coordination lock + launch --------------------------------------
scp "${SSH_OPTS[@]}" scripts/board_power_sampler.py "root@[${ADDR}%$IF]:/root/power_sampler.py"
scp "${SSH_OPTS[@]}" scripts/board_thermal_lockin.sh "root@[${ADDR}%$IF]:/root/thermal_lockin.sh"

mins=$(( CYCLES * 191 / 60 + 2 ))
cat > "$LOCK" <<EOF
---
name: thermal-run-in-progress
description: "ACTIVE LOCK: ~${mins}-min lock-in power/temp run on VCK190 ($img image) — HOLD reboots/deploys until this file is deleted"
metadata:
  type: project
---

**ACTIVE (launched $(date -u +%Y-%m-%dT%H:%MZ)):** $CYCLES-cycle ON/OFF lock-in measurement
running on the VCK190 (\`$img\` image, 90 s load / 90 s idle, ~${mins} min total).
Board address: \`$ADDR%$IF\`. Method per [[vck190-power-telemetry]].

**How to apply:** HOLD BOOT.BIN deployments/reboots until this file is deleted.
The launching session removes it when the run finishes; if it is much older than
${mins} min, the run is over — ignore/delete.
EOF

ssh "${SSH_OPTS[@]}" "$R" \
  "chmod +x /root/thermal_lockin.sh && nohup /root/thermal_lockin.sh $CYCLES $XCLBIN $ITERS $HOST > /root/thermal_lockin.out 2>&1 & echo launched pid \$!" </dev/null
echo "running: $CYCLES cycles, ~${mins} min. Progress: ssh ... 'tail /root/mod_phase.txt'"
echo "REMEMBER: delete $LOCK when the run is done."
