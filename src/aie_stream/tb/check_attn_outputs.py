#!/usr/bin/env python3
"""
check_attn_outputs.py
---------------------
Post-sim helper for the AIE attention testbench.

Parses the PLIO output text files written by aiesimulator (or x86simulator),
dequantizes int16 -> float, and compares against the phase-3 golden vectors
for the same event.

Mirrors the PL testbench's compare<N_ROWS>() reporting:
    - per-test max abs error and RMSE
    - PASS / FAIL based on a tolerance
    - skips padded rows for obj/cross
    - returns nonzero if any test fails

Output files searched (first match wins):
    ./data/<name>.txt                    (some flows write back to data/)
    ./aiesimulator_output/data/<name>.txt
    ./x86simulator_output/data/<name>.txt
    --output-dir <dir>/<name>.txt        (overrides everything)

Files expected:
    obj_x_out_L0.txt      [N_MAX x E_DIM]  vs stage3_layer0_post_obj_selfattn.npy
    cand_c_out_L0.txt     [T_DIM x E_DIM]  vs stage3_layer0_post_cand_selfattn.npy
    cross_x_out_L0.txt    [N_MAX x E_DIM]  vs stage3_layer0_post_cross_attn.npy
"""

import argparse
import os
import sys
import numpy as np


# ----------------------------------------------------------------------------
# Architectural constants (must match attn_aie_types.h and gen_attn_inputs.py)
# ----------------------------------------------------------------------------
N_MAX = 12
E_DIM = 16
T_DIM = 3
DATA_FRAC_BITS = 9  # retrained layout: data Q6.9 (was 11)
DATA_SCALE = 1 << DATA_FRAC_BITS

# Default tolerance — matches the relaxed bound your PL stream tb used for the
# 16-bit quantized pipeline. Tighten once you trust the kernel.
DEFAULT_TOL = 0.5


# ----------------------------------------------------------------------------
# PLIO text file -> int16 array
# ----------------------------------------------------------------------------
def parse_plio_text(path, expected_count):
    """Read a PLIO text file (4 signed-decimal int16s per line) into a flat int16 array.

    aiesimulator interleaves "T <time> ps|ns" timestamp lines with the data
    lines in its outputs; these are skipped here. Trailing zero-padding (added
    by gen_attn_inputs to align to 64-bit transfers) is trimmed back to
    expected_count.
    """
    if not os.path.isfile(path):
        sys.exit(f"output file not found: {path}")

    vals = []
    with open(path, "r") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("T ") or line == "TLAST":
                continue
            for tok in line.split():
                try:
                    vals.append(int(tok))
                except ValueError:
                    sys.exit(f"{path}:{line_no}: cannot parse token '{tok}'")

    if len(vals) < expected_count:
        sys.exit(f"{path}: only {len(vals)} values, expected at least {expected_count}")

    return np.array(vals[:expected_count], dtype=np.int16)


def parse_plio_text_float(path):
    """Parse a FLOAT_AIE PLIO output file: tokens are decimal floats (2 per
    64-bit line). Timestamp lines are skipped."""
    if not os.path.isfile(path):
        sys.exit(f"output file not found: {path}")
    vals = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "TLAST" or line.startswith("#") or line.startswith("T "):
                continue
            for s in line.split():
                vals.append(float(s))
    return np.array(vals, dtype=np.float32)


def parse_plio_text_with_timestamps(path):
    """Parse PLIO output preserving per-line ('T <time> ns') timestamps.

    Returns (timestamps_ns_per_4word: list[float], data_int16: np.ndarray flat).
    The i-th timestamp is the simulated time (in ns) at which the i-th 4-int16
    chunk arrived on the PLIO output.
    """
    if not os.path.isfile(path):
        sys.exit(f"output file not found: {path}")
    times, vals = [], []
    last_t = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "TLAST" or line.startswith("#"):
                continue
            if line.startswith("T "):
                # "T 12345 ns" or "T 12345 ps"
                tok = line.split()
                t = float(tok[1])
                unit = tok[2] if len(tok) > 2 else "ns"
                if unit == "ps": t *= 1e-3
                last_t = t
                continue
            # data line
            for s in line.split():
                vals.append(int(s))
            if last_t is not None:
                times.append(last_t)
    return times, np.array(vals, dtype=np.int16)


def dequantize(int16_arr, scale=DATA_SCALE):
    """int16 -> float. Default Q4.11 for obj/cross outputs; cand uses scale=512 (Q6.9)."""
    return int16_arr.astype(np.float32) / scale


# cand pipeline runs at Q6.9 throughout (see attn_aie_types.h CAND_SCALE).
CAND_SCALE = 1 << 9   # 512


# ----------------------------------------------------------------------------
# Comparison (mirrors compare<N_ROWS> in tb_helpers.h)
# ----------------------------------------------------------------------------
def compare(name, computed, golden, mask=None, tol=DEFAULT_TOL):
    """Return (passed, max_err, rmse). Reports the same way compare<>() does."""
    diff = (computed - golden).astype(np.float64)

    if mask is not None:
        # mask shape (N_ROWS,): True = padded -> skip that row entirely
        keep = ~mask
        diff = diff[keep]
        n_compared = diff.size
    else:
        n_compared = diff.size

    if n_compared == 0:
        print(f"  {name:<20s}  no rows to compare (all masked)")
        return True, 0.0, 0.0

    max_err = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    passed = max_err < tol
    status = "PASS" if passed else "FAIL"
    print(f"  {name:<20s}  max_err={max_err:.6f}  rmse={rmse:.6f}  "
          f"n={n_compared}  tol={tol:.3f}  {status}")
    return passed, max_err, rmse


# ----------------------------------------------------------------------------
# File search
# ----------------------------------------------------------------------------
def find_output(filename, override_dir=None):
    """Locate a PLIO output file, checking the usual simulator output dirs."""
    if override_dir:
        cand = os.path.join(override_dir, filename)
        if os.path.isfile(cand):
            return cand
        sys.exit(f"--output-dir override: file not found at {cand}")

    candidates = [
        os.path.join("aiesimulator_output", "data", filename),
        os.path.join("x86simulator_output", "data", filename),
        os.path.join("data", filename),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    sys.exit(f"output file not found in any of: {candidates}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3", required=True,
                    help="Phase-3 export directory")
    ap.add_argument("--event", type=int, default=0,
                    help="Starting event index that was tested (default: 0)")
    ap.add_argument("--num-events", type=int, default=None,
                    help="Number of events to validate (default: auto-detect)")
    ap.add_argument("--data-dir", default="./data",
                    help="Where gen_attn_inputs wrote its sidecar files")
    ap.add_argument("--output-dir", default=None,
                    help="Override directory containing PLIO output .txt files")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"Pass tolerance on max_err (default: {DEFAULT_TOL})")
    ap.add_argument("--timing-out", default=None,
                    help="If set, dump per-event first-output timestamps as JSON to this path")
    ap.add_argument("--float", dest="float_mode", action="store_true",
                    help="FLOAT_AIE build: parse float PLIO outputs, no "
                         "dequantization, check ALL 6 blocks (L0 + L1)")
    ap.add_argument("--per-event-out", default=None,
                    help="If set, dump per-event per-block max abs errors as JSON")
    ap.add_argument("--all-blocks", action="store_true",
                    help="Check all 6 blocks (L0 + L1) in int16 mode too")
    args = ap.parse_args()

    tv = os.path.join(args.phase3, "test_vectors")
    ev0 = args.event

    # Padding masks: now shape (nev, N_MAX) when generated by N-event mode
    mask_path = os.path.join(args.data_dir, "padding_mask_event.npy")
    if os.path.isfile(mask_path):
        pad_masks_arr = np.load(mask_path).astype(bool)
        if pad_masks_arr.ndim == 1:   # legacy single-event format
            pad_masks_arr = pad_masks_arr[None, :]
    else:
        pad_masks_arr = np.load(os.path.join(tv, "stage0_padding_mask.npy"))[ev0:ev0+1].astype(bool)

    # Determine N
    if args.num_events is not None:
        nev = args.num_events
    else:
        rng_path = os.path.join(args.data_dir, "event_range.txt")
        if os.path.isfile(rng_path):
            with open(rng_path) as f:
                _, n_str = f.read().split()
                nev = int(n_str)
        else:
            nev = pad_masks_arr.shape[0]

    # ---- all-6-block mode (FLOAT_AIE, or int16 with --all-blocks) ----------
    if args.float_mode or args.all_blocks:
        mode = "FLOAT" if args.float_mode else "int16"
        print(f"{mode} mode: checking all 6 blocks vs PyTorch "
              f"(events {ev0}..{ev0+nev-1}, tol={args.tol})\n")
        blocks = [
            ("obj_L0",   "obj_x_out_L0.txt",   "stage3_layer0_post_obj_selfattn.npy",  N_MAX, True),
            ("cand_L0",  "cand_c_out_L0.txt",  "stage3_layer0_post_cand_selfattn.npy", T_DIM, False),
            ("cross_L0", "cross_x_out_L0.txt", "stage3_layer0_post_cross_attn.npy",    N_MAX, True),
            ("obj_L1",   "obj_x_out_L1.txt",   "stage3_layer1_post_obj_selfattn.npy",  N_MAX, True),
            ("cand_L1",  "cand_c_out_L1.txt",  "stage3_layer1_post_cand_selfattn.npy", T_DIM, False),
            ("cross_L1", "cross_x_out_L1.txt", "stage3_layer1_post_cross_attn.npy",    N_MAX, True),
        ]
        per_event = {}
        failures = 0
        for name, fname, gname, n_rows, masked in blocks:
            d = parse_plio_text_float(find_output(fname, args.output_dir))
            if not args.float_mode:
                d = d / DATA_SCALE   # cand runs at the same Q6.9 scale now
            gold = np.load(os.path.join(tv, gname))
            per = n_rows * E_DIM
            errs = []
            for i in range(min(nev, len(d) // per)):
                ev = ev0 + i
                o = d[i*per:(i+1)*per].reshape(n_rows, E_DIM)
                diff = (o - gold[ev]).astype(np.float64)
                if masked:
                    diff = diff[~pad_masks_arr[i]]
                err = float(np.max(np.abs(diff))) if diff.size else 0.0
                errs.append(err)
                if err >= args.tol:
                    failures += 1
            a = np.array(errs)
            per_event[name] = errs
            print(f"  {name:<10s} N={len(a):3d}  max={a.max():.3e}  "
                  f"mean={a.mean():.3e}  fail={(a >= args.tol).sum()}")
        if args.per_event_out:
            import json
            with open(args.per_event_out, "w") as f:
                json.dump(per_event, f, indent=2)
            print(f"\nwrote per-event errors to {args.per_event_out}")
        print()
        if failures == 0:
            print(f"ALL PASSED: 0 failure(s) across {nev} events x 6 blocks")
            sys.exit(0)
        print(f"FAILED: {failures} block-events over tol")
        sys.exit(1)

    print(f"checking AIE attention outputs against golden (events {ev0}..{ev0+nev-1}, tol={args.tol})\n")

    # Output file paths
    obj_path   = find_output("obj_x_out_L0.txt",   args.output_dir)
    cand_path  = find_output("cand_c_out_L0.txt",  args.output_dir)
    cross_path = find_output("cross_x_out_L0.txt", args.output_dir)

    # Parse all output streams with timestamps for timing extraction
    obj_t,   obj_d   = parse_plio_text_with_timestamps(obj_path)
    cand_t,  cand_d  = parse_plio_text_with_timestamps(cand_path)
    cross_t, cross_d = parse_plio_text_with_timestamps(cross_path)

    # Per-event element counts
    obj_per   = N_MAX * E_DIM       # 192
    cand_per  = T_DIM * E_DIM       # 48
    cross_per = N_MAX * E_DIM       # 192

    # Load golden
    obj_gold_arr   = np.load(os.path.join(tv, "stage3_layer0_post_obj_selfattn.npy"))
    cand_gold_arr  = np.load(os.path.join(tv, "stage3_layer0_post_cand_selfattn.npy"))
    cross_gold_arr = np.load(os.path.join(tv, "stage3_layer0_post_cross_attn.npy"))

    failures_obj = failures_cand = failures_cross = 0
    obj_errs, cand_errs, cross_errs = [], [], []

    for i in range(nev):
        ev = ev0 + i
        pad_mask = pad_masks_arr[i]

        # OBJ
        if (i+1) * obj_per <= len(obj_d):
            o16 = obj_d[i*obj_per:(i+1)*obj_per]
            o = dequantize(o16).reshape(N_MAX, E_DIM)
            diff = (o - obj_gold_arr[ev]).astype(np.float64)
            diff = diff[~pad_mask]
            err = float(np.max(np.abs(diff))) if diff.size else 0.0
            obj_errs.append(err)
            if err >= args.tol: failures_obj += 1

        # CAND
        if (i+1) * cand_per <= len(cand_d):
            c16 = cand_d[i*cand_per:(i+1)*cand_per]
            c = dequantize(c16, scale=CAND_SCALE).reshape(T_DIM, E_DIM)
            diff = (c - cand_gold_arr[ev]).astype(np.float64)
            err = float(np.max(np.abs(diff)))
            cand_errs.append(err)
            if err >= args.tol: failures_cand += 1

        # CROSS
        if (i+1) * cross_per <= len(cross_d):
            x16 = cross_d[i*cross_per:(i+1)*cross_per]
            x = dequantize(x16).reshape(N_MAX, E_DIM)
            x[pad_mask] = 0.0
            diff = (x - cross_gold_arr[ev]).astype(np.float64)
            diff = diff[~pad_mask]
            err = float(np.max(np.abs(diff))) if diff.size else 0.0
            cross_errs.append(err)
            if err >= args.tol: failures_cross += 1

    def stat(name, errs):
        if not errs:
            print(f"  {name:<14s} no data"); return
        a = np.array(errs)
        print(f"  {name:<14s} N={len(a):3d}  max={a.max():.5f}  mean={a.mean():.5f}  fail={(a >= args.tol).sum()}")

    print("per-event max abs error vs golden:")
    stat("obj",   obj_errs)
    stat("cand",  cand_errs)
    stat("cross", cross_errs)

    # Per-event first-output timestamps for timing extraction
    if args.timing_out:
        import json
        # obj_t is per-4-word-chunk (one entry per data line of 4 int16s).
        # Per-event first-output line = i * (per/4)
        obj_chunks_per_ev   = obj_per   // 4    # 48
        cand_chunks_per_ev  = cand_per  // 4    # 12
        cross_chunks_per_ev = cross_per // 4    # 48
        def per_ev_first(times, chunks_per):
            return [times[i*chunks_per] for i in range(len(times)//chunks_per)]
        def per_ev_last(times, chunks_per):
            return [times[(i+1)*chunks_per - 1] for i in range(len(times)//chunks_per)]
        timing = {
            "obj":   {"first_ns": per_ev_first(obj_t,   obj_chunks_per_ev),
                      "last_ns":  per_ev_last (obj_t,   obj_chunks_per_ev)},
            "cand":  {"first_ns": per_ev_first(cand_t,  cand_chunks_per_ev),
                      "last_ns":  per_ev_last (cand_t,  cand_chunks_per_ev)},
            "cross": {"first_ns": per_ev_first(cross_t, cross_chunks_per_ev),
                      "last_ns":  per_ev_last (cross_t, cross_chunks_per_ev)},
        }
        with open(args.timing_out, "w") as f:
            json.dump(timing, f, indent=2)
        print(f"\nwrote per-event timing to {args.timing_out}")

    failures = failures_obj + failures_cand + failures_cross
    print()
    if failures == 0:
        print(f"ALL PASSED: 0 failure(s) across {nev} events")
        sys.exit(0)
    else:
        print(f"FAILED: obj={failures_obj} cand={failures_cand} cross={failures_cross}")
        sys.exit(1)


if __name__ == "__main__":
    main()