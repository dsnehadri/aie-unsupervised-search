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
DATA_FRAC_BITS = 11
DATA_SCALE = 1 << DATA_FRAC_BITS

# Default tolerance — matches the relaxed bound your PL stream tb used for the
# 16-bit quantized pipeline. Tighten once you trust the kernel.
DEFAULT_TOL = 0.5


# ----------------------------------------------------------------------------
# PLIO text file -> int16 array
# ----------------------------------------------------------------------------
def parse_plio_text(path, expected_count):
    """Read a PLIO text file (4 hex int16s per line) into a flat int16 array.

    Trailing zero-padding (added by gen_attn_inputs to align to 64-bit transfers)
    is trimmed back to expected_count.
    """
    if not os.path.isfile(path):
        sys.exit(f"output file not found: {path}")

    vals = []
    with open(path, "r") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.split():
                # accept "0x1234" or "1234" (hex assumed)
                tok = tok.lower().lstrip("0x") or "0"
                try:
                    u = int(tok, 16) & 0xFFFF
                except ValueError:
                    sys.exit(f"{path}:{line_no}: cannot parse token '{tok}'")
                # uint16 -> int16 two's complement
                vals.append(u - 0x10000 if u >= 0x8000 else u)

    if len(vals) < expected_count:
        sys.exit(f"{path}: only {len(vals)} values, expected at least {expected_count}")

    return np.array(vals[:expected_count], dtype=np.int16)


def dequantize(int16_arr):
    """int16 -> float using the same Q4.11 scaling as gen_attn_inputs.py."""
    return int16_arr.astype(np.float32) / DATA_SCALE


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
                    help="Event index that was tested (default: 0)")
    ap.add_argument("--data-dir", default="./data",
                    help="Where gen_attn_inputs wrote its sidecar files")
    ap.add_argument("--output-dir", default=None,
                    help="Override directory containing PLIO output .txt files")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"Pass tolerance on max_err (default: {DEFAULT_TOL})")
    args = ap.parse_args()

    tv = os.path.join(args.phase3, "test_vectors")
    ev = args.event

    # Padding mask saved by gen_attn_inputs.py for this event
    mask_path = os.path.join(args.data_dir, "padding_mask_event.npy")
    if os.path.isfile(mask_path):
        pad_mask = np.load(mask_path).astype(bool)
    else:
        # Fallback: load directly from phase-3
        pad_mask = np.load(os.path.join(tv, "stage0_padding_mask.npy"))[ev].astype(bool)

    failures = 0
    print(f"checking AIE attention outputs against golden (event {ev}, tol={args.tol})\n")

    # ---- test 1: obj ---------------------------------------------------------
    print("test 1: OBJ")
    out_path = find_output("obj_x_out_L0.txt", args.output_dir)
    obj_int16 = parse_plio_text(out_path, expected_count=N_MAX * E_DIM)
    obj_out = dequantize(obj_int16).reshape(N_MAX, E_DIM)
    obj_gold = np.load(os.path.join(tv, "stage3_layer0_post_obj_selfattn.npy"))[ev]
    ok, _, _ = compare("obj_blocks", obj_out, obj_gold, mask=pad_mask, tol=args.tol)
    if not ok:
        failures += 1

    # ---- test 2: cand --------------------------------------------------------
    print("\ntest 2: CAND")
    out_path = find_output("cand_c_out_L0.txt", args.output_dir)
    cand_int16 = parse_plio_text(out_path, expected_count=T_DIM * E_DIM)
    cand_out = dequantize(cand_int16).reshape(T_DIM, E_DIM)
    cand_gold = np.load(os.path.join(tv, "stage3_layer0_post_cand_selfattn.npy"))[ev]
    ok, _, _ = compare("cand_blocks", cand_out, cand_gold, mask=None, tol=args.tol)
    if not ok:
        failures += 1

    # ---- test 3: cross -------------------------------------------------------
    print("\ntest 3: CROSS")
    out_path = find_output("cross_x_out_L0.txt", args.output_dir)
    cross_int16 = parse_plio_text(out_path, expected_count=N_MAX * E_DIM)
    cross_out = dequantize(cross_int16).reshape(N_MAX, E_DIM)
    # Re-mask padded rows the same way the PL testbench does before comparing
    cross_out[pad_mask] = 0.0
    cross_gold = np.load(os.path.join(tv, "stage3_layer0_post_cross_attn.npy"))[ev]
    ok, _, _ = compare("cross_blocks", cross_out, cross_gold, mask=pad_mask, tol=args.tol)
    if not ok:
        failures += 1

    # ---- summary -------------------------------------------------------------
    print()
    if failures == 0:
        print(f"ALL PASSED: 0 failure(s)")
        sys.exit(0)
    else:
        print(f"FAILED: {failures} failure(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()