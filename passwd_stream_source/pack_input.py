"""
pack_input.py - convert .npy files into binary for VCK190 host app

Usage:
    python pack_input.py \
        --jets /home/snehadri/repos/unsupervised-search/phase3_export/test_vectors/stage0_input_raw.npy \
        --mask /home/snehadri/repos/unsupervised-search/phase3_export/test_vectors/stage0_padding_mask.npy \
        --output input.bin \
        --n_events 10

"""

import argparse
import numpy as np
import struct

def float_to_ap_fixed_16_5(val):
    frac_bits = 11
    scale = 2.0 ** frac_bits

    # clamp to representative range
    max_val = (2**15 - 1) / scale
    min_val = -(2**15) / scale
    val = np.clip(val, min_val, max_val)

    # quantize: truncate towards negative infinity (matches HLS default AP_TRN)
    fixed_int = int(np.floor(val * scale))

    # wrap to 16-bit two's complement
    fixed_int = fixed_int & 0xFFFF

    return fixed_int

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jets", required=True, help = "Path to stage0_input_raw.npy")
    parser.add_argument("--mask", required=True, help = "Path to stage0_padding_mask.npy")
    parser.add_argument("--output", default = "input.bin", help = "Output binary file")
    parser.add_argument("--n_events", type=int, default=1, help = "Number of events to pack")
    args = parser.parse_args()

    # load test vectors
    jets = np.load(args.jets)
    mask = np.load(args.mask)

    n_avail = min(jets.shape[0], mask.shape[0])
    n_events = min(args.n_events, n_avail)
    print(f"Packing {n_events} (of {n_avail} available)")

    words = []
    for ev in range(n_events):
        for i in range(12):
            for j in range(5):
                words.append(float_to_ap_fixed_16_5(jets[ev, i, j]))

        for i in range(12):
            words.append(1 if mask[ev, i] else 0)

    # write as binary uint32
    data = struct.pack(f"{len(words)}I", *words)
    with open(args.output, "wb") as f:
        f.write(data)

    print(f"written {len(words)} words ({len(data)} bytes) to {args.output}")
    print(f" {n_events} events = 72 words/event = {n_events * 72} words")

if __name__ == "__main__":
    main()