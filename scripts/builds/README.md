# Build recipes

One script per hardware design in the paper. Each one hard-codes the flags that
design was compiled with, and writes them to `build_flags.txt` in the build
directory before it starts, so a finished build always carries its own recipe.

Flags are never read from the environment. Earlier builds took their AI Engine
flags from `AIE_XPRE` in the launching shell, which left no record of what was
actually compiled; the sets below were recovered afterwards from the compilers'
own project files (`_x/<kernel>/<kernel>/<kernel>.tcl`) and from
`aie_build.log`, which is the only place they survived.

Everything needs Vitis/Vivado 2025.2. The 2022.2 toolchain does not produce a
bootable image for this board.

| script | design | measured |
|---|---|---|
| `fabric_t2n.sh` | PL-only, 156.25 MHz | 43.9 µs, 9.3 µs/event |
| `fabric_t2p.sh` | PL-only, t2n logic at 170 MHz | 40.3 µs, 8.74 µs/event |
| `fabric_t2r.sh` | PL-only, dynamic rows + single-pass norm + wide reads | 40.3 µs, 8.93 µs/event |
| `hybrid_v5.sh` | AIE-PL hybrid, attention stack on the array | 45.6 µs, 5.0 µs/event |
| `hybrid_v6.sh` | v5 + pairwise MLP on the array (v6f = this at 133 MHz) | 42.4 µs, 4.2 µs/event |
| `hybrid_v8.sh` | v6 + dynamic rows + softmax table | 37.4–39.7 µs, 4.4 µs/event |
| `blocks_aie.sh` | one object/candidate/cross block on the array (Figure 7 left) | |
| `blocks_pl.sh` | the same three blocks in fabric (Figure 7 left) | |
| `obj20_sweep.sh` | 1–20 object-block instances (Figure 7 right) | up to 4.11M event/s |
| `link_bench.sh` | AIE-PL payload sweep (Figure 4) | 8.3 GB/s per direction |

## Order

The fabric designs need `src/pl_stream/weights_rom.h`, and the array designs
need the headers under `src/attn_block_aie/kernels/weights/`. Generate them
first with `src/pl_stream/export_weights_to_header.py`,
`src/attn_block_aie/slice_weights_for_aie.py` and
`scripts/export_{embed,pairwise}_weights_aie.py`, which all read the trained
checkpoint.

A hybrid build compiles the array graph into `libadf.a`, compiles the fabric
kernel into a `.xo`, links them, and packages `BOOT.BIN`. Deploying is
`deploy_bootbin.sh`, which verifies the image's md5 on the card before
rebooting; the image must contain `Version=2025.2`.
