# Our AIE-vs-PL deployment in the LARE frame (arXiv:2604.19106)

*Ma, Abarajithan, Danopoulos, Weng, Restuccia, Kastner — "Design Rules for Extreme-Edge
Scientific Computing on AI Engines."* Same board (VCK190), same domain (real-time HEP ML).
Their **LARE** metric is the formal version of the AIE-vs-PL decision this repo made empirically.

## What LARE is
For a layer, sweep the PL reuse factor to trace a **resource↔performance** curve (more reuse =
fewer resources, worse latency). Measure the AIE performance. **LARE = the PL resource needed for
the PL design to *match* the AIE performance.** Two uses:
- **Decision boundary** — if your available PL budget for that layer **> LARE**, PL can match/beat
  AIE (use PL); if **< LARE**, resource-starved PL is worse (use AIE).
- **Efficiency indicator** — a *low* LARE means the AIE result is matchable with little PL, i.e.
  the AIE is being used inefficiently and wants better tiling/dataflow.
Normalization: 1 AIE-ML tile ≈ **58 DSP58** (256 int8 MAC/cyc). *(Our board is the VC1902 with
**400 AIE1** tiles at ~128 int8 MAC/cyc ≈ **29 DSP-eq** — so we quote the 29–58 range.)*

## Our two measured points (Passwd-ABC on VCK190)
| | all-PL | AIE-hybrid | note |
|---|---|---|---|
| latency | 2.09 ms/ev | **0.894 ms/ev** | AIE 2.34× |
| throughput | 478 ev/s | **1118 ev/s** | AIE 2.34× |
| DSP | 1055 | 1680 | of 1968 |
| **LUT** | **204,895** | **120,809** | AIE **frees 41%** |
| AIE tiles | 0 | 78 | of 400 |
| accuracy error | **0% (exact)** | 16.9% | int16 AIE-attn quant |

Figure: `figs/lare_comparison.png` recasts these in the LARE normalization.

## Where we agree with the paper
- **We're a concrete instance of their thesis.** They argue spatial-dataflow PL (hls4ml) scales out
  early and AIE wins for *larger* NNs. Passwd-ABC (multi-head attention + autoencoder) is exactly
  such a model — offloading the attention to AIE gave a clean **2.34×**. Our `throughput_crossover`
  chart is an end-to-end LARE crossover.
- **The crossover depends on the rate budget, as LARE says.** Linear-scaling our PL point to match
  the hybrid's 1118 ev/s needs **~2470 DSP > the 1968 available** → at that throughput the PL budget
  is *below* LARE → **AIE-favorable**. But at 478 ev/s, PL fits with headroom → **PL-favorable**. We
  sit right on the crossover; which side depends on the required event rate.
- **The efficiency indicator fires on us.** Throughput per DSP-equivalent is *lower* for the hybrid
  (**0.18–0.28** vs **0.45** for PL) — the 78 tiles are under-used. Our own profiling said the same
  ("AIE tiles are not the bottleneck"; the PL↔AIE bridge / data movement is). That is precisely
  LARE's "low efficiency → needs tiling/dataflow optimization," which is the paper's next section.

## Where our case differs / extends LARE
1. **LARE has no accuracy axis — and for anomaly detection that's the deciding one.** The AIE offload
   cost **16.9% output error** (int16 quantization of the attention), which moves an *AUC*. We
   ultimately shipped the **all-PL, exact** design (0% error, hardware AUC 0.967) — *not* the faster
   hybrid — because at our modest rate PL was feasible and lossless. Our decision **inverts** LARE's
   resource-only verdict. For unsupervised anomaly work, LARE needs a third dimension (AUC/accuracy)
   or an "AIE quant must preserve AUC" constraint.
2. **We're LUT-bound in PL; LARE models DSP.** Attention in PL is dominated by softmax / layernorm /
   reshuffling — *control and LUT*, not MACs. Moving it to AIE freed **41% of LUTs**, not DSPs (DSP
   actually rose). LARE's tile→DSP normalization therefore *undercounts* the AIE benefit for
   attention; a LARE for transformer-like layers should normalize on the **binding** resource (LUT
   here), not DSP.
3. **Different rate regime.** The paper targets the **40 MHz** L1 trigger with tiny nets (jet-tagger,
   τ-selection). We run a **much heavier** per-event model at **~kHz**. Because our model is far
   larger, the crossover favors AIE at a far lower event rate than their small-net examples.
4. **End-to-end vs per-layer.** LARE is a per-dense-layer analytic sweep; ours is a whole-pipeline
   measurement (embed + pairwise + attention + AE + host bridge), so our "crossover" folds in the
   real data-movement and bridge costs that the per-layer model omits — which is exactly why our AIE
   utilization looks low.

## Verdict for this model
By LARE's letter — *heavy attention layer, PL DSP budget below the resource-to-match at target rate*
— the attention block is **AIE-favorable**, and that matched our measured 2.34×. But two things LARE
doesn't score flipped our final choice to **all-PL**: (a) the AIE int16 quantization moved the AUC,
and (b) our required rate was low enough that PL fit losslessly. So this project is both a
**confirmation** of LARE (AIE wins the compute-density/throughput argument, and the tile
underutilization is real) and a **case for extending it** with an accuracy axis and a binding-resource
(not DSP-only) normalization when the workload is attention rather than dense GEMM.

## Improving the AIE's LARE efficiency (measured)

The low efficiency indicator above is **not** the AIE being weak — it's us running **one event at a
time**, so the tiles idle waiting for the PL bridge. Our own profiling proves it:
- Per-tile busy%: the `obj` attention tiles run **99%** busy but the `cand` tiles only **~37%**
  (load imbalance), and the standalone AIE attention subgraphs sustain **obj 4738 / cand 14808 ev/s**
  — far above the 551 ev/s the single-event pipeline delivers. *"AIE attention is faster than the
  pipeline can feed it; tiles sit idle."*
- **Cross-event pipelining (N=100), measured: 7549 ev/s on the same 78 tiles — 13.7×, no extra tiles.**

Recomputing the LARE efficiency (throughput per DSP-equivalent):

| config | throughput | perf / DSP-eq | vs PL |
|---|---|---|---|
| all-PL | 478 | 0.45 | — |
| all-AIE, 1 event live | 551 | 0.11–0.21 | **below PL** |
| **all-AIE + pipelining** | **7549** | **1.56–2.93** | **3.4–6.5× above PL** |

So keeping the AIE fed **flips the LARE verdict**: the AIE goes from *worse* than PL per DSP-equivalent
to **3–6× better**, at **15.8× the PL throughput** — and the all-AIE build also **frees the fabric**
(DSP 1055→314, −70%; LUT 204,895→70,607, −66%). Concrete levers, by measured impact:
1. **Cross-event pipelining** (many events in flight) — the 13.7× above; the single biggest fix.
2. **Load rebalancing** — the `cand` tiles at 37% busy are over-provisioned; folding that work onto
   fewer tiles lowers the DSP-equivalent and raises efficiency further.
3. **Clock/bridge** — the AIE can run ~1 GHz vs our 100 MHz PL feed; widening the PL↔AIE PLIO and
   deepening the bridge pipeline recovers the rest.

**Caveat unchanged:** this is the LARE (latency/resource) axis only. The all-AIE path still carries the
int16 attention quantization (the AUC cost), which is why the *shipped* golden design is all-PL. The
improvement here is the right move when throughput/resource is the binding constraint and the AUC hit
is acceptable (or the attention is quantized more carefully).
