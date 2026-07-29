# Reading list — Passwd-ABC and its neighborhood

Curated around what this project actually did: reproduce **Passwd-ABC** (unsupervised jet→parent
assignment + anomaly score), compare it to standard jet-assignment algorithms, study the mass
*sculpting* problem, and deploy the model on an FPGA. ★ = start here.

---

## 0. The paper we reproduced
- ★ **Badea & Montejo Berlingen**, *A data-driven and model-agnostic approach to solving
  combinatorial assignment problems in searches for new physics* (Passwd-ABC).
  PRD 109, L011702 — **arXiv:2309.05728**. The ABC layer, the dual-autoencoder anomaly score,
  the min-mass-asymmetry baseline (our Fig 3 / assignment comparison).

## 1. Combinatorial jet↔parent assignment with ML
The core task Passwd-ABC solves in an *unsupervised* way; these are the supervised precursors it contrasts with.
- ★ **Fenton et al.**, *Permutation-invariant SPANet for reconstructing pair-produced particles*
  — PRD 105, 112008 — **arXiv:2010.09206**. The attention-based supervised assignment network.
- **Shmakov et al.**, *SPANet with symmetries / arbitrary final states* — SciPost 12, 178 — **arXiv:2106.03898**.
- **Badea, Fawcett, Huth, Khoo, Poggi, Lee**, jet assignment for resonances — PRD 106, 016001 — **arXiv:2201.02205**.
- **Qiu, Han, Ju, Nachman, Wang** — **arXiv:2203.05687**.
- **Ehrke, Raine, Zoch, Guth, Golling** — **arXiv:2303.13937**.
- **Lee, Park, Watson, Yang** — **arXiv:2012.03542**.
- **Kieseler**, *Object condensation* (assign hits/jets to objects without fixed multiplicity) — **arXiv:2002.03605**.

## 2. Permutation-invariant / attention architectures for particle physics
Why the ABC layer is built on attention over a *set* of jets.
- ★ **Vaswani et al.**, *Attention Is All You Need* — **arXiv:1706.03762**.
- **Lee et al.**, *Set Transformer* — **arXiv:1810.00825** (permutation-equivariant attention, closest to the ABC layer).
- **Zaheer et al.**, *Deep Sets* — **arXiv:1703.06114**.
- **Komiske, Metodiev, Thaler**, *Energy Flow Networks* (Deep Sets for jets) — **arXiv:1810.05165**.
- **Qu, Gouskos**, *ParticleNet* — **arXiv:1902.08570**; **Qu, Li, Qian**, *Particle Transformer* — **arXiv:2202.03772**.
- Discrete-assignment machinery the paper tested: **Gumbel-Softmax** (**arXiv:1611.01144**) and the
  **Concrete distribution** (**arXiv:1611.00712**).

## 3. Unsupervised & weakly-supervised anomaly detection for new physics
The conceptual family Passwd-ABC belongs to (train on background, flag the rest).
- ★ **Karagiorgi, Kasieczka, Kravitz, Nachman, Shih**, *Machine learning in the search for new
  fundamental physics* (review) — Nat. Rev. Phys. — **arXiv:2112.03769**.
- ★ **Kasieczka, Nachman, Shih et al.**, *The LHC Olympics 2020* (anomaly-detection community challenge)
  — **arXiv:2101.08320**; **Aarrestad et al.**, *Dark Machines Anomaly Score Challenge* — **arXiv:2105.14027**.
- **Farina, Nakai, Shih**, *Searching for New Physics with Deep Autoencoders* — **arXiv:1808.08992**.
- **Heimel, Kasieczka, Plehn, Thompson**, *QCD or What?* (autoencoder tagging) — **arXiv:1808.08979**.
- **Cerri, Nguyen, Pierini, Spiropulu, Vlimant**, *VAEs for New Physics Mining* — **arXiv:1811.10276**.
- Weakly-supervised / density line: **CWoLa** (**arXiv:1708.02949**), *CWoLa Hunting* (**arXiv:1805.02664**),
  **ANODE** (**arXiv:2001.04990**), **CATHODE** (**arXiv:2109.00546**).
- Cautionary: **Fraser, Golling, Kasieczka, Nachman et al.**, *Challenges for unsupervised anomaly detection*
  — **arXiv:2110.06948** (why AE reconstruction loss is a fragile anomaly metric — relevant to our AUC caveats).

## 4. Mass decorrelation / anti-sculpting  ← directly relevant to our assignment study
The paper flags that its loss correlates with reconstructed mass; our turn-on plots showed the heuristics
*sculpt* the QCD background upward. These are the standard fixes.
- **Louppe, Kagan, Cranmer**, adversarial decorrelation — **arXiv:1611.01046**.
- **Shimmin et al.**, adversarial mass decorrelation — **arXiv:1703.03507**.
- **Kitouni, Nachman, Weisser, Williams**, monotonic/robust decorrelation — **arXiv:2010.09745**.
- ★ **Kasieczka, Shih**, *DisCo* (distance-correlation decorrelation) — **arXiv:2001.05310**.
- **Klein, Golling** — **arXiv:2211.02486**; **Algren, Raine, Golling** — **arXiv:2307.05187**.

## 5. Real-time & FPGA machine learning / triggers  ← relevant to the hardware deployment + trigger study
- ★ **Duarte et al.**, *hls4ml: fast NN inference in FPGAs for particle physics* — JINST 13, P07027 — **arXiv:1804.06913**.
- ★ **Govorkova et al.**, *Autoencoders on FPGAs for real-time unsupervised new-physics detection at 40 MHz*
  — Nat. Mach. Intell. — **arXiv:2108.03986** (an autoencoder anomaly trigger on FPGA — closest analogue to what we deployed).
- CMS **AXOL1TL** and **CICADA** L1 anomaly triggers (anomaly detection in the CMS Level-1 trigger; see the
  CMS trigger public results / DP notes) — the production-scale version of the idea.
- Context for the trigger turn-on study: LHC L1/HLT jet & HT triggers and their turn-on/efficiency methodology
  (ATLAS/CMS trigger performance papers).

## 6. Physics target — RPV SUSY multijet & the experimental searches
Why the samples look the way they do (6–12 jets, no leptons/MET), and the searches this method extends.
- **Barbier et al.**, *R-parity-violation review* — Phys. Rept. 420, 1 — **arXiv:hep-ph/0406039**;
  **Mohapatra**, RPV review — **arXiv:1503.06478**; **Csáki, Grossman, Heidenreich**, baryon-number violation — **arXiv:1111.1239**.
- Multijet / pair-produced-resonance searches: ATLAS **arXiv:1710.07171**, **arXiv:1806.04030**,
  **arXiv:2307.14944**, and **arXiv:2301.03212** — the di-object/multijet bump-hunts Passwd-ABC aims to generalize.

## 7. Foundations & tooling (used in the pipeline)
- **Schroff, Kalenichenko, Philbin**, *FaceNet* triplet loss (the anti-collapse term) — **arXiv:1503.03832**.
- **Komiske, Metodiev, Thaler**, *Energy Mover's Distance* — **arXiv:1902.02346**, **arXiv:2004.04159**
  (the similarity metric Passwd-ABC benchmarks against).
- **Kingma, Ba**, *Adam* — **arXiv:1412.6980**.
- **Alwall et al.**, *MadGraph5_aMC@NLO* — **arXiv:1405.0301**; **Sjöstrand et al.**, *Pythia 8* — **arXiv:1410.3012**;
  **de Favereau et al.**, *Delphes 3* — **arXiv:1307.6346** (the generation/detector chain — and where the
  stable-neutralino sample bug lived).

---

### Suggested path
For the **method**: 0 → 1 (SPANet) → 2 (Set Transformer). For the **anomaly framing**: 3 (review + LHC Olympics)
→ 4 (DisCo). For **our extensions**: 4 (sculpting ↔ our assignment turn-ons) and 5 (hls4ml + Govorkova ↔ the FPGA build).

*IDs are given as published; the CMS L1 anomaly-trigger entries in §5 are best found via the CMS trigger public
results pages, as the primary references are detector-performance notes rather than a single arXiv paper.*
