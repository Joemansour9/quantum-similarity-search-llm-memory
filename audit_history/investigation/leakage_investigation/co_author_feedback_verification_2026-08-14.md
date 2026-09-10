# Co-Author Feedback Verification — Two Checkable-Fact Concerns — 2026-08-14

Both tasks investigated rigorously, code traced precisely, empirical checks run against this
project's own real corpora/pipeline (no new embeddings, no QPU quota — Task 1 is pure classical
math/code, Task 2 is classical simulation only). **Bottom line up front: Task 1's concern is
correct and, for the chain topology specifically, stronger than stated (genuinely efficient, not
just "closed form"). Task 2's concern is also correct and material — it measurably inflates
Result 1's PCA-vs-random gap at several qubit counts. Originally reported here as "doesn't touch
n_qubits=4" based on the 480-item corpus alone — extending the check to the other three
qubit-sweep sources (2026-08-14, second pass) overturned that mitigating claim: at the 48-item
hard corpus, the corpus underlying Section 5.1's own headline numbers and all of
`hardware_validation_2026-07-30.md`'s hardware work, n_qubits=4 shows the LARGEST leakage effect
found anywhere in this investigation, not the smallest. See the extended section below — this is
a correction, not a footnote.**

## Task 1 — Classical simulability of the fidelity kernel: CONFIRMED, and stronger than asked

**Derivation.** After the Ry rotation layer, the state is a product state
`|φ(θ)⟩ = ⊗ᵢ [cos(θᵢ/2)|0⟩ + sin(θᵢ/2)|1⟩]`. `RZZ(φ) = exp(-iφ/2·Z⊗Z)` is **diagonal** in the
computational basis: `RZZ(φ)|bᵤb_v⟩ = exp(-iφ/2·zᵤz_v)|bᵤb_v⟩` where `z = +1` for bit 0, `-1` for
bit 1. Since every entangling gate in this construction is such a diagonal RZZ, the full encoded
state is `|ψ(θ)⟩ = D(θ)|φ(θ)⟩`, where `D(θ)` is diagonal with
`D(θ)|b⟩ = exp(-i/2·Σ_{(u,v)∈edges} θᵤθ_v·zᵤ(b)z_v(b))|b⟩`. Substituting into the fidelity:

```
⟨ψ_q|ψ_c⟩ = Σ_b ⟨φ(α)|b⟩⟨b|φ(β)⟩ · exp(-i/2·Σ_edges (β_uβ_v - α_uα_v)·z_u(b)z_v(b))
```

— an **exact closed-form sum over 2ⁿ basis states**, built entirely from cosines/sines of the two
angle vectors and the edge list. No gate is simulated, no unitary matrix multiplied, no quantum
circuit built at all — this is a direct algebraic consequence of every entangler in the
construction being diagonal.

**For the chain specifically**, this sum has the structure of a 1D transfer-matrix contraction
(a path graph has treewidth 1), computable via a simple qubit-by-qubit DP recursion in
**O(n_qubits) time** — not just "closed form," but genuinely **efficient**, sidestepping the
exponential Hilbert-space dimension entirely (implemented as `chain_transfer_matrix_overlap()`
in `scratch_verify_classical_simulability.py`).

**Empirical verification** (`scratch_verify_classical_simulability.py`, run against the real
480-item corpus and this project's actual `fit_pca_projection_scale`/`fit_projection_scale`
pipeline, comparing against `q_search.statevector_fidelity_from_circuits` — the exact Qiskit
ground truth used throughout the paper):

| test | topology | n cases | max \|closed-form − Qiskit\| |
|---|---|---|---|
| General brute-force sum | chain (n_qubits 4,6,8,10; PCA+random; mult 0.2/1.2/4.0) | 72 | 3.6e-16 |
| O(n) transfer-matrix | chain (same 72 cases) | 72 | 5.6e-16 |

All deviations are at floating-point noise level (~1e-16) — **exact match, not approximate**.

**Correction (2026-09-10): the correlation-derived-topology row above has been removed.** This
entry originally reported a third row here — a general brute-force check against correlation-derived
topologies (n_qubits 4,6,8,10; random projection; 36 cases; up to 10 edges on 10 qubits; max
deviation 5.6e-16). `scratch_verify_classical_simulability.py` was modified on 2026-08-27 (13 days
after this entry was first written) to a version whose correlation-topology check covers only
n_qubits 4/6/8 at a single fixed multiplier, rather than the n_qubits 4/6/8/10 × three-multiplier
sweep the chain-topology row above still uses. Run as it exists today, at its threshold=0.3, this
narrower check produces zero cases with any correlation-derived edge at all — for both PCA- and
random-projected data — so the 36-case result is not reproducible from the script currently on
disk. No earlier version of the script survives to check directly (no version control is in place
for this project), so whether the original 36-case run used a materially different threshold, a
wider n_qubits/multiplier sweep, or both cannot be determined further than this. The general
closed-form derivation below (which holds for any edge set as an algebraic consequence of every
entangler being diagonal, independent of which topologies happen to be empirically tested) is
unaffected by this; what is no longer independently confirmed is the specific empirical exercise
of that derivation on dense correlation-derived graphs.

**Does this hold beyond the chain?** As a matter of derivation, yes: the *exact closed-form sum*
holds for **any** edge set, a direct algebraic consequence of every entangler in the construction
being diagonal, independent of topology. Whether this also holds up empirically on dense,
non-planar correlation-derived graphs is not currently verified (see correction above). One
structural note that does still hold: `KnowledgeGraph.from_correlation` can **never** produce
edges when fed PCA-projected data, because distinct PCA components are orthogonal by construction
(their correlation is exactly 0, confirmed directly — max\|corr\|=0.000 across n_qubits 4/6/8).
What does *not* generalize, regardless of correlation-topology verification status, is the
**efficiency**: the O(n) DP specifically exploits the chain's path structure (treewidth 1). A
general/dense correlation graph would need a tensor-network contraction whose cost scales with the
graph's treewidth — not guaranteed polynomial for an arbitrary graph, though still expressible in
exact closed form.

**Verdict: this co-author feedback is correct, and for the chain topology it is stronger than
stated** — this isn't merely "can be computed in closed form," it is **classically simulable in
polynomial time**, which undercuts any claim that this specific encoding (chain topology, RZZ
entangler, statevector-fidelity readout) requires a quantum computer to evaluate at all. This
should be flagged prominently wherever the paper frames the chain-topology fidelity kernel as a
"quantum" computation rather than an efficiently-classically-simulable one — the actual quantum
resource requirement in this pipeline lives elsewhere (e.g., real SWAP-test/destructive-circuit
execution on hardware, which is a separate, genuinely non-classically-simulated measurement
process — not the same claim as this one, which is specifically about the fidelity *value* being
computable classically ahead of time for the chain).

## Task 2 — PCA-fit query leakage: CONFIRMED and material, but doesn't touch n_qubits=4

**Trace.** Grepped every call site of `fit_pca_projection_scale` across the entire codebase
(library code and every benchmark/sweep/hardware script written this project). Universally,
without exception:

```
full_batch = np.vstack([query_vec[None, :], candidate_batch])
R, lo, hi = fit_pca_projection_scale(full_batch, n_qubits)
```

This traces back to `agent_memory.py`'s `AgentMemory.query()` (lines 205-206), which every
benchmark script's `evaluate_retrieval()` call routes through — `benchmark_qubit_sweep_realdata.py`,
`benchmark_qubit_sweep_realdata_hard.py`, `benchmark_query_noise_realdata.py`,
`benchmark_query_noise_realdata_hard.py`, and `scratch_large_corpus_sweep.py` (Result 1/2 at 480
items) all inherit this behavior identically, since none of them bypass `AgentMemory.query()`'s
internal fitting. **Confirmed: yes, the query vector itself (the noisy copy being evaluated) is
included in the batch used to fit PCA's principal components, for every single query, throughout
Sections 5.1/5.2 and their real-corpus counterparts, with no exception found.**

This is a genuine, PCA-specific concern — **not** shared by random projection, whose matrix `R`
is drawn purely from `(seed, n_qubits, embedding_dim)` and never touches the data at all (per
`agent_memory.py`'s own docstring, confirmed independently by the empirical result below: random's
recall is unchanged when the query is excluded from fitting, at 5 of 6 tested qubit counts, with
the sixth a +0.05 blip consistent with ordinary sampling noise, not leakage — since removing the
query can only shift `lo, hi` slightly for random, never the projection directions `R` itself).

**Leak-free rerun** (`scratch_verify_pca_leakage.py`): refit PCA/random on `candidate_batch`
alone (query excluded entirely from the fit), then project the held-out query through the
resulting fixed `R`/scale — mirroring `scratch_large_corpus_sweep.py`'s exact setup (480-item
corpus, mult=1.2, n_qubits∈{4,6,8,10,12,14}, n=20/cell, `rng=default_rng(1)`), compared directly
against the already-archived leaky numbers in `scratch_large_corpus_qubit_sweep.pkl`:

| n_qubits | PCA leaky | PCA leak-free | PCA Δ | random leaky | random leak-free | random Δ | gap leaky | gap leak-free | **gap Δ** |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.250 | 0.250 | 0.000 | 0.050 | 0.050 | 0.000 | 0.200 | 0.200 | **0.000** |
| 6 | 0.550 | 0.400 | **-0.150** | 0.000 | 0.000 | 0.000 | 0.550 | 0.400 | **-0.150** |
| 8 | 0.800 | 0.750 | -0.050 | 0.050 | 0.100 | +0.050 | 0.750 | 0.650 | **-0.100** |
| 10 | 0.900 | 0.850 | -0.050 | 0.150 | 0.150 | 0.000 | 0.750 | 0.700 | **-0.050** |
| 12 | 0.950 | 0.950 | 0.000 | 0.200 | 0.200 | 0.000 | 0.750 | 0.750 | **0.000** |
| 14 | 0.900 | 0.900 | 0.000 | 0.350 | 0.350 | 0.000 | 0.550 | 0.550 | **0.000** |

**Verdict: this co-author feedback is correct and material, but doesn't overturn the finding.**
Removing the leakage:
- **Never increases** PCA's recall at any qubit count (every non-zero PCA Δ is negative — exactly
  the direction leakage-inflation predicts; there is no case where the "cleaner" fit did better),
  while random's Δ is 0 at 5/6 counts and the one exception (+0.05 at n_qubits=8) is within
  ordinary single-run sampling noise for random projection, whose fit doesn't depend on the query.
- **Measurably shrinks the PCA-vs-random gap at n_qubits=6, 8, 10** — by 0.150, 0.100, and 0.050
  absolute respectively (up to 27% relative at n_qubits=6). This is a real methodological flaw:
  Result 1's reported gap at those three qubit counts is partly an artifact of PCA getting to
  fit its principal components on a batch that includes the exact point being evaluated, for
  every query, individually — an advantage random projection cannot receive or benefit from by
  construction.
- **Leaves the gap completely unchanged at n_qubits=4, 12, and 14 — ON THIS 480-ITEM CORPUS
  SPECIFICALLY.** This was initially reported as good news for n_qubits=4 (the setting used for
  the query-noise dissociation study, corpus-size-transition work, and every real-hardware batch
  this check). **That framing does not generalize — see "Does the 480-item pattern hold at
  other corpus sizes?" below, which found the opposite result at the 48-item corpus.** Left here
  unedited, struck through in spirit, so the correction is visible rather than quietly smoothed
  into the original text.
- **Even in the affected range, PCA still massively outperforms random leak-free** (gaps of
  0.40-0.75 remain, vs. randoms's near-floor recall) — the qualitative "PCA over random" finding
  survives; only the reported *magnitude* at 3 of 6 qubit counts in the original Result 1 qubit
  sweep needs correcting.

**Scope caveat:** this is a single-seed rerun (n=20/cell, one RNG stream, matching the
original sweep's own convention exactly) rather than a multi-seed statistically-powered
comparison — appropriate for a representative-subset scope, but the exact
magnitudes (0.150/0.100/0.050) should be read as a real, directionally-confirmed effect rather
than a precisely-bounded one. A multi-seed version of this same leak-free rerun would be the
natural next step if a defensible corrected number for Result 1's qubit-sweep table is needed for
publication.

## Task 2, extended (2026-08-14) — does the 480-item pattern hold at other corpus sizes? NO

Extended the same leaky-vs-leak-free rerun to the three qubit-sweep sources not yet checked:
the synthetic corpus (Table 1), the real-easy corpus (Table 2), and the real-hard 48-item corpus
(Section 5.1's own prose numbers, 0.96-0.98 PCA / 0.06-0.42 random). Each was recomputed **both**
leaky and leak-free fresh within the same run (`scratch_verify_pca_leakage_remaining.py`), using
each source's own n_seeds=5/n_queries=10 (n=50/cell) convention exactly, rather than trusting old
archived leaky numbers — necessary for the 48-item corpus specifically, since its archived table
predates a pool-size change (`_POOL_PER_CATEGORY` 60→65) that would otherwise confound the
comparison for reasons unrelated to leakage (same caveat noted in
`pca_vector_classical_baseline_2026-08-04.md`).

**Synthetic corpus (Table 1, 24 items, d=32, cluster_spread=2.0):**

| n_qubits | PCA leaky | PCA leak-free | PCA Δ | gap leaky | gap leak-free | gap Δ |
|---|---|---|---|---|---|---|
| 4 | 0.640 | 0.560 | **-0.080** | 0.140 | 0.040 | **-0.100** |
| 6 | 0.800 | 0.860 | +0.060 | 0.180 | 0.200 | +0.020 |
| 8 | 0.960 | 0.980 | +0.020 | 0.280 | 0.300 | +0.020 |
| 10 | 0.980 | 0.960 | -0.020 | 0.220 | 0.180 | -0.040 |
| 12 | 1.000 | 0.960 | -0.040 | 0.160 | 0.100 | -0.060 |
| 14 | 1.000 | 0.960 | -0.040 | 0.160 | 0.100 | -0.060 |

**Real-easy corpus (Table 2, 24 items, MiniLM, 8 maximally-distinct topics):**

| n_qubits | PCA leaky | PCA leak-free | PCA Δ | gap leaky | gap leak-free | gap Δ |
|---|---|---|---|---|---|---|
| 4 | 1.000 | 0.980 | -0.020 | 0.860 | 0.860 | 0.000 |
| 6 | 1.000 | 1.000 | 0.000 | 0.760 | 0.800 | +0.040 |
| 8 | 1.000 | 1.000 | 0.000 | 0.700 | 0.640 | -0.060 |
| 10 | 1.000 | 1.000 | 0.000 | 0.660 | 0.680 | +0.020 |
| 12 | 1.000 | 1.000 | 0.000 | 0.640 | 0.640 | 0.000 |
| 14 | 1.000 | 1.000 | 0.000 | 0.620 | 0.580 | -0.040 |

**Real-hard 48-item corpus (Section 5.1's own numbers):**

| n_qubits | PCA leaky | PCA leak-free | PCA Δ | gap leaky | gap leak-free | gap Δ |
|---|---|---|---|---|---|---|
| **4** | **0.880** | **0.660** | **-0.220** | **0.720** | **0.480** | **-0.240** |
| 6 | 0.880 | 0.920 | +0.040 | 0.720 | 0.800 | +0.080 |
| 8 | 0.920 | 0.940 | +0.020 | 0.640 | 0.720 | +0.080 |
| 10 | 0.980 | 1.000 | +0.020 | 0.640 | 0.620 | -0.020 |
| 12 | 0.980 | 1.000 | +0.020 | 0.640 | 0.620 | -0.020 |
| 14 | 1.000 | 1.000 | 0.000 | 0.660 | 0.580 | -0.080 |

**None of the three repeats the 480-item pattern, and each fails differently:**

- **Real-easy corpus: no meaningful effect, but for a structural reason unrelated to leakage.**
  PCA is already at the 1.000 recall ceiling at every qubit count in both conditions — there is
  no room for leakage to show up regardless of whether it's present, so this corpus is simply
  uninformative about the question, not evidence of "unaffected."
- **Synthetic and real-hard-48 both show *mixed-sign* deltas — leak-free sometimes does BETTER
  than leaky (e.g. synthetic n_qubits=6/8: +0.060/+0.020; real-hard-48 n_qubits=6/8/10/12: all
  positive).** This contradicts the clean "leakage can only inflate PCA, never hurt it" pattern
  that held uniformly at 480 items (every non-zero Δ there was negative). The mechanism is
  straightforward: PCA components are fit via SVD on the batch, and at large N (480 items) one
  extra point (the query) is a ~0.2% perturbation to that SVD — negligible, so removing it can
  only ever relax whatever small inflation it caused (monotonic). At small N (24-48 items), the
  query is a 2-4% perturbation to the SVD — large enough to meaningfully shift which directions
  PCA picks as top components, which can happen to help *or* hurt that specific query's own
  retrieval, seed-by-seed. **This is a qualitatively different regime, not a smaller version of
  the same effect.**
- **Most consequential: n_qubits=4 on the 48-item corpus shows the single LARGEST leakage effect
  found in this entire investigation** — a 0.220 absolute drop in PCA recall (0.880→0.660, 25%
  relative) and a 0.240 absolute shrinkage in the PCA-random gap (0.720→0.480, 33% relative) —
  compared to *zero* change at n_qubits=4 on the 480-item corpus. Rough statistical-power check:
  at n=50/cell here, SE≈0.046, so this -0.220 shift is ~4.8 SEs — not plausibly noise. (For
  comparison, the 480-item corpus's largest reported effect, -0.150 at n_qubits=6 with n=20/cell,
  SE≈0.111, is only ~1.35 SEs — worth reading with more caution than it was given above, on
  reflection; it's directionally consistent with 8/10's smaller-but-same-sign effects, but weaker
  standing alone than the 48-item n_qubits=4 result.)

**Why this matters beyond the numbers:** n_qubits=4 is the setting used for
`hardware_validation_2026-07-30.md` in its entirety (all 20 real-hardware jobs), for Section 5.1's
own prose ("0.96-0.98 PCA / 0.06-0.42 random"), and for the query-noise dissociation study at the
48-item scale. The original framing in this document — "doesn't touch n_qubits=4" — was accurate
only for the specific 480-item corpus it was tested on, and should not have been stated as if it
were a property of n_qubits=4 in general. **The corrected picture: leakage's effect on the
PCA-vs-random gap is corpus-size-dependent and does not have a single "safe" qubit count that
holds across corpus sizes — it needs to be checked per corpus, not assumed to transfer.**

## Files

- `scratch_verify_classical_simulability.py` — Task 1: derivation implementation
  (`closed_form_overlap_bruteforce`, `chain_transfer_matrix_overlap`) and empirical comparison
  against Qiskit statevector fidelity, chain and correlation topologies.
- `scratch_verify_pca_leakage.py` / `scratch_pca_leakage_results.pkl` — Task 2: leak-free qubit
  sweep rerun and comparison against the archived leaky numbers, 480-item corpus.
- `scratch_large_corpus_qubit_sweep.pkl` — reused unchanged, the original leaky Result 1 numbers
  from `large_corpus_generalization_2026-07-30.md`.
- `scratch_verify_pca_leakage_remaining.py` / `scratch_pca_leakage_remaining_results.pkl` — Task 2
  extended: leaky-vs-leak-free recomputed fresh (both conditions, same run) for the synthetic,
  real-easy, and real-hard-48-item corpora.
