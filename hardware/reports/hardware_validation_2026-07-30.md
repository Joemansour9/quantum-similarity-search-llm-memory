# Real-Hardware SWAP-Test Validation, 2026-07-30

## QUICK REFERENCE (checkpoint, verbatim core numbers — see full tests below for methodology)

**QPU quota remaining: 153s** (consumed 447/600, rolling 28-day window, checked at time of this
checkpoint).

**Gate-count vs. PCA/random gap, four qubit counts (Tests 9, 11, 13):**

| n_qubits | gates | PCA mean gap | Random mean gap | diff | p | Cohen's d |
|---|---|---|---|---|---|---|
| 4 | 31 | 0.1030 | 0.1515 | 0.0486 | 0.00001 | 1.791 |
| 5 | 42 | 0.1335 | 0.1579 | 0.0244 | 0.324 | 0.511 |
| 6 | 53 | 0.2341 | 0.2404 | 0.0063 | 0.807 | 0.124 |
| 8 | 73 | 0.2429 | 0.2387 | -0.0042 | 0.863 | -0.089 |

**Ranking-preservation, n=12 trials (Test 14):**
- Top-1 agreement (hardware matches simulator): **12/12**
- Full 4-way ranking agreement: **2/12**

---


Twenty jobs, fourteen conceptual tests, run on real IBM Quantum hardware (`ibm_fez`, `ibm_kingston`)
against the quantum-native agent-memory prototype (`q_encoder.py`/`q_graph.py`/`q_search.py`/
`agent_memory.py`), to check whether the simulator finding — **PCA-projected quantum retrieval
degrades in lockstep with classical cosine similarity under query noise, with no isolated
quantum-specific weakness, while random projection shows a large isolated quantum-specific
weakness** — holds on real hardware, not just `AerSimulator`.

Fixed test conditions throughout: hard 20-Newsgroups corpus (`benchmark_realdata_embeddings_hard.py`,
confusable comp.*/rec.autos+motorcycles/sci.electronics topics), n_qubits=4, PCA projection unless
noted, query-noise multiplier=1.2, corpus seed=0, all-MiniLM-L6-v2 embeddings (384-dim).
Account: IBM Quantum Open plan, rolling 28-day usage window (confirmed empirically, not a fixed
monthly reset — see "Quota" section).

## Quota state at completion

- `usage_consumed_seconds`: 447 / 600 (limit)
- `usage_remaining_seconds`: **153**
- `usage_period`: rolling 28-day window ending at query time (confirmed by two consecutive
  `service.usage()` calls whose start/end both shifted forward by the exact gap between calls) —
  quota recovers gradually as old usage ages out past 28 days, not via a hard reset date.
- Total consumed: 2+2+2+3+2+5+2+2+4+6+6+4+7+7+7+7+7+4 = **79s** across all twenty jobs.
- A separate all-time export (`all_time-workloads.csv`, 220 jobs incl. unrelated prior work)
  showed 1052s (17.53 min) of lifetime usage over a 55-day span — not 12 months, and no evidence
  anywhere (API or export) of a promotional threshold/bonus tier. Documented for the record; not acted on, since nothing confirmed such a mechanism exists.
- Wall-clock note: queue depth was a weak/unreliable predictor of actual wait time (a 9-pending
  job once took 1157s; two ~4-pending batches later took ~105s each) — real-time budgeting for
  future planning should expect this variance, not assume queue-depth ≈ wait-time.

## Circuit constructions used

1. **Ancilla/CSWAP** (`q_search.swap_test_circuit`) — standard SWAP test: 1 ancilla + n_qubits×2
   registers, H–CSWAP×n–H–measure(ancilla). 2n+1 qubits.
2. **Destructive/Bell-basis** (`q_search.destructive_swap_test_circuit`, added in this round of testing) —
   CNOT(a_i→b_i)+H(a_i) per qubit pair, measure both registers into one 2n-bit classical
   register, fidelity = E[(−1)^Σx_i·y_i] (`destructive_fidelity_from_counts`). 2n qubits, no
   ancilla. Validated against exact `Statevector` fidelity before use (diff ≤0.002 at 20k
   shots on RZZ-entangled states; diff=0.0001–0.0013 on the actual staged pairs below).

Both wired into `q_search.evaluate_similarity()` as `method="swap_test"` /
`method="destructive_swap_test"` — not just one-off scripts.

## Test 1 — PCA fidelity, CSWAP vs. destructive (same pair, same backend)

Pair: query = noisy `item_0_c0` (seed=0, q_idx=0, mult=1.2), candidate = `item_0_c0` itself.
Exact simulator fidelity for this pair: **0.7700** (statevector-exact; a `0.7701` label
propagated in some intermediate scripts — treat 0.7700 as authoritative, difference is
noise-level rounding, doesn't affect any conclusion).

| | Job ID | Backend | Transpiled depth | 2Q gates | Shots | Hardware fidelity | Sim | Δ |
|---|---|---|---|---|---|---|---|---|
| CSWAP | `d9l0mrqbr2fc73e812hg` | ibm_fez | 138 | 66 | 1024 | 0.5293 | 0.7700 | −0.241 |
| Destructive | `d9l17rbjf64c739isibg` | ibm_fez | 54 | 31 | 1024 | 0.6348 | 0.7700 | −0.135 |

**Finding:** ~half the 2-qubit gates → roughly half the degradation. Consistent with
gate-count-driven noise, not a stable/fixed hardware effect. n=1 pair, direction is suggestive
not proven.

## Test 2 — Random-projection fidelity (destructive only)

Same pair construction, `projection="random"` (seed=0 fit, same convention `AgentMemory` uses).

| Job ID | Backend | Gates | Shots | Hardware fidelity | Sim | Δ |
|---|---|---|---|---|---|---|
| `d9l17vrjf64c739isifg` | ibm_fez | 31 | 1024 | 0.1895 | 0.1793 | +0.010 |

**Finding: inconclusive by construction, not by result.** This pair's simulator fidelity was
already near the ~0 noise floor (0.18), leaving little dynamic range to detect additional
hardware degradation — unlike the PCA pair, which started at 0.77 and had room to fall. A
same-style test on a high-simulator-fidelity random-projection pair would be needed to
actually test whether random's isolated quantum-specific weakness (the headline synthetic/sim
finding) shows up on hardware. **Not done in this round of testing — flagged as the clearest follow-up.**

## Test 3 — Ranking preservation (1 query × 4 candidates, destructive, batched)

Query = noisy `item_0_c0`; candidates = `item_0_c0` (true match) + 3 distractors from other topics.

| Job ID | Backend | Total 2Q gates | Shots | Wall clock | Usage |
|---|---|---|---|---|---|
| `d9l7shabr2fc73e8ao40` | ibm_kingston | 124 (4×31) | 1024 | 84.6s | 3s |

| candidate | sim_fid | hw_fid |
|---|---|---|
| item_0_c0 | 0.7700 | 0.5918 |
| item_6_c1 | 0.1412 | 0.1152 |
| item_7_c1 | 0.0270 | 0.0293 |
| item_8_c1 | 0.0330 | 0.0293 |

Sim ranking: `[item_0_c0, item_6_c1, item_8_c1, item_7_c1]`
HW ranking: `[item_0_c0, item_6_c1, item_7_c1, item_8_c1]`

**Finding:** top-1 matched (True) — the retrieval-relevant outcome held. Full ranking didn't
(bottom two flipped), but that pair was a 0.006-fidelity statistical near-tie in the noise-free
simulator to begin with — well inside the 1024-shot noise floor, not evidence of a
ranking-scrambling hardware effect on pairs that actually matter.

## Test 4 — Mid-gate-count check (opt_level=0 vs. opt_level=3, same pair, same backend)

Attempted to get a third point on the Test-1 gate-count curve using the same destructive
circuit at a lower transpiler optimization level (no hybrid ancilla/destructive construction).

| opt_level | Job ID | Backend | 2Q gates | Hardware fidelity |
|---|---|---|---|---|
| 3 (from Test 3's item_0_c0 row) | `d9l7shabr2fc73e8ao40` | ibm_kingston | 31 | 0.5918 |
| 0 | `d9l820rhdfks73ckkco0` | ibm_kingston | 34 | 0.6074 |

**Finding: null result, reported as such.** 31→34 gates (opt_level 0–3 all collapse to
essentially the same decomposition on this small a circuit — 0,1,2,3 gave 34,31,31,31) produced
no resolvable trend; the 34-gate point landed *higher* than the 31-gate point, opposite the
expected direction, consistent with the ~0.04–0.05 backend/run-to-run variance already observed
swamping a 3-gate difference. The opt_level knob cannot produce a genuine CSWAP↔destructive
midpoint on a circuit this small — a real midpoint would need a different circuit-level lever,
not attempted in this round of testing.

## Test 5 — Multi-candidate retrieval benchmark (3 queries × 4 candidates, destructive, batched)

Queries = noisy `item_0_c0`/`item_1_c0`/`item_2_c0` (q_idx 0/1/2, same corpus/mult), each against
its own true match + 3 same distractor topics.

| Job ID | Backend | Total 2Q gates | Shots | Wall clock | Usage |
|---|---|---|---|---|---|
| `d9l842rjf64c739j5q4g` | ibm_kingston | 308 (range 15–31/circuit) | 1024 | 1156.7s | 5s |

| query | relevant | sim (rel) | hw (rel) | sim top-1 correct | hw top-1 correct |
|---|---|---|---|---|---|
| 0 | item_0_c0 | 0.7700 | 0.6289 | True | True |
| 1 | item_1_c0 | 0.3082 | 0.2441 | True | True |
| 2 | item_2_c0 | 0.5495 | 0.4609 | True | True |

**Recall@1: simulator 3/3, hardware 3/3.** All three margins held despite real degradation
(−0.14, −0.06, −0.09 respectively) — narrowest case (query 1) still had a ~2.5× gap between
the correct answer (0.244) and its closest hardware competitor (0.098). Notable side artifact:
`item_6_c1` in query 1 was noise-inflated from sim=0.014 to hw=0.061 (>4× relative, small in
absolute terms) — consistent with depolarizing noise pushing low fidelities toward a floor;
never came close to threatening the correct top-1.

## Test 6 — High-fidelity random-projection pair (closes the gap left by Test 2)

Test 2's random-projection pair had simulator fidelity already near the noise floor (0.18),
leaving no room to detect additional hardware degradation. This test searched for a
random-projection pair with high simulator fidelity to make a real test possible.

**Search:** scanned all 48 query indices (q_idx=0..47) in the hard corpus (seed=0, mult=1.2,
n_qubits=4, random projection), using the exact rng/projection-seed convention
`benchmark_query_noise_realdata_hard.py`'s actual sweep uses (sequential noise draws from one
generator seeded `seed+1`; projection fit seed=q_idx, matching `AgentMemory`). Mean fidelity
across all 48 was 0.269, median 0.212 — confirms random projection is *usually* low on this
corpus, consistent with every earlier finding — but the top hits were high: q_idx=27
(`item_27_c4`) gave **sim_fid=0.7798**, closely matched to the PCA pair's 0.7700, making it
the cleanest possible same-magnitude comparison. Validated locally (destructive, 20k shots,
local sim): 0.7795, diff=0.0003 from exact.

`ibm_fez` had spiked to 1965 pending jobs at submission time (vs. 58 on `ibm_kingston`) —
submitted on kingston instead, which is one of the two backends already used in this round of testing
(Tests 3–5), not a new untested one.

| Job ID | Backend | Gates | Shots | Hardware fidelity | Sim | Gap |
|---|---|---|---|---|---|---|
| `d9l8n22br2fc73e8bma0` | ibm_kingston | 31 | 1024 | 0.6152 | 0.7798 | 0.1646 |

**The correct comparison is NOT against Test 1's fez-based PCA gap (0.1352)** — different
backend, and this round of testing confirmed ~0.04–0.05 backend-to-backend variance, so that
comparison isn't apples-to-apples. The clean comparison is against **Test 3's kingston-based
PCA point** (`item_0_c0`, same backend, same 31 gates, comparable starting fidelity):

| | Backend | Gates | Sim | HW | Gap |
|---|---|---|---|---|---|
| PCA (`item_0_c0`, Test 3) | kingston | 31 | 0.7700 | 0.5918 | **0.1782** |
| Random (`item_27_c4`, Test 6) | kingston | 31 | 0.7798 | 0.6152 | **0.1646** |

**Finding: random's hardware-noise gap is not larger than PCA's — marginally smaller**
(0.1646 vs 0.1782, diff=0.0136), on the one same-backend, same-gate-count, matched-starting-
fidelity pair available. That 0.0136 difference is well inside single-sample (n=1) shot noise —
not proof of anything at scale, but a clean negative result on the specific question this
validation round was chasing, not an inconclusive one.

**Important distinction:** the original simulator finding (random shows an isolated
quantum-specific weakness) was about *query/embedding noise*, tested across the qubit and
query-noise sweeps earlier in this project. This test is a different axis — *physical hardware
gate noise* — and answers a different, hardware-specific question: whether random projection is
ALSO disproportionately fragile to real QPU noise on top of its known embedding-noise weakness.
On this one data point: no evidence of that. The two noise sources (embedding-space query noise
vs. physical gate noise) should stay analytically separate in any write-up.

## Test 7 — n=4-per-method statistical comparison (strengthens Test 6)

Test 6 established a single matched PCA-vs-random pair (n=1 each). This test added 3 more
pairs per method — all sim_fid≥0.5, same corpus/seed/mult/n_qubits convention, same backend
(ibm_kingston), gate counts confirmed identical (all 31) across all 6 new circuits, no
gate-count confound. Job `d9l9kb3jf64c739j7ni0` (6 PUBs), usage=~2s, wall_clock=73.1s.

| method | q_idx | key | sim | hw | gap |
|---|---|---|---|---|---|
| pca | 6 | item_6_c1 | 0.7034 | 0.5801 | 0.1233 |
| pca | 18 | item_18_c3 | 0.6787 | 0.5664 | 0.1123 |
| pca | 10 | item_10_c1 | 0.6552 | 0.5664 | 0.0888 |
| random | 3 | item_3_c0 | 0.8099 | 0.6777 | 0.1322 |
| random | 16 | item_16_c2 | 0.7289 | 0.5645 | 0.1645 |
| random | 28 | item_28_c4 | 0.6761 | 0.5254 | 0.1507 |

Combined with the two Test 3/Test 6 matched points (both kingston, 31 gates): **n=4 per method.**

| | n | mean gap | std | range |
|---|---|---|---|---|
| PCA | 4 | 0.1257 | 0.0379 | [0.0888, 0.1782] |
| Random | 4 | 0.1530 | 0.0153 | [0.1322, 0.1646] |

Welch's t-test: t=1.339, **p=0.252**. Cohen's d=0.947 (large nominal effect size, but unreliable
at n=4 — easily produced by noise alone at this sample size).

**Finding: not statistically distinguishable.** Random's mean gap is numerically higher, but
p=0.252 is far from significance, and reaching real significance at this effect size would need
roughly n≈15-20 per group. PCA's spread (std=0.038) is mostly driven by one outlier
(`item_0_c0`, gap=0.178); without it PCA's other 3 points (0.089–0.123) look tighter than
random's, which shows how fragile any "which group is more variable" read is here. Treat Test 6
+ Test 7 together as: **no evidence yet that random projection is more fragile to real hardware
gate noise than PCA, but the sample size is too small to rule it out either — genuinely open,
not resolved in either direction.**

## Test 8 — n≈18-per-method, multi-seed (resolves Test 7's p=0.252)

Test 7 (n=4/method) was underpowered. This test expanded the candidate pool using additional
**corpus seeds** (seed=1, seed=2 — confirmed via code inspection that `seed` only changes which
real, already-embedded documents get sampled from the fixed per-topic pool; it does NOT
re-embed or generate synthetic data), keeping the same 0.5+ simulator-fidelity
floor. PCA had abundant headroom (69 candidates by seed=2); random was the binding constraint
(27 candidates by seed=2, vs needing ~15-20) — both facts reported before proceeding.

Selected 14 new pairs per method (seeds 0-2, same range for both, to avoid a seed-range
confound), batched into 2 jobs of 14 (`d9l9uh0ii2cc73eh7bl0`,
`d9l9vbabr2fc73e8d8o0`), both on ibm_kingston, wall-clock 104.2s and 106.1s (much faster than
Test 5's 12-circuit job at 1156.7s under similar-looking queue depth — confirms queue depth is
an unreliable wait-time predictor, not that batch size alone drives wall-clock).

**Gate-count catch:** 7 of the 14 new PCA circuits transpiled to 15 gates instead of 31 (a
recurring near-zero RZZ rotation angle across most seed=1 PCA queries — angle-dependent optimizer
simplification, not a bug). All 14 new random circuits stayed at 31. This broke gate-count
matching for part of the new PCA batch, so results are reported both ways:

| | n | mean gap | std | range |
|---|---|---|---|---|
| **Full dataset** — PCA (18, includes 7 mismatched 15-gate) | 18 | 0.1123 | 0.0242 | [0.0572, 0.1782] |
| **Full dataset** — Random (18, all 31-gate) | 18 | 0.1515 | 0.0253 | [0.1092, 0.1891] |
| **Gate-matched only** (31 gates both) — PCA | 11 | 0.1109 | 0.0310 | [0.0572, 0.1782] |
| **Gate-matched only** (31 gates both) — Random | 18 | 0.1515 | 0.0253 | [0.1092, 0.1891] |

Full dataset: Welch t=4.746, **p<0.0001**, Cohen's d=1.582.
Gate-matched only: Welch t=3.666, **p=0.0018**, Cohen's d=1.436.

As a sanity check, the 7 mismatched 15-gate PCA circuits alone (which should have an *unfair
advantage* from having half the gates) were compared against the full 31-gate random group:
mean gap 0.1145 vs 0.1515, still p<0.0001. The significant difference is not an artifact of the
gate-count mismatch — it survives in the properly controlled subset.

**Finding: RESOLVED, not inconclusive.** With adequate sample size (n=11-18/method vs Test 7's
n=4), random projection shows a statistically significant, moderately-large (d≈1.4-1.6) larger
real-hardware gate-noise degradation gap than PCA. This is the first test in this document with
real statistical power behind it, and it now positively answers — rather than leaves open — the
question Test 2 originally couldn't resolve: **yes, random projection's known weakness also
shows up as extra fragility to physical QPU noise, on top of its separate embedding-noise
weakness found earlier in the project.**

## Test 9 — n=18 vs n=18, fully gate-matched (definitive number)

Test 8's headline p=0.0018 used an unequal-n gate-matched comparison (PCA n=11 vs random n=18) —
not yet a comparable range between groups. Rather than trust the mismatched-n result, this test
pulled 7 more PCA pairs and — this time — **pre-confirmed each candidate's gate count via
free transpilation against the real `ibm_kingston` backend before submitting anything**, so no
more quota was spent finding out a candidate collapsed to fewer gates after the fact. Of 16
candidates checked, 12 confirmed at exactly 31 gates (also surfaced a third collapse value, 23
gates, on 2 more candidates — confirms this angle-dependent effect isn't binary/one-off). Took
the top 7 by simulator fidelity, re-confirmed 31 gates again at submission time (asserted in
code, not assumed), submitted as one 7-PUB batch (`d9la4sgii2cc73eh7j3g`, wall-clock 91.9s).

Combined with Test 8's 11 gate-matched PCA points: **PCA n=18, Random n=18, every single circuit
confirmed at exactly 31 two-qubit gates — no exceptions, no mixed weighting.**

| | n | mean gap | std | range |
|---|---|---|---|---|
| PCA | 18 | **0.1030** | 0.0288 | [0.0572, 0.1782] |
| Random | 18 | **0.1515** | 0.0253 | [0.1092, 0.1891] |

**Welch t=5.372, p=0.00001, Cohen's d=1.791.** More significant than Test 8's unequal-n figure
(p=0.0018), not less — the effect strengthened under the fully symmetric, fully gate-matched
comparison, which is the direction that should make you trust it more, not less. **This is the
number for the write-up**, superseding both Test 7 (n=4, p=0.252, inconclusive) and Test 8's
n=11-vs-18 intermediate figure.

## Test 10 — n_qubits=8 replication (does Test 9's effect hold at a different qubit count?)

Established the actual transpiled gate count for n_qubits=8 destructive circuits on ibm_kingston
(73 gates — verified, not assumed, same discipline as Test 9's 31). Random projection was even
more constrained than at n_qubits=4: only 5 candidates ≥0.5 sim fidelity after 4 corpus seeds
(192 pairs scanned), needing 8 seeds (384 pairs) to reach 10. PCA had abundant headroom (118
candidates by seed=7). Selected n=8/method, all pre-confirmed and re-confirmed at exactly 73
gates (12/14 PCA and 10/10 random candidates cleared the pre-check), batched as one 16-PUB job
(`d9lac03jf64c739j8i50`, wall-clock 87.0s).

| | n | mean gap | std | range |
|---|---|---|---|---|
| PCA | 8 | 0.2429 | 0.0275 | [0.2018, 0.2845] |
| Random | 8 | 0.2387 | 0.0612 | [0.1490, 0.3162] |

**Welch t=-0.177, p=0.863, Cohen's d=-0.089 — no effect, not even trending in the original
direction.** Both groups' mean gaps roughly doubled versus n_qubits=4 (≈0.10-0.15 → ≈0.24),
consistent with the larger gate count (73 vs 31) driving substantially more degradation for
both methods roughly equally, swamping whatever structural advantage PCA had at 31 gates.

## Test 11 — n_qubits=6 intermediate depth (gradient or step function?)

With 31-gate-effect-present and 73-gate-effect-absent as the only two points, tested whether the
transition is gradual or sharp. Established n_qubits=6's real transpiled gate count on kingston:
**53 gates** (a genuine midpoint between 31 and 73). Found 12 random candidates ≥0.5 by corpus
seed=4 (111 PCA candidates by the same point). Selected n=8/method, all confirmed at exactly 53
gates (12/12 random, 11/14 PCA cleared pre-check on first pass), batched as one 16-PUB job
(`d9lal32br2fc73e8e0sg`, wall-clock 305.1s — the slowest of the three qubit-count tests, another
data point on queue-time unpredictability).

| | n | mean gap | std | range |
|---|---|---|---|---|
| PCA | 8 | 0.2341 | 0.0544 | [0.1350, 0.2971] |
| Random | 8 | 0.2404 | 0.0466 | [0.1670, 0.3290] |

**Welch t=0.249, p=0.807, Cohen's d=0.124 — clean null, statistically indistinguishable from
n_qubits=8's null (p=0.863).**

**Combined three-qubit-count picture — NOT a smooth gradient:**

| n_qubits | gates | PCA mean gap | Random mean gap | diff | p | Cohen's d |
|---|---|---|---|---|---|---|
| 4 | 31 | 0.1030 | 0.1515 | 0.0486 | **0.00001** | 1.791 |
| 6 | 53 | 0.2341 | 0.2404 | 0.0063 | 0.807 | 0.124 |
| 8 | 73 | 0.2429 | 0.2387 | −0.0042 | 0.863 | −0.089 |

The effect is not "present but weaker" at n_qubits=6 — it's already gone, as fully gone as at
n_qubits=8. More striking: the mean gaps themselves roughly DOUBLE from n=4 to n=6 (≈0.10-0.15 →
≈0.23-0.24) but go essentially FLAT from n=6 to n=8 (≈0.23-0.24 → ≈0.24-0.24) for both methods.
This points to a **threshold/saturation effect concentrated somewhere in the 31-53 gate range**,
not a gradient spread across 31-73 — past that threshold both circuit types are already
noise-dominated enough that PCA's structural advantage disappears, and further gate count barely
matters. Where exactly between 31 and 53 gates the transition happens is untested.

## Test 12 — Strengthened ranking/retrieval tests (n=1 → n=12 trials)

Tests 3 and 5's "retrieval decisions survive hardware noise" claims rested on n=1 each (1
ranking trial, 1 three-query retrieval trial — 4 total query-candidate-set trials). Added 8 more
queries, same construction as the originals (PCA, n_qubits=4, mult=1.2, corpus seed=0,
destructive circuit, ibm_kingston, relevant + 3 distractors from other topics), spread across 6
different topics and **not fidelity-filtered** (unlike Tests 9-11 — retrieval correctness, not
fidelity magnitude, is what's being tested here, so the ≥0.5 floor used to avoid floor effects
in the gap comparisons doesn't apply). 32 new circuits, batched into 2×16
(`d9lavtibr2fc73e8ed5g`, `d9lb080ii2cc73eh8h7g`), wall-clock 41.3s and 72.4s.

Gate counts varied naturally per query (15/23/31, all 4 candidates within a query sharing that
query's count since the query-side circuit dominates the collapse) — this matches Test 5's
original methodology exactly (which reported the same 15-31 range unprompted), not a new
inconsistency introduced by relaxing the fidelity floor.

All 8 new queries had unambiguous sim top-1 (relevant candidate ranked first), spanning a real
difficulty range: `item_4_c0` (wide margin, sim 0.581 vs 0.076 runner-up) down to `item_25_c4`
(tightest margin of the whole test round, sim 0.068 vs 0.014 runner-up).

**Result: hardware top-1 matched simulator top-1 on all 8 new queries — including `item_25_c4`,
the tightest margin tested.** Combined with the original 4 trials: **Recall@1 = 12/12 on both
simulator and hardware.** Zero retrieval-decision flips across every ranking/retrieval trial run
this round of testing.

## Test 13 — n_qubits=5, the unresolved window between 31 and 53 gates

Established n_qubits=5's real transpiled gate count on kingston: **42 gates** (verified, matching
the predicted ~40-45 range). Random projection needed 2 corpus seeds to reach n=13 candidates
≥0.5 (45 PCA candidates by the same point). Selected n=8/method; PCA needed a fill-in round (7/14
initial candidates collapsed to fewer gates, mostly 32), pulled more until 8 confirmed at exactly
42. Batched as one 16-PUB job (`d9lblv8ii2cc73eh9a40`), wall-clock 2432.1s (the slowest job of
the test round — another queue-unpredictability data point).

| | n | mean gap | std | range |
|---|---|---|---|---|
| PCA | 8 | 0.1335 | 0.0440 | [0.0575, 0.2126] |
| Random | 8 | 0.1579 | 0.0511 | [0.0702, 0.2343] |

Welch t=1.022, p=0.324, Cohen's d=0.511 — not significant, but **two separate things move at
different rates here, worth keeping distinct:**

1. **PCA-vs-random difference**: fading gradually, not stepping. d goes 1.791 (31g) → 0.511 (42g)
   → 0.124 (53g) → -0.089 (73g) — a real gradient in the effect's significance/size.
2. **Absolute degradation magnitude**: does NOT move gradually. n_qubits=5's mean gaps (0.13-0.16)
   sit much closer to n_qubits=4's (0.10-0.15) than to n_qubits=6's (0.23-0.24) — the noise-
   magnitude saturation jump is concentrated specifically **between 42 and 53 gates**, sharper
   than the effect-size gradient above.

Full four-point picture:

| n_qubits | gates | PCA mean gap | Random mean gap | diff | p | Cohen's d |
|---|---|---|---|---|---|---|
| 4 | 31 | 0.1030 | 0.1515 | 0.0486 | 0.00001 | 1.791 |
| 5 | 42 | 0.1335 | 0.1579 | 0.0244 | 0.324 | 0.511 |
| 6 | 53 | 0.2341 | 0.2404 | 0.0063 | 0.807 | 0.124 |
| 8 | 73 | 0.2429 | 0.2387 | -0.0042 | 0.863 | -0.089 |

## Test 14 — Ranking-preservation and CSWAP-vs-destructive strengthened beyond n=1

**Ranking-preservation, n=1 → n=12 (zero new quota — reused Tests 3/5/12's already-collected
raw per-candidate sim/hw data).** Parsed all 12 four-candidate trials from this round's
retrieval-style tests and computed sim-vs-hardware ranking agreement directly:
- **Top-1 agreement: 12/12** (consistent with Test 12's Recall@1 finding).
- **Full 4-way ranking agreement: 2/12.** Top-1 is robust to hardware noise; the exact ordering
  among lower-fidelity distractors is not — most trials reshuffle somewhere below the top pick.
  This is the more precise claim to make going forward: retrieval survives, fine-grained ranking
  among near-tied/low-fidelity candidates does not.

**CSWAP-vs-destructive, n=1 → n=4 paired.** Reused 4 already-validated PCA pairs
(`item_0_c0`/`item_6_c1`/`item_18_c3`/`item_10_c1`, n_qubits=4, mult=1.2), built and transpiled
BOTH circuit constructions for each on `ibm_fez` (matching Test 1's original backend to avoid
the backend-variance confound), confirmed clean at 31/66 gates with no collapses, batched as one
8-PUB job (`d9lcqtjjf64c739jbkrg`), wall-clock 8.3s.

| | mean gap | std |
|---|---|---|
| Destructive | 0.1320 | 0.0304 |
| CSWAP | 0.2536 | 0.0473 |

All 4 pairs showed CSWAP degrading more (paired diffs 0.090-0.156, always positive). **Paired
t-test: t=8.933, p=0.00296.** Confirms Test 1's n=1 finding (0.241 vs 0.135) with real
statistical backing — destructive's advantage over CSWAP is consistent, not a one-pair fluke.

## Headline synthesis

- **Gate count tracks degradation, but the PCA-vs-random effect itself is gate-count/depth-
  dependent, not a fixed property of the two projection methods (Tests 9, 11, 13).** Full picture
  across four qubit counts: d=1.791 (31g, p=0.00001) → d=0.511 (42g, p=0.324) → d=0.124 (53g,
  p=0.807) → d=-0.089 (73g, p=0.863). **Two distinct phenomena, not one**: the PCA-vs-random
  effect fades gradually across this range, but the absolute noise MAGNITUDE (mean gap for both
  methods) jumps sharply between 42 and 53 gates specifically, not gradually across 31-73. **This
  means the headline n_qubits=4 finding should not be generalized to other qubit counts/gate
  depths without re-testing** — it's real at the tested depth, fading by 42 gates, and gone by 53.
- **No retrieval-relevant hardware failure observed anywhere in this round of testing, across 12 Recall@1
  trials (Test 12) — but full-ranking fidelity is a separate, weaker claim (Test 14).** Every
  top-1 decision survived real hardware noise, including the tightest margin tested in this round
  (`item_25_c4`, sim 0.068 vs 0.014 runner-up): **Recall@1 = 12/12.** However, computed from the
  same 12 trials' raw data: **full 4-candidate ranking agreement (hardware matches simulator on
  every position, not just top-1) is only 2/12.** State these separately going forward — "rankings
  survive hardware noise" is true for the decision that matters (top-1) and false for fine-grained
  ordering among distractors.
- **CSWAP-vs-destructive gate-count advantage confirmed beyond n=1 (Test 14).** Paired
  comparison, same 4 pairs through both circuit constructions, same backend: destructive mean gap
  0.132 vs CSWAP's 0.254, paired t=8.933, p=0.00296, all 4 pairs consistent in direction. Test 1's
  original single-pair finding (0.135 vs 0.241) wasn't a fluke.
- **Random projection's real-hardware-noise question: RESOLVED (Test 9, definitive), after
  passing through n=1 (Test 6, looked like a clean null) → n=4 (Test 7, p=0.252, genuinely
  inconclusive) → n=11-vs-18 unequal, gate-matched (Test 8, p=0.0018) → n=18-vs-18, every circuit
  confirmed at exactly 31 gates (Test 9, p=0.00001, Cohen's d=1.791).** Random projection shows a
  statistically significant, large real-hardware gate-noise degradation gap beyond PCA's — mean
  gap 0.1515 vs 0.1030 — and the effect got MORE significant, not less, as the comparison became
  more symmetric and more tightly gate-controlled, which is the pattern you want to see if it's
  real rather than an artifact. This answers, rather than leaves open, whether random's known
  weakness extends from embedding-space noise (the original simulator finding) to physical QPU
  gate noise — it does. **Sample-size progression here is itself a useful cautionary note**: n=1
  and n=4 both would have supported the wrong or no conclusion; it took n=18-vs-18 with gate count
  pre-verified before submission (not just checked after) to reach a defensible number.
- Noise sweep (2–3 points × n=1, per the original plan) was **not run** — out of scope for this
  validation round; Tests 1–8 covered fidelity-vs-gate-count, ranking, retrieval, and the
  statistically powered PCA-vs-random comparison instead.

## Files added

- `q_search.py` — added `destructive_swap_test_circuit`, `destructive_fidelity_from_counts`,
  `destructive_swap_test_fidelity_from_circuits`, `destructive_swap_test_fidelity`; wired into
  `evaluate_similarity(method="destructive_swap_test")`.
- `scratch_hw_submit*.py` (9 files) — one-shot submission scripts per job above, each reads a
  pre-built/pre-transpiled circuit from a matching `scratch_hw_circuit*.pkl`/`scratch_hw_*.pkl`.
- `scratch_hw_result_*.txt` — raw stdout logs for each job, referenced above.
- `scratch_hw_ngroup2_rows.pkl` — full per-pair dataset for Test 8 (method, corpus seed, q_idx,
  key, gate count, sim/hw fidelity, gap) — the source data behind the final t-test.
