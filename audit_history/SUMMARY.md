# Audit History — Summary

This folder documents a methodology issue found in this project's quantum retrieval pipeline, how
large its effect was, and how it was corrected. Every result reported in the current paper reflects
the corrected methodology; this material is retained for transparency and reproducibility of the
correction itself, not as a live caveat about the paper's current numbers.

## The issue

To encode a document or a search query into a quantum circuit, the pipeline reduces its
high-dimensional embedding to a handful of numbers, one per qubit. One reduction method is PCA,
which selects the directions that capture the most variation in whatever data it is shown. The
pipeline's original implementation fit that PCA projection on a batch that included the query
itself, every time a query was evaluated — the resulting directions were then used to encode that
same query. This allowed the projection to incorporate information from the query being evaluated, introducing
an optimistic bias into the retrieval estimate. The random
projection alternative was not affected in the same way, since its projection direction does not
depend on the data batch at all; this asymmetry also meant the original PCA-vs-random comparisons
overstated PCA's advantage.

## Discovery and scale

Co-author feedback raised the concern on 2026-08-14. Tracing every call site confirmed the leak was
universal — every benchmark script in the project routed through the same leaky fitting call. A
first out-of-sample rerun, on the 480-item corpus, found the leak inflated PCA's reported advantage
by a real but modest amount at most qubit counts, and initially appeared not to affect the
project's headline n_qubits=4 setting. A same-day check on the other corpora used in the paper
overturned that impression: on the 48-item "hard" corpus — the corpus behind the paper's Section 5.1
headline result and every one of the original hardware-validation jobs — n_qubits=4 showed the
largest leakage effect found anywhere in the investigation, with Recall@1 dropping from a published
0.88 to a corrected 0.66 once the query-included batch was replaced with a query-excluded one.

The effect grew sharply with query-noise strength. On the 480-item corpus's noise sweep, the leak
was small at low noise but, at the higher multipliers the paper reports (4.0 and 6.0), corrected PCA
recall fell to 0.03 and 0.00 respectively — down from a published 0.42 and 0.56. The multiplier-6.0
cell had originally been flagged as "genuinely borderline" because its confidence interval
overlapped classical retrieval's; a follow-up check confirmed this borderline reading was not a
sample-size artifact (rerunning it at five times the original sample size did not resolve it under
the original, still-leaky methodology) — it was an artifact of the leak itself. Once corrected, the
cell is not borderline at all: it is the single most decisive, non-overlapping result of the four
noise levels tested, in the opposite direction from what the leaky number suggested.

The investigation ultimately re-verified every affected table in the paper this way, one at a time:
the synthetic and real-easy qubit sweeps, the second embedding model (mpnet) comparison, the
480-item corpus's full qubit and noise sweeps, and all six intermediate corpus sizes between 48 and
480 items. Every one of them showed the same qualitative pattern — a real, previously unmeasured gap
between the published, query-included numbers and the corrected, query-excluded ones, growing with
query-noise strength.

## Related issues corrected alongside the leak

Two further issues surfaced and were corrected during the same effort, independent of the query-leak
itself:

- **The 48-item corpus's construction changed partway through the project** (`_POOL_PER_CATEGORY`,
  the number of documents drawn per topic before subsampling, went from 60 to 65), which made the
  original Section 5.1 numbers unreproducible from the current codebase for a second, unrelated
  reason. The current value (65) is the one that reproduces today's published tables exactly; an
  earlier attempt to reconstruct the historical pool=60 corpus produced numbers that did not match
  the published table and is preserved in `investigation/other_checks/` as a record of the
  superseded reconstruction attempt.
- **The entangling-layer (RZZ) ablation was deliberately run with query-included fitting**, on the
  reasoning that the ablation question — whether the entangling layer helps or hurts retrieval — is
  orthogonal to the leakage question. That reasoning left an unstated exception to the paper's own
  Limitations claim that every reported result uses out-of-sample fitting, so the ablation was
  rerun out-of-sample as well. Its qualitative conclusion changed substantially: the original,
  leaky version found the entangling layer never improves Recall@1 and hurts by a consistent
  0.08–0.12 at low noise only; the corrected version instead shows a qubit-count- and
  noise-dependent effect, hurting substantially more at n_qubits=8 (up to 0.260 absolute) than the
  leaky version ever showed.

## Current status

As of 2026-09-08, every table and every prose numeric claim in the paper has been re-audited against
its actual source file and confirmed to use out-of-sample (query-excluded) fitting throughout.
`repo/main_reproduction/` contains only these corrected sources — see its README for the complete
table-by-table mapping. The correction occurred in the result-generation scripts rather than the core library itself; the
library remained unchanged: `agent_memory.py`, `q_encoder.py`, `q_graph.py`, and `q_search.py` are
byte-for-byte identical to the original implementation, and the fix lies entirely in how
`fit_pca_projection_scale()` is called (query excluded from the fitting batch) rather than in the
function itself. What changed is that every
result-producing script now excludes the query from its fitting batch, and the resulting numbers
have replaced every leaky one the paper previously reported.

## Folder guide

- **`original_implementation/`** — the pipeline library code, unchanged from the leaky era (see
  "Current status" above for why); `benchmark_realdata_embeddings_hard_CURRENT_pool65.py`, a
  reconstruction of the corpus-loading script at its current, correct pool size; and
  `stale_main_reproduction_results_2026-09-08/`, a preserved copy of what `main_reproduction/`
  contained before its 2026-09-08 rebuild — kept here because several of those files turned out to
  still be leaky despite being labeled as current at the time.
- **`investigation/`** — the diagnostic work, in three threads: `leakage_investigation/` (the
  leakage check itself, corpus by corpus and embedding model by embedding model),
  `transition_boundary_followup/` (the six intermediate-corpus-size study), and `other_checks/` (the
  RZZ ablation, a sample-size robustness check on the multiplier-6.0 borderline cell, and the
  superseded pool=60 baseline reconstruction).
- **`before_after_comparisons/`** — every leaky-vs-leak-free number computed during this
  investigation, paired for direct comparison, plus `COMPARISON.md` rendering them as readable
  tables. This is the folder to consult for the exact magnitude of the leak on any specific table or
  corpus size. Several of `main_reproduction/`'s current source files are direct extracts of the
  leak-free branch of a file here (with the leaky branch stripped out — see
  `main_reproduction/results/scratch_extract_clean_only.py`); others, like the Table 3/13 baselines
  and the RZZ ablation rerun, are independent out-of-sample computations rather than extracts of
  anything in this folder. Either way, the paired files here are kept for the comparison itself, not
  duplicated as a second, redundant copy of the current numbers.
