# Appendix — Destructive-Circuit Validation: Full Per-Test Breakdown

Feeds the paper's **Appendix Table `destructive-validation-appendix`** and the validation
sentence in **Section 8.1** (`sec:hw:destructive`): a local, no-hardware, 20,000-shot
`AerSimulator` check of the destructive (Bell-basis) SWAP-test circuit's shot-based fidelity
estimate against exact statevector fidelity, across all 144 query–candidate pairs actually staged
for the hardware tests of Section 8. Grouped by the 13 source files (Test 1, Test 2, Test 6,
Test 3, Priority 2, Test 5, Test 7, Test 8, Test 9 fill, Test 10, Test 11, Priority 1, Test 12) —
the same grouping the appendix table itself uses.

## Which file to use

`scratch_verify_destructive_validation_all_staged_pairs_results.pkl`, produced by
`scratch_verify_destructive_validation_all_staged_pairs_CORRECTED.py`. This is the current,
authoritative file — its numbers are what the manuscript's appendix table and Section 8.1 report
(min/mean/median/max absolute difference per test group, overall summary across 144 pairs, and the
worst-outlier pair: `item_21_c3`, Test 9 fill, exact fidelity 0.5086, local 20k-shot fidelity
0.5283, absolute difference 0.0197).

## Fitting convention and history

Of the 144 pairs, 101 are PCA-projected and 43 are random-projected (Test 2, Test 6, and the
random halves of Tests 7/8/10/11/Priority 1 — confirmed per-pair from each source file's own
method labels). The query-exclusion leak is a PCA-fitting-specific issue: random projection's
matrix is drawn from a fixed seed independent of the data batch, so there is nothing to correct
for the 43 random pairs. `scratch_verify_destructive_validation_all_staged_pairs_CORRECTED.py`
therefore rebuilds all 101 PCA pairs from scratch with out-of-sample fitting (candidate batch
alone, query excluded — the same convention as every other result in this paper) and carries the
43 random pairs over unchanged from their original staged circuits. Every one of the 144 circuits,
rebuilt or unchanged, was then freshly run through `AerSimulator` at 20,000 shots.

`scratch_verify_destructive_validation_all_staged_pairs.py` (also in this folder) is the earlier,
pre-correction script: it re-simulated the *original* staged circuits, before out-of-sample
PCA fitting was applied to this specific check. Both scripts write to the same output filename;
the corrected script's run is what is currently saved as
`scratch_verify_destructive_validation_all_staged_pairs_results.pkl` and is the version this
folder's numbers, and the paper's, come from. The original script is retained here for provenance
— it is what first established that no saved script or data anywhere in the project backed the
paper's now-superseded "diff=0.0001–0.0013" claim, and it is what the corrected script's own
docstring compares itself against — but its own output was superseded in place, not archived
separately.

One consequence of the correction worth noting explicitly: `item_0_c0`'s self-match pair (used
directly in Table `hw-gatecount` and Table `hw-ranking`, and appearing again here as the self-match
entry for Test 1, Test 3, Priority 2, and Test 5) now reads exact fidelity 0.9552 in every one of
those places — before the correction, this same nominal pair read 0.9552 in the real-hardware
tables but 0.7701 in this appendix, because the appendix was still built from the original
(query-included) circuit while the hardware tables already used the corrected one. That
inconsistency is resolved by this rerun.
