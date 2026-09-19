# Section 7 — Corpus-Scarcity Finding

Feeds the paper's **Section 7** (`sec:corpus-scale`) paragraph beginning "The corpus-scarcity
finding does not hold proportionally either": the fraction of query–candidate pairs whose exact
simulator fidelity clears a 0.5 threshold, at both the 480-item and 48-item hard corpora, for both
PCA and random projection.

## Which file to use

`scratch_scarcity_oos_results.pkl`, produced by `scratch_verify_scarcity_oos.py`. Fields:
`corpus_480` (PCA fraction, out-of-sample) and `corpus_48` (PCA fraction, out-of-sample; random
fraction, fresh recomputation — see below).

Numbers quoted in the paper: 480 items — PCA 471/480 (98.1%), random 184/480 (38.3%); 48 items —
PCA 48/48 (100.0%), random 9/48 (18.8%).

## A note on "8/48": why it's in this repo but not in the paper

Random projection's fitting is not affected by the query-inclusion leak (its projection matrix is
drawn from a fixed seed, independent of the data batch), so its fractions were not required to be
rechecked out-of-sample. The 480-item random figure (184/480) is the original, still-valid number.

The 48-item random figure has a wrinkle worth recording here, even though it no longer appears in
the paper. An earlier draft of this project's own analysis referenced "8/48" for this comparison,
but no script computing that number survives anywhere in the project — it exists only as a
hardcoded reference comment in `scratch_large_corpus_sweep.py`, itself citing a number from some
earlier, now-untraceable source. This file's own fresh reconstruction of the 48-item corpus/seed
and a from-scratch random-projection recomputation, included specifically to sanity-check that
reconstruction, gives **9/48**, not 8/48
(`corpus_48['matches_published_random_8_of_48'] = False` in the saved pkl).

No PCA/random fraction for the 48-item corpus was ever published in any version of the manuscript
before this file computed one — this comparison is being reported for the first time, not
corrected from a prior published value. So the paper reports this script's own result, 9/48
(18.8%), plainly, with no hedge and no mention of "8/48": there is nothing to reconcile, since "8/48"
was never a number the paper itself claimed. The discrepancy above is kept in this README as a
record of the investigation, not as an open question the manuscript needs to carry.

## Fitting convention

Out-of-sample throughout for PCA (query excluded from the batch used to fit the projection, same
convention as every other result in this paper). The 480-item PCA figure corrects a leaky
predecessor (originally 456/480, published as `PUBLISHED_480` in this file for reference); the
48-item PCA figure (100.0%, i.e. 48/48) had no prior leaky value to correct, since no PCA fraction
was ever published for the 48-item corpus before this file.
