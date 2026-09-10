# Background §2.3 — Variance/Enrichment Percentages

Feeds the paper's **Background §2.3** (`sec:background`, "Dimensionality Reduction for Quantum
Encoding") sentence: on the 48-item hard-corpus embeddings, the top-4 and top-16 principal
components' share of total variance, for both embedding models, plus the resulting enrichment
relative to an isotropic null.

## Which file to use

`scratch_verify_variance_enrichment_results.pkl`, produced by
`scratch_verify_variance_enrichment.py`. Fields: `minilm_evr_top4_pct`, `minilm_evr_top16_pct`,
`mpnet_evr_top4_pct`, `mpnet_evr_top16_pct`.

Numbers quoted in the paper: top-4 PCs capture 21.7% of variance for MiniLM (d=384) and 21.0% for
mpnet (d=768); top-16 capture 58.8% and 57.5% respectively. Enrichment relative to an isotropic
null (4/384≈1.04%, 4/768≈0.52%): ~21× for MiniLM, ~40× for mpnet.

## Corpus used

MiniLM: the 48-item hard corpus regenerated via `make_hard_real_corpus(48, 8, seed=0)` from
`benchmark_realdata_embeddings_hard.py` — the seed=0 corpus, the first of the several corpora used
in `table3_realdata_hard_and_table13_baselines/`'s qubit-count sweep. The 20 Newsgroups fetch and
`all-MiniLM-L6-v2` encoding are both deterministic given the same inputs, so this reproduces the
same corpus exactly. mpnet: the pre-existing saved fixture `scratch_mpnet_corpus.pkl` (48 items,
d=768), the same corpus `table6_mpnet_generalisation/`'s scripts load.

No leaky predecessor exists for this specific computation — PCA's explained-variance ratio does
not depend on a query vector at all (it is computed directly from the candidate/corpus batch), so
there is no query-inclusion leak to correct here.
