# Methods §3.1 — Entangling-layer (RZZ) ablation

Feeds the paper's **Methods, Section 3.1** prose: a comparison of the standard encoding (Ry
rotations + RZZ entangling layer) against a plain product-state encoding (Ry only, entangling gates
removed), on the 48-item hard corpus, at both n_qubits=4 and n_qubits=8, across the full noise
sweep. This is the evidence behind the paper's claim that the entangling layer's effect on Recall@1
is "qubit-count- and noise-dependent rather than consistent, and more often harmful than beneficial."

## Which file to use

`scratch_rzz_ablation_oos_results.pkl`, produced by `scratch_verify_rzz_ablation_oos.py` (both in
this folder). Results dict is keyed by `(n_qubits, noise_multiplier, variant)` for
`variant in {'rzz', 'none'}`.

Specific numbers quoted in the paper: at n_qubits=4, mult=1.2, product-state Recall@1 is higher by
0.160 absolute; at n_qubits=4, mult=4.0–8.0, RZZ helps by 0.02–0.06; at n_qubits=8, mult=1.2–8.0,
RZZ hurts by up to 0.260 absolute, helping only mildly at the two highest noise levels tested.

## Fitting convention and history

Out-of-sample (leak-free): the query is excluded from the PCA fitting batch. This is a rerun of an
earlier, deliberately leaky version, `scratch_check_rzz_ablation.py` — that script's own docstring
argued the leakage question was orthogonal to the ablation question, which may be true in isolation,
but the paper's Limitations section makes a blanket claim that *every* reported result uses
out-of-sample fitting, and the leaky version would have been an unstated exception to that claim.
This rerun closes that gap. The original leaky version and its results are archived at
`repo/audit_history/investigation/other_checks/scratch_check_rzz_ablation.py` — its qualitative
conclusion was noticeably different (it found the entangling layer "never improves Recall@1" and
hurts by a consistent "0.08–0.12" at low noise only); that conclusion did not survive out-of-sample
correction, which is itself part of why this rerun mattered.
