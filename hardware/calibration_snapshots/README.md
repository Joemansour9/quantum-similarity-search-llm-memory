# Calibration Snapshots

This directory records the backend calibration data used to evaluate whether backend drift could
plausibly explain any of the hardware-side effects reported in the project.

## Summary

- Epochs A-C (2026-07-30 cluster): no measurable whole-backend drift.
- Epochs D-E (2026-08-13 cluster): recalibration occurred, but no gate- or readout-error values changed.
- Conclusion: backend drift is not a plausible explanation for the hardware effects investigated in these job groups.

Calibration data was pulled for two clusters of jobs: the n=18 drift check (Test 3/6/7/8/9) and a
separate four-job mult=4.0 corpus-size drift check. Three distinct calibration epochs were found
for the first group, via `backend.properties(datetime=<job creation time>)`:

| Epoch | `last_update_date` | Jobs |
|---|---|---|
| A | 2026-07-30T08:10:48+10:00 | Test 3, Test 6 |
| B | 2026-07-30T09:31:34+10:00 | Test 7 |
| C | 2026-07-30T10:36:54+10:00 | Test 8 (both batches), Test 9 fill |

Whole-chip aggregate two-qubit-gate and readout-error statistics were identical to five decimal
places across epochs A-C (mean 2Q error 0.03926; mean readout error 0.02546), indicating no
measurable backend-level drift across this job cluster. This check addresses whether backend drift
could be an alternative explanation for the n=18 comparison's p<0.0001 effect. **Scope caveat:** this is a
whole-chip aggregate, not a check of the exact physical qubits each circuit was transpiled onto
(the original transpiled layouts were not preserved), and no snapshot was pulled for the other 12
jobs in this cluster's era (Test 1, 2, 4, 5, 10, 11, 12, Priority 1/n5, Priority 2, or the
large-corpus breakdown job).

**A second drift check** covers the four mult=4.0 corpus-size jobs `d9ug1i8u5hac73ah6a7g` (216),
`d9ug9q8u5hac73ah6im0` (224), `d9uiq7535hes73fk02lg` (232), `d9ugf5s98n5s7392r6ag` (240) — all
`ibm_kingston`, all 2026-08-13, spanning 09:24 to 12:34 — whose conclusion
(`transition_boundary_hardware_2026-08-13.md`'s "gradual, not sharp" finding) rests on a
scan-count/wrong-rate trend compared across those four jobs' submission times. Three of the four
(216/224/240) fall under one calibration publish (epoch D, `last_update_date`
2026-08-13T09:05:12+10:00); the 232-item job, submitted roughly 3 hours later, falls under a
different, later publish (epoch E, `last_update_date` 10:45:14+10:00) — a real recalibration event
occurred inside this window, unlike A/B/C, which were all one epoch. A full elementwise diff of
every individual 2-qubit gate error and every individual qubit readout error between epoch D and
epoch E found **zero differing values**, and whole-chip aggregates also matched exactly (mean 2Q
error 0.04511, mean readout error 0.02617, identical in both epochs). **Conclusion: backend drift
is not a viable alternative explanation for the four-point "gradual, not sharp" trend in
`transition_boundary_hardware_2026-08-13.md`.** Calibration inputs were identical, at the level of
every individual gate and qubit, across the full 09:24–12:34 window, including across the one
calibration-publish boundary inside it. See `epoch_D.json` / `epoch_E.json` for the full pulled
data.
