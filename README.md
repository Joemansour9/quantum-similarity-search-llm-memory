# Quantum Retrieval Archive

This repository accompanies a research paper on quantum-encoded document retrieval, comparing a PCA-based quantum encoding against random projection and classical retrieval baselines. The methods are evaluated in both simulation and on real IBM Quantum hardware.

## What's here

### `main_reproduction/`

The pipeline code and result files behind every table and figure in the paper. Start here to reproduce a result or trace where a specific reported number originates. See `main_reproduction/README.md` for the complete table-by-table and figure-by-figure mapping.

### `hardware/`

Records for every real IBM Quantum hardware job associated with the paper's hardware-validation section — 65 job records in total, each with its backend, timestamp, circuit configuration, and the specific result it contributed to. See `hardware/README.md` for the complete job-by-job mapping.

### `audit_history/`

Documentation of a methodology issue identified during an earlier stage of the project (a PCA-fitting procedure that briefly allowed the projection to incorporate information from the evaluation target), how the issue was discovered, how large its effect was, and how it was corrected. All results reported in the current paper already reflect the corrected methodology. This folder is retained for transparency, auditability, and reproducibility of the correction process. See `audit_history/SUMMARY.md`.

## Where to start

- To reproduce a specific table, figure, or reported result: `main_reproduction/README.md`
- To look up a specific hardware job: `hardware/README.md`
- To review the methodology correction discussed in the paper's Limitations section: `audit_history/SUMMARY.md`
