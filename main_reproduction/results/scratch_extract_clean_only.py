"""
scratch_extract_clean_only.py -- one-off archive-restructuring helper.

Several source pkls in this project store BOTH the leaky and the
out-of-sample (leak-free) computation for the same table, keyed by a
'leaky'/'leak_free' tag, e.g. {(n_qubits, 'pca', 'leaky'): ..., (n_qubits,
'pca', 'leak_free'): ...}. The paper's published numbers always come from
the leak_free branch. For main_reproduction/ to be genuinely clean ("no
leaky data, no exceptions"), this script extracts a leak_free-ONLY copy of
each such file. The original dual-mode file (with both branches) is left
untouched on disk and belongs in audit_history/ instead, since it's the
direct evidence of the leakage investigation (how large the leaky-vs-clean
gap was).

This script only reads existing pkls and writes new *_CLEAN.pkl files
into the current directory; it does not modify or delete anything.
"""
import pickle

def extract_tuple_keyed(fn, out_fn, note):
    """Handles dicts keyed like (n, 'pca', 'leaky') / (n, 'pca', 'leak_free')."""
    with open(fn, "rb") as f:
        d = pickle.load(f)
    clean = {}
    for outer_key, val in d.items():
        if isinstance(outer_key, tuple) and outer_key[-1] in ("leaky", "leak_free"):
            if outer_key[-1] == "leak_free":
                clean[outer_key[:-1]] = val
        elif isinstance(outer_key, dict):
            pass
        else:
            # nested structure (synthetic/real_easy/real_hard_48 style)
            if isinstance(val, dict):
                inner_clean = {}
                for ik, iv in val.items():
                    if isinstance(ik, tuple) and ik[-1] in ("leaky", "leak_free"):
                        if ik[-1] == "leak_free":
                            inner_clean[ik[:-1]] = iv
                    else:
                        inner_clean[ik] = iv
                clean[outer_key] = inner_clean
            else:
                clean[outer_key] = val
    with open(out_fn, "wb") as f:
        pickle.dump({"description": note, "source_dual_mode_file": fn, "leak_free_only": clean}, f)
    print(f"{fn} -> {out_fn}  ({len(clean)} top-level keys)")


def extract_listpair_keyed(fn, out_fn, note):
    """Handles dicts keyed by mult (or (n_items,mult)) -> dict of
    {('pca','leaky'): val_or_(val,ci), ..., 'classical': val_or_(val,ci)}."""
    with open(fn, "rb") as f:
        d = pickle.load(f)
    clean = {}
    for outer_key, row_dict in d.items():
        row = {}
        for k, v in row_dict.items():
            if isinstance(k, tuple) and k[-1] in ("leaky", "leak_free"):
                if k[-1] == "leak_free":
                    row[k[:-1]] = v
            else:
                row[k] = v
        clean[outer_key] = row
    with open(out_fn, "wb") as f:
        pickle.dump({"description": note, "source_dual_mode_file": fn, "leak_free_only": clean}, f)
    print(f"{fn} -> {out_fn}  ({len(clean)} top-level keys)")


# --- Table 1, 2, and Table 6's MiniLM row (48-item hard corpus) ---
extract_tuple_keyed(
    "scratch_pca_leakage_remaining_results.pkl",
    "scratch_table1_table2_table6minilm_CLEAN.pkl",
    "Out-of-sample-only extract of synthetic (Table 1), real_easy (Table 2), "
    "and real_hard_48/MiniLM (Table 6 MiniLM row) qubit sweeps. Leaky branch "
    "stripped; see audit_history for the full leaky-vs-clean comparison this "
    "was extracted from.",
)

# --- Table 7 (480-item qubit sweep) ---
with open("scratch_pca_leakage_results.pkl", "rb") as f:
    d = pickle.load(f)
with open("scratch_table7_480qubitsweep_CLEAN.pkl", "wb") as f:
    pickle.dump({
        "description": "Out-of-sample-only extract (Table 7, 480-item qubit sweep). "
                        "Leaky branch stripped; see audit_history for the full comparison.",
        "source_dual_mode_file": "scratch_pca_leakage_results.pkl",
        "leak_free_only": d["leak_free"],
    }, f)
print("scratch_pca_leakage_results.pkl -> scratch_table7_480qubitsweep_CLEAN.pkl")

# --- Table 6 mpnet row ---
extract_tuple_keyed(
    "scratch_pca_leakage_mpnet_qubitsweep_results.pkl",
    "scratch_table6_mpnet_CLEAN.pkl",
    "Out-of-sample-only extract of the mpnet qubit sweep (Table 6 mpnet row). "
    "Leaky branch stripped; see audit_history for the full comparison.",
)

# --- Table 8 (480-item n=100 confirmatory noise sweep, both files) ---
extract_listpair_keyed(
    "scratch_pca_leakage_480_n100_confirm_results.pkl",
    "scratch_table8_480n100_highmult_CLEAN.pkl",
    "Out-of-sample-only extract, Table 8 mult=4.0/6.0 rows. Leaky branch stripped.",
)
extract_listpair_keyed(
    "scratch_pca_leakage_480_n100_confirm_lowmult_results.pkl",
    "scratch_table8_480n100_lowmult_CLEAN.pkl",
    "Out-of-sample-only extract, Table 8 mult=0.2/1.2/2.0 rows. Leaky branch stripped.",
)

# --- Section 6 prose: mean-deviation/ratio numbers ---
extract_listpair_keyed(
    "scratch_pca_leakage_mpnet_noisesweep_nq8_results.pkl",
    "scratch_sec6_mpnet_noisesweep_nq8_CLEAN.pkl",
    "Out-of-sample-only extract, mpnet n_qubits=8 noise sweep (Section 6 prose). Leaky branch stripped.",
)
extract_listpair_keyed(
    "scratch_pca_leakage_minilm_noisesweep_nq4_results.pkl",
    "scratch_sec6_minilm_noisesweep_nq4_CLEAN.pkl",
    "Out-of-sample-only extract, MiniLM n_qubits=4 noise sweep (Section 6 prose). Leaky branch stripped.",
)
# note: minilm_noisesweep_nq8 (Table 5 source) also needs a CLEAN extract
extract_listpair_keyed(
    "scratch_pca_leakage_minilm_noisesweep_nq8_results.pkl",
    "scratch_table5_minilm_noisesweep_nq8_CLEAN.pkl",
    "Out-of-sample-only extract, MiniLM n_qubits=8 full noise sweep (Table 5, "
    "and feeds Section 6 prose ratio). Leaky branch stripped.",
)

# --- Table 9 (corpus-size transition, 96-400 items) ---
extract_listpair_keyed(
    "scratch_pca_leakage_transition_sizes_results.pkl",
    "scratch_table9_transition_sizes_CLEAN.pkl",
    "Out-of-sample-only extract, Table 9 (corpus sizes 96-400, mults 1.2/2.0). Leaky branch stripped.",
)

print("\nDone. All *_CLEAN.pkl files contain ONLY the out-of-sample branch.")
