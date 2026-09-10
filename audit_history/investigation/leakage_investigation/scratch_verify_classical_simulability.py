"""
scratch_verify_classical_simulability.py — Task 1 (co-author feedback):
Is |<psi_q|psi_c>|^2 for the chain-topology Ry+RZZ encoding classically
computable in closed form from the two angle vectors alone, without
building/simulating the quantum circuit?

DERIVATION:
State after the Ry layer is a product state:
    |phi(theta)> = tensor_i [cos(theta_i/2)|0> + sin(theta_i/2)|1>]
RZZ(phi) = exp(-i*phi/2 * Z(x)Z) is DIAGONAL in the computational basis:
    RZZ(phi)|b_u b_v> = exp(-i*phi/2 * z_u*z_v) |b_u b_v>,  z = +1 if bit=0, -1 if bit=1
So the full state is |psi(theta)> = D(theta) |phi(theta)>, where D(theta) is diagonal:
    D(theta)|b> = exp(-i/2 * sum_{(u,v) in edges} theta_u*theta_v * z_u(b)*z_v(b)) |b>
Therefore:
    <psi_q|psi_c> = sum_b <phi(alpha)|b><b|phi(beta)> * exp(-i/2 * sum_edges (beta_u*beta_v - alpha_u*alpha_v)*z_u(b)*z_v(b))
This is an EXACT closed-form sum over 2^n_qubits basis states -- no gates simulated,
just cosines/sines and phases from the two angle vectors and the edge list. This holds
for ANY edge set (chain or correlation-derived), general graph.

For the specific CHAIN topology (edges = path graph (0,1),(1,2),...), this sum has the
structure of a 1D transfer-matrix / tensor-network contraction (treewidth 1), computable
in O(n_qubits) time via a simple DP recursion -- genuinely avoiding the exponential
blowup of both brute-force enumeration AND Qiskit's dense-statevector simulation.
"""
import cmath
import itertools
import numpy as np
from q_encoder import fit_pca_projection_scale, fit_projection_scale, project_to_n_features, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits


def closed_form_overlap_bruteforce(theta_q, theta_c, edges, n_qubits):
    """General closed-form sum, ANY edge set. O(2^n * n) -- for correctness
    verification, not efficiency; still no quantum circuit is built/simulated."""
    total = 0j
    for bits in itertools.product((0, 1), repeat=n_qubits):
        amp = 1.0
        for i in range(n_qubits):
            if bits[i] == 0:
                amp *= np.cos(theta_q[i] / 2) * np.cos(theta_c[i] / 2)
            else:
                amp *= np.sin(theta_q[i] / 2) * np.sin(theta_c[i] / 2)
        if amp == 0.0:
            continue
        phase = 0.0
        for (u, v) in edges:
            if u == v:
                continue
            zu = 1 - 2 * bits[u]
            zv = 1 - 2 * bits[v]
            phase += -0.5 * (theta_c[u] * theta_c[v] - theta_q[u] * theta_q[v]) * zu * zv
        total += amp * cmath.exp(1j * phase)
    return total


def chain_transfer_matrix_overlap(theta_q, theta_c, n_qubits):
    """O(n_qubits) DP, ONLY valid for the canonical chain edges (i, i+1)."""
    V = np.array([
        np.cos(theta_q[0] / 2) * np.cos(theta_c[0] / 2),
        np.sin(theta_q[0] / 2) * np.sin(theta_c[0] / 2),
    ], dtype=complex)
    for i in range(1, n_qubits):
        w_i = np.array([
            np.cos(theta_q[i] / 2) * np.cos(theta_c[i] / 2),
            np.sin(theta_q[i] / 2) * np.sin(theta_c[i] / 2),
        ])
        newV = np.zeros(2, dtype=complex)
        for b_i in (0, 1):
            z_i = 1 - 2 * b_i
            s = 0j
            for b_prev in (0, 1):
                z_prev = 1 - 2 * b_prev
                phase = -0.5 * (theta_c[i - 1] * theta_c[i] - theta_q[i - 1] * theta_q[i]) * z_prev * z_i
                s += V[b_prev] * cmath.exp(1j * phase)
            newV[b_i] = s * w_i[b_i]
        V = newV
    return V[0] + V[1]


# --- Step 0: sanity-check the RZZ phase-sign convention against Qiskit directly ---
print("=== Step 0: RZZ phase convention sanity check (2 qubits, isolated) ===")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

rng0 = np.random.default_rng(0)
for _ in range(3):
    a1, a2 = rng0.uniform(0, np.pi, 2)
    qc = QuantumCircuit(2)
    qc.ry(a1, 0)
    qc.ry(a2, 1)
    qc.rzz(a1 * a2, 0, 1)
    psi = Statevector.from_instruction(qc).data
    # formula for the same 2-qubit "self" state (theta_q=theta_c=[a1,a2])
    closed_form = np.zeros(4, dtype=complex)
    for idx, bits in enumerate(itertools.product((0, 1), repeat=2)):
        amp = 1.0
        for i, b in enumerate(bits):
            amp *= np.cos([a1, a2][i] / 2) if b == 0 else np.sin([a1, a2][i] / 2)
        z0 = 1 - 2 * bits[0]
        z1 = 1 - 2 * bits[1]
        phase = -0.5 * (a1 * a2) * z0 * z1  # theta_q==theta_c here -> uses beta convention directly
        closed_form[idx] = amp * cmath.exp(1j * phase)
    # qiskit little-endian: statevector index bits are (q1,q0) reversed; compare via inner product magnitude instead
    overlap = abs(np.vdot(psi, closed_form))
    print(f"  a1={a1:.3f} a2={a2:.3f}  |<qiskit|closed_form>|={overlap:.6f}  (expect ~1.0, up to global phase)")

# --- Step 1: load real corpus, real R/scale from this project's own pipeline ---
import pickle

with open("scratch_large_corpus_480.pkl", "rb") as f:
    d = pickle.load(f)
corpus, keys = d["corpus"], d["keys"]
d_dim = next(iter(corpus.values())).shape[0]
candidate_batch = np.stack([corpus[k] for k in keys])
CORPUS_STD = 0.04882

print("\n=== Step 1: chain topology -- closed-form vs Qiskit statevector, across n_qubits/projection/noise ===")
max_dev_chain_bruteforce = 0.0
max_dev_chain_transfer = 0.0
n_tested = 0
rng = np.random.default_rng(1)
for n_qubits in (4, 6, 8, 10):
    edges = KnowledgeGraph.chain(n_qubits).edges()
    for proj in ("pca", "random"):
        for mult in (0.2, 1.2, 4.0):
            noise = mult * CORPUS_STD
            for trial in range(3):
                q_idx = trial + hash((n_qubits, proj, mult, trial)) % 400
                relevant_key = keys[q_idx % len(keys)]
                cand_idx = (q_idx + 7) % len(keys)  # a different, arbitrary candidate
                cand_key = keys[cand_idx]
                query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
                full_batch = np.vstack([query_vec[None, :], candidate_batch])
                if proj == "pca":
                    R, lo, hi = fit_pca_projection_scale(full_batch, n_qubits)
                else:
                    R, lo, hi = fit_projection_scale(full_batch, n_qubits, seed=q_idx)
                scale = (lo, hi)

                theta_q = project_to_n_features(query_vec, n_qubits, scale, R)
                theta_c = project_to_n_features(candidate_batch[cand_idx], n_qubits, scale, R)

                qc_query = build_feature_map(query_vec, n_qubits, scale, edges, R, entangler="rzz")
                fid_qiskit, _ = statevector_fidelity_from_circuits(
                    qc_query, candidate_batch[cand_idx], n_qubits, scale, edges, R, entangler="rzz"
                )

                ov_bf = closed_form_overlap_bruteforce(theta_q, theta_c, edges, n_qubits)
                fid_bf = abs(ov_bf) ** 2

                ov_tm = chain_transfer_matrix_overlap(theta_q, theta_c, n_qubits)
                fid_tm = abs(ov_tm) ** 2

                dev_bf = abs(fid_bf - fid_qiskit)
                dev_tm = abs(fid_tm - fid_qiskit)
                max_dev_chain_bruteforce = max(max_dev_chain_bruteforce, dev_bf)
                max_dev_chain_transfer = max(max_dev_chain_transfer, dev_tm)
                n_tested += 1

print(f"  tested {n_tested} (n_qubits, proj, mult, trial) combinations, chain topology")
print(f"  max |closed_form_bruteforce - qiskit_statevector| = {max_dev_chain_bruteforce:.3e}")
print(f"  max |chain_transfer_matrix  - qiskit_statevector| = {max_dev_chain_transfer:.3e}")

print("\n=== Step 2: correlation topology -- closed-form (general) vs Qiskit statevector ===")
max_dev_corr = 0.0
n_tested_corr = 0
for n_qubits in (4, 6, 8):
    for proj in ("pca", "random"):
        for trial in range(3):
            q_idx = trial * 13 + n_qubits
            relevant_key = keys[q_idx % len(keys)]
            cand_idx = (q_idx + 11) % len(keys)
            query_vec = corpus[relevant_key] + rng.normal(scale=1.2 * CORPUS_STD, size=d_dim)
            full_batch = np.vstack([query_vec[None, :], candidate_batch])
            if proj == "pca":
                R, lo, hi = fit_pca_projection_scale(full_batch, n_qubits)
            else:
                R, lo, hi = fit_projection_scale(full_batch, n_qubits, seed=q_idx)
            scale = (lo, hi)
            kg = KnowledgeGraph.from_correlation(candidate_batch, R, n_qubits, threshold=0.3)
            edges = kg.edges()
            if not edges:
                continue  # need at least one edge to test entanglement-dependent overlap

            theta_q = project_to_n_features(query_vec, n_qubits, scale, R)
            theta_c = project_to_n_features(candidate_batch[cand_idx], n_qubits, scale, R)

            qc_query = build_feature_map(query_vec, n_qubits, scale, edges, R, entangler="rzz")
            fid_qiskit, _ = statevector_fidelity_from_circuits(
                qc_query, candidate_batch[cand_idx], n_qubits, scale, edges, R, entangler="rzz"
            )
            ov_bf = closed_form_overlap_bruteforce(theta_q, theta_c, edges, n_qubits)
            fid_bf = abs(ov_bf) ** 2
            dev = abs(fid_bf - fid_qiskit)
            max_dev_corr = max(max_dev_corr, dev)
            n_tested_corr += 1
            print(f"  n_qubits={n_qubits} proj={proj} trial={trial} edges={edges}  "
                  f"qiskit={fid_qiskit:.6f}  closed_form={fid_bf:.6f}  dev={dev:.2e}")

print(f"\n  tested {n_tested_corr} correlation-topology cases, max deviation = {max_dev_corr:.3e}")
print("\nDONE")
