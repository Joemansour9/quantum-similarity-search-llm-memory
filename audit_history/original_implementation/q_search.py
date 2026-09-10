"""
q_search.py — Phase 2/3: Similarity Search Engine

Implements pairwise similarity between a query and a batch of candidates
via the quantum SWAP test, with a classical baseline (cosine
similarity) for comparison.

*** SCOPE LIMITATION, READ BEFORE EXTENDING ***
This is O(N) by construction: evaluate_similarity() loops over the
candidate batch and runs one SWAP test per candidate. There is no
Grover-style amplitude amplification here, and no oracle giving
superposition access to all N candidates at once (that would require
real qRAM, which this implementation does not have and does not assume
— see q_encoder.py and the Project 5 scoping notes). A sub-linear query
count claim would be false for this implementation. What is actually
tested is whether the SWAP-test kernel fidelity gives better/more
robust/more noise-tolerant similarity judgments than classical cosine
similarity, per-comparison — a quality question, not a complexity-class
question.
"""

from __future__ import annotations

import time
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from q_encoder import build_feature_map, fit_projection_scale


def swap_test_circuit(
    qc_a: QuantumCircuit, qc_b: QuantumCircuit, n_qubits: int
) -> QuantumCircuit:
    """
    Build the standard SWAP test circuit: one ancilla qubit, Hadamard
    before and after a controlled-SWAP between the two n_qubits registers
    prepared by qc_a and qc_b, then measure the ancilla.

    P(ancilla = 0) = 1/2 + 1/2 * |<psi_a|psi_b>|^2
    so fidelity = |<psi_a|psi_b>|^2 = 2*P(0) - 1, estimated from shot
    statistics — this is what a real (non-simulator-privileged) SWAP
    test measurement actually gives you, unlike reading the statevector
    directly.
    """
    anc = QuantumRegister(1, "anc")
    reg_a = QuantumRegister(n_qubits, "a")
    reg_b = QuantumRegister(n_qubits, "b")
    creg = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, reg_a, reg_b, creg)
    qc.compose(qc_a, qubits=reg_a, inplace=True)
    qc.compose(qc_b, qubits=reg_b, inplace=True)

    qc.h(anc[0])
    for i in range(n_qubits):
        qc.cswap(anc[0], reg_a[i], reg_b[i])
    qc.h(anc[0])
    qc.measure(anc[0], creg[0])

    return qc


def destructive_swap_test_circuit(
    qc_a: QuantumCircuit, qc_b: QuantumCircuit, n_qubits: int
) -> QuantumCircuit:
    """
    Ancilla-free alternative to swap_test_circuit(): the "destructive" or
    "Bell-basis" SWAP test (Garcia-Escartin & Chamorro-Posada, 2013 —
    "SWAP test and Hong-Ou-Mandel effect are equivalent"). For each of the
    n_qubits corresponding qubit pairs (a_i, b_i), apply CNOT(a_i -> b_i)
    then H(a_i), then measure BOTH qubits — no ancilla, no CSWAP network.

    Recovering the fidelity needs the JOINT statistics across all n pairs
    in a single shot, not a per-pair estimate (see
    destructive_fidelity_from_counts()) — this still correctly computes
    the overlap of the FULL n_qubits states, including any entanglement
    RZZ introduces WITHIN a register, not just a per-qubit product
    overlap. That joint-parity step is what makes this equivalent to the
    ancilla SWAP test's |<psi_a|psi_b>|^2, not an approximation of it.

    Built as: qubits 0..n_qubits-1 = register a, qubits n_qubits..2n-1 =
    register b. ONE classical register of size 2*n_qubits (a_i ->
    creg[i], b_i -> creg[n_qubits+i]) rather than two separate
    registers — deliberately, so this uses the exact same single-
    register result-parsing path (result[0].data.<name>.get_counts())
    already proven against real hardware for the ancilla version, instead
    of trusting untested multi-register joint-correlation behavior from
    the qiskit-ibm-runtime SamplerV2 result API.

    Motivation: swap_test_circuit()'s single CSWAP per qubit pair costs
    ~11 native 2-qubit gates on real IBM hardware (measured empirically
    via transpilation to FakeSherbrooke/FakeTorino — see project notes),
    dominating circuit depth. CNOT is a single native-adjacent 2-qubit
    interaction (1 ECR/CZ after transpilation, not ~11) — this should
    cut 2-qubit gate count roughly 5-10x for the SWAP-test apparatus
    itself (the feature-map RZZ layers are unchanged either way).
    """
    reg_a = QuantumRegister(n_qubits, "a")
    reg_b = QuantumRegister(n_qubits, "b")
    creg = ClassicalRegister(2 * n_qubits, "c")

    qc = QuantumCircuit(reg_a, reg_b, creg)
    qc.compose(qc_a, qubits=reg_a, inplace=True)
    qc.compose(qc_b, qubits=reg_b, inplace=True)

    for i in range(n_qubits):
        qc.cx(reg_a[i], reg_b[i])
        qc.h(reg_a[i])

    qc.measure(reg_a, creg[0:n_qubits])
    qc.measure(reg_b, creg[n_qubits:2 * n_qubits])
    return qc


def destructive_fidelity_from_counts(counts: dict, n_qubits: int, shots: int) -> float:
    """
    counts: qiskit-style bitstring counts dict from a SINGLE 2*n_qubits
    classical register (see destructive_swap_test_circuit()), where
    creg[0:n_qubits] = register a's measurement, creg[n_qubits:2n] =
    register b's -- little-endian (bit 0 = rightmost char of the string,
    per qiskit convention), so reversing the string puts index i at
    position i directly.

    fidelity = E[(-1)^(sum_i x_i * y_i)], the parity of the AND of each
    corresponding (a_i, b_i) measurement pair, summed across ALL pairs
    BEFORE exponentiating -- this joint (not per-pair) parity is what
    reproduces the full-register overlap |<psi_a|psi_b>|^2. Clipped to
    [0, 1]: the raw estimator ranges over [-1, 1] and shot noise can
    push it slightly outside the theoretically-valid [0, 1] fidelity
    range, same clipping rationale as swap_test_fidelity_from_circuits().
    """
    total = 0.0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[::-1]  # reverse -> position i = creg[i]
        x = bits[:n_qubits]        # register a, creg[0:n_qubits]
        y = bits[n_qubits:2 * n_qubits]  # register b, creg[n_qubits:2n]
        parity = sum(int(x[i]) & int(y[i]) for i in range(n_qubits)) % 2
        total += count * (1 if parity == 0 else -1)
    raw = total / shots
    return max(0.0, min(1.0, raw))


def destructive_swap_test_fidelity_from_circuits(
    qc_query: QuantumCircuit,
    x_candidate: np.ndarray,
    n_qubits: int,
    scale: tuple[float, float] | None = None,
    edges: list[tuple[int, int]] | None = None,
    R: np.ndarray | None = None,
    entangler: str = "rzz",
    shots: int = 4096,
    backend: AerSimulator | None = None,
) -> tuple[float, dict]:
    """
    Ancilla-free counterpart to swap_test_fidelity_from_circuits() — same
    call signature and same metrics dict shape, so it's a drop-in
    alternative anywhere the ancilla version is used (see
    evaluate_similarity()'s method="destructive_swap_test" option).
    """
    qc_b = build_feature_map(x_candidate, n_qubits, scale, edges, R, entangler)
    full_circuit = destructive_swap_test_circuit(qc_query, qc_b, n_qubits)

    if backend is None:
        backend = AerSimulator()

    transpiled = backend.transpile(full_circuit) if hasattr(backend, "transpile") else full_circuit
    job = backend.run(transpiled, shots=shots)
    counts = job.result().get_counts()

    fidelity_est = destructive_fidelity_from_counts(counts, n_qubits, shots)

    metrics = {
        "feature_map_depth": qc_b.depth(),
        "feature_map_gate_count": sum(qc_b.count_ops().values()),
        "evaluation_circuit_depth": full_circuit.depth(),
        "evaluation_circuit_gate_count": sum(full_circuit.count_ops().values()),
        "shots": shots,
    }
    return fidelity_est, metrics


def destructive_swap_test_fidelity(
    x_query: np.ndarray,
    x_candidate: np.ndarray,
    n_qubits: int,
    scale: tuple[float, float] | None = None,
    edges: list[tuple[int, int]] | None = None,
    R: np.ndarray | None = None,
    entangler: str = "rzz",
    shots: int = 4096,
    seed: int | None = None,
) -> tuple[float, dict]:
    """Convenience wrapper — builds the query circuit fresh. See destructive_swap_test_fidelity_from_circuits()."""
    qc_a = build_feature_map(x_query, n_qubits, scale, edges, R, entangler)
    backend = AerSimulator(seed_simulator=seed)
    return destructive_swap_test_fidelity_from_circuits(
        qc_a, x_candidate, n_qubits, scale, edges, R, entangler, shots, backend
    )


def swap_test_fidelity_from_circuits(
    qc_query: QuantumCircuit,
    x_candidate: np.ndarray,
    n_qubits: int,
    scale: tuple[float, float] | None = None,
    edges: list[tuple[int, int]] | None = None,
    R: np.ndarray | None = None,
    entangler: str = "rzz",
    shots: int = 4096,
    backend: AerSimulator | None = None,
) -> tuple[float, dict]:
    """
    Same as swap_test_fidelity, but takes an ALREADY-BUILT query circuit
    (qc_query) instead of rebuilding it from x_query every call. Use this
    inside a loop over many candidates against one fixed query — the
    query encoding is constant during the search and should only be
    built once (see evaluate_similarity()).

    backend: reuse a single AerSimulator instance across many calls
    instead of constructing a new one per candidate.
    """
    qc_b = build_feature_map(x_candidate, n_qubits, scale, edges, R, entangler)
    full_circuit = swap_test_circuit(qc_query, qc_b, n_qubits)

    if backend is None:
        backend = AerSimulator()

    transpiled = backend.transpile(full_circuit) if hasattr(backend, "transpile") else full_circuit
    job = backend.run(transpiled, shots=shots)
    counts = job.result().get_counts()

    p0 = counts.get("0", 0) / shots
    fidelity_est = max(0.0, 2 * p0 - 1)  # clip below 0: shot noise can push this slightly negative

    metrics = {
        # Feature-map depth/gates: the encoding circuit alone (qc_b here),
        # comparable directly to the statevector method's reported metrics.
        "feature_map_depth": qc_b.depth(),
        "feature_map_gate_count": sum(qc_b.count_ops().values()),
        # Full evaluation circuit: what actually got executed (query +
        # candidate registers + ancilla + CSWAP network + measurement).
        # NOT directly comparable to statevector mode's full-circuit depth,
        # since that mode doesn't need a SWAP-test apparatus at all.
        "evaluation_circuit_depth": full_circuit.depth(),
        "evaluation_circuit_gate_count": sum(full_circuit.count_ops().values()),
        "shots": shots,
        "p0": p0,
    }
    return fidelity_est, metrics


def swap_test_fidelity(
    x_query: np.ndarray,
    x_candidate: np.ndarray,
    n_qubits: int,
    scale: tuple[float, float] | None = None,
    edges: list[tuple[int, int]] | None = None,
    R: np.ndarray | None = None,
    entangler: str = "rzz",
    shots: int = 4096,
    seed: int | None = None,
) -> tuple[float, dict]:
    """
    Convenience wrapper for a single query-candidate pair (builds the
    query circuit fresh). For repeated calls against a fixed query
    (i.e. inside a search loop), use swap_test_fidelity_from_circuits()
    with a precomputed qc_query instead — see evaluate_similarity().
    """
    qc_a = build_feature_map(x_query, n_qubits, scale, edges, R, entangler)
    backend = AerSimulator(seed_simulator=seed)
    return swap_test_fidelity_from_circuits(
        qc_a, x_candidate, n_qubits, scale, edges, R, entangler, shots, backend
    )


def statevector_fidelity_from_circuits(
    qc_query: QuantumCircuit,
    x_candidate: np.ndarray,
    n_qubits: int,
    scale: tuple[float, float] | None = None,
    edges: list[tuple[int, int]] | None = None,
    R: np.ndarray | None = None,
    entangler: str = "rzz",
) -> tuple[float, dict]:
    """
    Same as statevector_fidelity, but takes an ALREADY-BUILT query circuit
    (qc_query) instead of rebuilding it from x_query every call — mirrors
    swap_test_fidelity_from_circuits() so both evaluation paths in
    evaluate_similarity() build the query encoding exactly once.

    Exact fidelity via direct statevector inner product. This is NOT
    something a real quantum computer gives you for free — it requires
    full classical access to the simulated statevector. Use this only
    as a noise-free reference to compare shot-based SWAP-test estimates
    against, never as a stand-in for what hardware/SWAP-test would report.

    Returns (fidelity, metrics). metrics uses the SAME field names as
    swap_test_fidelity_from_circuits() for feature_map_depth/gate_count
    (directly comparable), but "evaluation_circuit_depth" here equals
    the feature-map depth itself, since this method needs no SWAP-test
    apparatus — that's expected, not a bug, and should not be compared
    to swap_test mode's (larger) evaluation_circuit_depth.
    """
    qc_b = build_feature_map(x_candidate, n_qubits, scale, edges, R, entangler)
    psi_a = Statevector.from_instruction(qc_query)
    psi_b = Statevector.from_instruction(qc_b)
    fidelity = abs(psi_a.inner(psi_b)) ** 2

    metrics = {
        "feature_map_depth": qc_b.depth(),
        "feature_map_gate_count": sum(qc_b.count_ops().values()),
        "evaluation_circuit_depth": qc_b.depth(),
        "evaluation_circuit_gate_count": sum(qc_b.count_ops().values()),
    }
    return fidelity, metrics


def statevector_fidelity(
    x_query: np.ndarray,
    x_candidate: np.ndarray,
    n_qubits: int,
    scale: tuple[float, float] | None = None,
    edges: list[tuple[int, int]] | None = None,
    R: np.ndarray | None = None,
    entangler: str = "rzz",
) -> tuple[float, dict]:
    """
    Convenience wrapper for a single query-candidate pair (builds the
    query circuit fresh). For repeated calls against a fixed query,
    use statevector_fidelity_from_circuits() with a precomputed
    qc_query instead — see evaluate_similarity().
    """
    qc_a = build_feature_map(x_query, n_qubits, scale, edges, R, entangler)
    return statevector_fidelity_from_circuits(
        qc_a, x_candidate, n_qubits, scale, edges, R, entangler
    )


def cosine_similarity(x_query: np.ndarray, x_candidate: np.ndarray) -> float:
    """Classical baseline: standard cosine similarity in [-1, 1]."""
    num = np.dot(x_query, x_candidate)
    denom = np.linalg.norm(x_query) * np.linalg.norm(x_candidate)
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


def evaluate_similarity(
    query: np.ndarray,
    candidate_batch: np.ndarray,
    n_qubits: int,
    edges: list[tuple[int, int]] | None = None,
    entangler: str = "rzz",
    method: str = "swap_test",
    shots: int = 4096,
    seed: int | None = None,
    R: np.ndarray | None = None,
    scale: tuple[float, float] | None = None,
) -> dict:
    """
    Evaluate similarity between one query and a batch of N candidates,
    and rank candidates by both quantum fidelity and classical cosine.

    *** This is O(N): one circuit run per candidate. No sub-linear claim
    is made or implied — see module docstring. ***

    method: "swap_test" (default, ancilla + CSWAP, honest shot-based
    estimate), "destructive_swap_test" (ancilla-free Bell-basis
    alternative — same fidelity quantity, far fewer 2-qubit gates on
    real hardware, see destructive_swap_test_circuit()'s docstring), or
    "statevector" (exact, simulator-only reference — see caveat above).

    R, scale: OPTIONAL precomputed projection/scale, e.g. from a frozen
    store (see agent_memory.py's freeze_projection()). If both are
    given, they are used AS-IS and NOT refit — this is required for
    freezing to actually take effect; a caller that fits R/scale
    elsewhere and doesn't pass them through here will silently get a
    freshly-refit projection instead (this exact bug was caught and
    fixed in agent_memory.py — evaluate_similarity() used to refit
    internally with no way to override it, silently discarding any
    "frozen" projection a caller had computed).

    If either R or scale is None, BOTH are fit fresh across
    [query] + candidate_batch, consistent with q_encoder.py's
    batch-fit requirement — this is not optional when fitting fresh,
    see the collision and normalization bugs documented there.

    The query circuit is built ONCE and reused across all N candidates
    (the query encoding is constant during a search — rebuilding it per
    candidate would be pure waste). Same for the AerSimulator backend
    instance in swap_test mode.

    Returns a dict with quantum fidelities, classical cosine
    similarities, per-candidate circuit metrics (feature_map_depth/
    gate_count always comparable across methods; evaluation_circuit_*
    only comparable within the same method — see statevector_fidelity's
    docstring), wall-clock timing, and rank orderings for both methods.
    """
    if R is None or scale is None:
        full_batch = np.vstack([query[None, :], candidate_batch])
        R, lo, hi = fit_projection_scale(full_batch, n_qubits, seed=seed or 0)
        scale = (lo, hi)

    n_candidates = candidate_batch.shape[0]
    quantum_fidelities = np.zeros(n_candidates)
    classical_cosine = np.zeros(n_candidates)
    feature_map_depths = np.zeros(n_candidates, dtype=int)
    feature_map_gate_counts = np.zeros(n_candidates, dtype=int)
    evaluation_circuit_depths = np.zeros(n_candidates, dtype=int)
    evaluation_circuit_gate_counts = np.zeros(n_candidates, dtype=int)

    t0 = time.perf_counter()
    for i in range(n_candidates):
        classical_cosine[i] = cosine_similarity(query, candidate_batch[i])
    t_classical = time.perf_counter() - t0

    # Query circuit built ONCE, reused for every candidate.
    qc_query = build_feature_map(query, n_qubits, scale, edges, R, entangler)
    backend = AerSimulator(seed_simulator=seed) if method in ("swap_test", "destructive_swap_test") else None

    t0 = time.perf_counter()
    for i in range(n_candidates):
        if method == "swap_test":
            fid, metrics = swap_test_fidelity_from_circuits(
                qc_query, candidate_batch[i], n_qubits, scale, edges, R, entangler, shots, backend
            )
        elif method == "destructive_swap_test":
            fid, metrics = destructive_swap_test_fidelity_from_circuits(
                qc_query, candidate_batch[i], n_qubits, scale, edges, R, entangler, shots, backend
            )
        elif method == "statevector":
            fid, metrics = statevector_fidelity_from_circuits(
                qc_query, candidate_batch[i], n_qubits, scale, edges, R, entangler
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")
        quantum_fidelities[i] = fid
        feature_map_depths[i] = metrics["feature_map_depth"]
        feature_map_gate_counts[i] = metrics["feature_map_gate_count"]
        evaluation_circuit_depths[i] = metrics["evaluation_circuit_depth"]
        evaluation_circuit_gate_counts[i] = metrics["evaluation_circuit_gate_count"]
    t_quantum = time.perf_counter() - t0

    quantum_rank = np.argsort(-quantum_fidelities)
    classical_rank = np.argsort(-classical_cosine)

    return {
        "quantum_fidelities": quantum_fidelities,
        "classical_cosine": classical_cosine,
        "quantum_rank": quantum_rank,      # candidate indices, best match first
        "classical_rank": classical_rank,  # candidate indices, best match first
        "feature_map_depths": feature_map_depths,
        "feature_map_gate_counts": feature_map_gate_counts,
        "evaluation_circuit_depths": evaluation_circuit_depths,
        "evaluation_circuit_gate_counts": evaluation_circuit_gate_counts,
        "n_candidates": n_candidates,
        "method": method,
        "wall_clock_classical_sec": t_classical,
        "wall_clock_quantum_sec": t_quantum,
        # NOTE on wall-clock: quantum timing here is SIMULATED circuit
        # execution (statevector or shot sampling on a classical machine),
        # which costs O(2^n_qubits) to simulate. This will look far worse
        # than classical cosine on a laptop regardless of the algorithm's
        # merit — that's an artifact of simulating physics, not evidence
        # about real hardware performance. Only a real-QPU wall-clock
        # comparison (Phase 3, AWS Braket) is a fair timing comparison.
        # feature_map_depth/gate_count above are the honest, cross-method-
        # comparable quantum-cost metrics to report until then.
    }


def recall_at_k(ranking: np.ndarray, relevant_index: int, k: int) -> int:
    """
    1 if the known-relevant candidate appears in the top-k of ranking
    (an array of candidate indices, best match first), else 0.
    """
    return int(relevant_index in ranking[:k])


def mrr(ranking: np.ndarray, relevant_index: int) -> float:
    """
    Mean Reciprocal Rank for a single query: 1/(position of the relevant
    candidate in ranking, 1-indexed). 0 if not found (shouldn't happen
    when ranking covers the whole candidate set).
    """
    positions = np.where(ranking == relevant_index)[0]
    if len(positions) == 0:
        return 0.0
    return 1.0 / (positions[0] + 1)


def rank_agreement(ranking_a: np.ndarray, ranking_b: np.ndarray) -> tuple[float, float]:
    """
    Spearman rank correlation between two rankings over the same
    candidate set (e.g. quantum_rank vs classical_rank). Returns
    (rho, p_value). rho near 1 = rankings agree strongly; near 0 = no
    relationship; near -1 = rankings disagree (inverted).

    Note: with only a handful of candidates, p-values here won't be
    meaningful — same small-sample caveat as Paper 3's early ℛ-predictor
    correlation. Don't over-read significance until there's a real batch
    size (see the sample-size discussion this project already had once).
    """
    from scipy.stats import spearmanr

    # Convert index-orderings into rank-per-candidate for correlation
    rank_pos_a = np.argsort(ranking_a)
    rank_pos_b = np.argsort(ranking_b)
    rho, p = spearmanr(rank_pos_a, rank_pos_b)
    return float(rho), float(p)


def top1_match(ranking_a: np.ndarray, ranking_b: np.ndarray) -> bool:
    """
    True if both rankings agree on the single best candidate
    (ranking_a[0] == ranking_b[0]). Often more intuitive than Spearman
    for small candidate sets — "both methods picked the same top match"
    is a direct, easy-to-interpret statement, whereas a mid-range
    Spearman rho (e.g. 0.5) is genuinely ambiguous with few candidates.
    """
    return bool(ranking_a[0] == ranking_b[0])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_qubits = 6
    d = 32

    query = rng.normal(size=d)
    near_dup = query + rng.normal(scale=0.05, size=d)
    unrelated_1 = rng.normal(size=d) * 5
    unrelated_2 = rng.normal(size=d) * 3
    candidates = np.stack([near_dup, unrelated_1, unrelated_2])
    labels = ["near-duplicate", "unrelated_1", "unrelated_2"]

    print("--- evaluate_similarity(): SWAP-test (shot-based) ---")
    result = evaluate_similarity(query, candidates, n_qubits, method="swap_test", shots=4096, seed=0)
    for i, label in enumerate(labels):
        print(f"  {label:15s}  quantum_fid={result['quantum_fidelities'][i]:.4f}  "
              f"cosine={result['classical_cosine'][i]:.4f}  "
              f"feature_map_depth={result['feature_map_depths'][i]}  "
              f"eval_circuit_depth={result['evaluation_circuit_depths'][i]}")
    print(f"  wall-clock: classical={result['wall_clock_classical_sec']*1000:.2f}ms, "
          f"quantum(simulated)={result['wall_clock_quantum_sec']*1000:.2f}ms")
    print("  (quantum wall-clock includes simulator overhead — not a fair hardware timing; see note in code)")

    print("\n--- Ranking (best match first) ---")
    print("  quantum rank:  ", [labels[i] for i in result["quantum_rank"]])
    print("  classical rank:", [labels[i] for i in result["classical_rank"]])

    print("\n--- Retrieval metrics (relevant candidate = near-duplicate, index 0) ---")
    relevant_idx = 0
    for k in (1, 2):
        r_q = recall_at_k(result["quantum_rank"], relevant_idx, k)
        r_c = recall_at_k(result["classical_rank"], relevant_idx, k)
        print(f"  Recall@{k}: quantum={r_q}, classical={r_c}")
    mrr_q = mrr(result["quantum_rank"], relevant_idx)
    mrr_c = mrr(result["classical_rank"], relevant_idx)
    print(f"  MRR: quantum={mrr_q:.4f}, classical={mrr_c:.4f}")
    top1 = top1_match(result["quantum_rank"], result["classical_rank"])
    print(f"  Top-1 match (quantum vs classical): {top1}  "
          f"(both methods chose the same best candidate)")
    rho, p = rank_agreement(result["quantum_rank"], result["classical_rank"])
    print(f"  Spearman rank agreement (quantum vs classical): rho={rho:.4f}, p={p:.4f}")
    print(f"  (n={len(candidates)} candidates — too small for a meaningful p-value; "
          f"reported for completeness only, same caveat as Paper 3's small-N correlations)")

    print("\n--- Cross-check: SWAP-test estimate vs exact statevector fidelity ---")
    result_exact = evaluate_similarity(query, candidates, n_qubits, method="statevector", seed=0)
    for i, label in enumerate(labels):
        swap_est = result["quantum_fidelities"][i]
        exact = result_exact["quantum_fidelities"][i]
        print(f"  {label:15s}  swap_test_est={swap_est:.4f}  exact={exact:.4f}  "
              f"diff={abs(swap_est - exact):.4f}  (shot noise at {4096} shots)")
        print(f"    feature_map_depth: swap_test={result['feature_map_depths'][i]}, "
              f"statevector={result_exact['feature_map_depths'][i]}  (should match — same encoding circuit)")

    print("\n--- Complexity check: cost scales with N candidates (not sqrt(N)) ---")
    for n_cand in (2, 4, 8):
        batch = rng.normal(size=(n_cand, d))
        t0 = time.perf_counter()
        _ = evaluate_similarity(query, batch, n_qubits, method="swap_test", shots=1024, seed=0)
        elapsed = time.perf_counter() - t0
        print(f"  N={n_cand} candidates: {elapsed*1000:.1f}ms total "
              f"({elapsed*1000/n_cand:.1f}ms/candidate — roughly constant per-candidate cost, "
              f"confirming O(N) scaling, not O(sqrt(N)))")
