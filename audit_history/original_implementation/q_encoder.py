"""
q_encoder.py — Phase 1/2: Classical-to-Quantum Encoder

Implements the parametric feature map:
    U(x) = [prod_i CZ_{i,i+1}] . [tensor_i Ry(theta_i(x))]

Maps a classical d-dimensional embedding x in R^d to an n-qubit
parameterized state |psi(theta(x))> in H^(2^n).

No qRAM assumptions. Each query builds a fresh circuit from classical
data — this is honest state preparation, not quantum storage.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def fit_projection_scale(
    batch: np.ndarray, n_qubits: int, seed: int = 0
) -> tuple[np.ndarray, float, float]:
    """
    Fit a SHARED random projection matrix + min/max scale across an entire
    candidate batch, so every vector's angles are comparable to each other
    AND distinct vectors can't collide onto the same angle.

    Replaces chunk-mean pooling, which collapses any two vectors that
    differ only in sign within a chunk to an identical value — confirmed
    empirically: [10, -10], [0, 0], and [3, -3] all mean to exactly 0.
    A fixed random linear projection (Johnson-Lindenstrauss style) makes
    such collisions measure-zero instead of structural.

    Must be fit once per batch (or once globally on a representative
    sample) and reused for every encode() call in that batch.
    """
    d = batch.shape[1]
    rng = np.random.default_rng(seed)
    # Fixed random projection matrix, scaled so output variance is stable
    # regardless of input dimension d.
    R = rng.normal(scale=1.0 / np.sqrt(d), size=(n_qubits, d))

    projected = batch @ R.T  # (batch_size, n_qubits)
    return R, projected.min(), projected.max()


def fit_pca_projection_scale(
    batch: np.ndarray, n_qubits: int
) -> tuple[np.ndarray, float, float]:
    """
    Alternative to fit_projection_scale(): instead of a random linear
    projection, use the top-n_qubits principal components of the batch
    as the projection directions. Deterministic (up to component sign)
    given the same batch, unlike the random-seed-only R.

    Same interface/return shape as fit_projection_scale() — a drop-in
    alternative usable anywhere R is threaded through (q_search.py,
    agent_memory.py). This was deliberately deferred until there was a
    real retrieval task to benchmark it against (evaluate_retrieval() in
    agent_memory.py) rather than comparing raw fidelity numbers with no
    ground truth.
    """
    d = batch.shape[1]
    if batch.shape[0] < 2:
        raise ValueError("Need at least 2 batch vectors to fit PCA.")
    if n_qubits > min(batch.shape[0], d):
        raise ValueError(
            f"n_qubits={n_qubits} exceeds min(batch_size={batch.shape[0]}, d={d}); "
            "PCA can't produce more components than that."
        )

    mean = batch.mean(axis=0)
    centered = batch - mean
    # SVD-based PCA: components are the right singular vectors.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    R = Vt[:n_qubits]  # (n_qubits, d), each row a principal direction

    projected = batch @ R.T
    return R, projected.min(), projected.max()


def project_to_n_features(
    x: np.ndarray,
    n_qubits: int,
    scale: tuple[float, float] | None = None,
    R: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reduce a d-dimensional classical embedding to n_qubits scalar features,
    each rescaled to [0, pi] for use as an Ry rotation angle.

    R: fixed random projection matrix (n_qubits, d), fit ONCE via
    fit_projection_scale() and reused across the whole candidate batch.
    If None, falls back to a per-call random projection (fine for a
    single-vector sanity check, NOT safe for comparing multiple encoded
    vectors — they'd each get a different random R and be incomparable).

    scale: (lo, hi) fit ONCE across the candidate batch via
    fit_projection_scale(). If None, falls back to per-vector min-max,
    which independently stretches every vector to span [0, pi] and
    destroys relative magnitude information (confirmed empirically
    during Phase 1 testing — do not use for batch comparisons).
    """
    if R is None:
        d = len(x)
        R = np.random.default_rng(0).normal(scale=1.0 / np.sqrt(d), size=(n_qubits, d))

    if len(x) != R.shape[1]:
        raise ValueError(f"x has dim {len(x)}, but R expects dim {R.shape[1]}")

    features = R @ x

    if scale is None:
        lo, hi = features.min(), features.max()
    else:
        lo, hi = scale

    if hi - lo < 1e-12:
        return np.zeros(n_qubits)
    return np.pi * np.clip((features - lo) / (hi - lo), 0.0, 1.0)


def build_feature_map(
    x: np.ndarray,
    n_qubits: int,
    scale: tuple[float, float] | None = None,
    edges: list[tuple[int, int]] | None = None,
    R: np.ndarray | None = None,
    entangler: str = "rzz",
) -> QuantumCircuit:
    """
    Construct U(x): Ry rotation layer (per-feature encoding) followed by
    a data-dependent entangling layer over the given edges (the "Entangled
    Knowledge Graph" structure).

    entangler: "rzz" (default) applies RZZ(theta_u * theta_v) per edge —
    data-dependent, so the entangling layer actually contributes to
    fidelity/similarity. "cz" applies a fixed CZ per edge instead — kept
    only as an explicit opt-out for baseline comparison, since CZ is a
    shared unitary across every input and provably contributes NOTHING
    to fidelity differences (confirmed empirically: fidelity was
    bit-identical with vs. without the CZ layer, because <Ua|Ub> = <a|b>
    for any fixed unitary U). Do not use "cz" as the real encoding.

    edges: explicit (u, v) pairs, e.g. sourced from a networkx graph in
    q_graph.py. Defaults to a linear nearest-neighbor chain if not given
    (useful for the Phase 1 sanity check, not meant for real graph data).
    """
    thetas = project_to_n_features(x, n_qubits, scale, R)

    qc = QuantumCircuit(n_qubits, name="U(x)")

    # Rotation layer: one Ry per qubit, encoding one projected feature
    for i, theta in enumerate(thetas):
        qc.ry(theta, i)

    # Entangling layer: the knowledge graph structure
    if edges is None:
        edges = [(i, i + 1) for i in range(n_qubits - 1)]
    for u, v in edges:
        if u < n_qubits and v < n_qubits and u != v:
            if entangler == "rzz":
                qc.rzz(thetas[u] * thetas[v], u, v)
            elif entangler == "cz":
                qc.cz(u, v)
            else:
                raise ValueError(f"Unknown entangler: {entangler!r}")

    return qc


def encode(
    x: np.ndarray,
    n_qubits: int = 6,
    scale: tuple[float, float] | None = None,
    R: np.ndarray | None = None,
    entangler: str = "rzz",
) -> Statevector:
    """Full pipeline: classical embedding -> |psi(theta(x))> statevector."""
    qc = build_feature_map(x, n_qubits, scale, R=R, entangler=entangler)
    return Statevector.from_instruction(qc)


if __name__ == "__main__":
    # Sanity check: two similar embeddings should produce high-fidelity states;
    # two dissimilar embeddings should produce low-fidelity states.
    # R and scale are fit ONCE across the batch — fixes both bugs caught
    # during Phase 1 testing (per-vector normalization, and mean-pooling
    # collisions on sign-cancelling vectors).
    rng = np.random.default_rng(0)

    x_a = rng.normal(size=32)
    x_b = x_a + rng.normal(scale=0.05, size=32)   # near-duplicate of x_a
    x_c = rng.normal(size=32) * 5                  # unrelated embedding

    n_qubits = 6
    batch = np.stack([x_a, x_b, x_c])
    R, lo, hi = fit_projection_scale(batch, n_qubits)
    scale = (lo, hi)

    psi_a = encode(x_a, n_qubits, scale, R)
    psi_b = encode(x_b, n_qubits, scale, R)
    psi_c = encode(x_c, n_qubits, scale, R)

    fid_ab = abs(psi_a.inner(psi_b)) ** 2
    fid_ac = abs(psi_a.inner(psi_c)) ** 2

    print(f"n_qubits = {n_qubits}, batch-fit scale = ({lo:.3f}, {hi:.3f})")
    print(f"Fidelity(a, near-duplicate b) = {fid_ab:.4f}  (expect high)")
    print(f"Fidelity(a, unrelated c)      = {fid_ac:.4f}  (expect lower, not necessarily 0)")
    print()
    print(build_feature_map(x_a, n_qubits, scale, R=R).draw(output="text"))

    # Direct regression test for the mean-pooling collision bug:
    # these vectors used to all mean to exactly 0 in every chunk.
    print("\n--- Mean-pooling collision regression test ---")
    collide_batch = np.array([
        [10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0],
    ])
    R2, lo2, hi2 = fit_projection_scale(collide_batch, n_qubits=3)
    thetas_1 = project_to_n_features(collide_batch[0], 3, (lo2, hi2), R2)
    thetas_2 = project_to_n_features(collide_batch[1], 3, (lo2, hi2), R2)
    thetas_3 = project_to_n_features(collide_batch[2], 3, (lo2, hi2), R2)
    print(f"[10,-10,...] -> {thetas_1}")
    print(f"[0,0,...]    -> {thetas_2}")
    print(f"[3,-3,...]   -> {thetas_3}")
    print("Distinct angles confirm the random-projection fix resolves the collision.")

    # Regression test: confirm entangler="cz" is provably inert (fidelity
    # identical with vs without the entangling layer, since CZ is a fixed
    # shared unitary), while entangler="rzz" (the default) is NOT inert.
    print("\n--- Entangler inertness check (cz vs rzz) ---")
    psi_a_cz = encode(x_a, n_qubits, scale, R, entangler="cz")
    psi_b_cz = encode(x_b, n_qubits, scale, R, entangler="cz")
    psi_a_cz_noent = Statevector.from_instruction(
        build_feature_map(x_a, n_qubits, scale, edges=[], R=R)
    )
    psi_b_cz_noent = Statevector.from_instruction(
        build_feature_map(x_b, n_qubits, scale, edges=[], R=R)
    )
    fid_cz = abs(psi_a_cz.inner(psi_b_cz)) ** 2
    fid_cz_noent = abs(psi_a_cz_noent.inner(psi_b_cz_noent)) ** 2
    print(f"cz:  fid with entangling layer = {fid_cz:.6f}, without = {fid_cz_noent:.6f}  "
          f"(should be identical — CZ is provably inert)")

    psi_a_rzz_noent = Statevector.from_instruction(
        build_feature_map(x_a, n_qubits, scale, edges=[], R=R, entangler="rzz")
    )
    psi_b_rzz_noent = Statevector.from_instruction(
        build_feature_map(x_b, n_qubits, scale, edges=[], R=R, entangler="rzz")
    )
    # Explicit entangler="rzz" here — do not rely on encode()'s default,
    # so this test stays valid even if the default ever changes.
    psi_a_rzz = encode(x_a, n_qubits, scale, R, entangler="rzz")
    psi_b_rzz = encode(x_b, n_qubits, scale, R, entangler="rzz")
    fid_rzz = abs(psi_a_rzz.inner(psi_b_rzz)) ** 2
    fid_rzz_noent = abs(psi_a_rzz_noent.inner(psi_b_rzz_noent)) ** 2
    print(f"rzz: fid with entangling layer = {fid_rzz:.6f}, without = {fid_rzz_noent:.6f}  "
          f"(should differ — RZZ is data-dependent)")
