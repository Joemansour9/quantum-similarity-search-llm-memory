"""
q_graph.py — Phase 2: Entangled Knowledge Graph

Manages the topology (which projected feature-dimensions/qubits get
entangled together) using networkx, and verifies that the resulting
circuit produces REAL physical entanglement — not just gates that look
entangling on paper.

Key distinction this module makes explicit: a two-qubit gate creating
entanglement (a physical fact, checkable via reduced-state purity) is
NOT the same as that entanglement being data-dependent / useful for
discrimination (checked separately in q_encoder.py's inertness test).
For generic Ry angles, CZ does entangle its qubits but that entanglement
is identical for every input with the same angle pattern, so it carries
no discriminative information. CZ does NOT entangle at all when either
qubit's angle is exactly 0 or pi (see the boundary-case test in this
file's __main__ block) — so even the "creates real entanglement" claim
for CZ is angle-dependent, not universal. RZZ(theta_i*theta_j) both
entangles (for generic angles) and is data-dependent.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

from q_encoder import build_feature_map, fit_projection_scale


class KnowledgeGraph:
    """
    Wraps a networkx.Graph over qubit indices (0..n_qubits-1), representing
    which projected feature dimensions are entangled together when a
    classical embedding is encoded via q_encoder.build_feature_map.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n_qubits))

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        if not (0 <= u < self.n_qubits and 0 <= v < self.n_qubits):
            raise ValueError(f"Edge ({u},{v}) out of range for n_qubits={self.n_qubits}")
        self.G.add_edge(u, v, weight=weight)

    def edges(self) -> list[tuple[int, int]]:
        return list(self.G.edges())

    @classmethod
    def chain(cls, n_qubits: int) -> "KnowledgeGraph":
        """Linear nearest-neighbor topology (the Phase 1 default)."""
        kg = cls(n_qubits)
        for i in range(n_qubits - 1):
            kg.add_edge(i, i + 1)
        return kg

    @classmethod
    def from_correlation(
        cls, batch: np.ndarray, R: np.ndarray, n_qubits: int, threshold: float = 0.5
    ) -> "KnowledgeGraph":
        """
        Build topology FROM DATA: project the whole batch through R (the
        same fixed random projection used for encoding), compute the
        Pearson correlation matrix across the n_qubits projected feature
        dimensions, and add an edge wherever |correlation| exceeds
        threshold.

        *** IMPORTANT CAVEAT — READ BEFORE USING ***
        This graph is NOT determined solely by the dataset. The pipeline is:

            embeddings -> random projection R -> correlation -> graph

        Because R is a fixed random matrix, the correlation structure in
        projected coordinates is a joint function of the data AND the
        arbitrary random draw of R, not the data alone. This was confirmed
        empirically: holding the same batch fixed and varying only R's
        random seed (0, 1, 2) produced edge sets of sizes 8, 2, and 3 with
        almost no overlap between them. In other words, "the knowledge
        graph structure" as currently computed is substantially an
        artifact of which random matrix happened to be drawn, not a
        stable property of the embeddings.

        This is not a bug — R is still a valid, reproducible (seeded)
        projection, and this method is not "wrong" — but it means results
        derived from this graph should NOT be described as reflecting the
        dataset's intrinsic structure without qualification. When
        PCA/SVD projection is added later (see q_encoder.py notes), expect
        the resulting "PCA graph" to differ substantially from any given
        "random projection graph" built here, precisely because PCA's
        components are dataset-derived and deterministic (up to sign),
        while R's are not. Any comparison between random-projection and
        PCA topologies should treat this as the primary variable being
        tested, not a nuisance difference.

        batch: (batch_size, d) array of classical embeddings.
        R: (n_qubits, d) projection matrix, must match what encode() uses
        for these vectors — reuse the R from fit_projection_scale().
        """
        projected = batch @ R.T  # (batch_size, n_qubits)
        if projected.shape[0] < 3:
            raise ValueError(
                "Need at least 3 batch vectors for a meaningful correlation matrix; "
                f"got {projected.shape[0]}."
            )
        corr = np.corrcoef(projected.T)  # (n_qubits, n_qubits)

        kg = cls(n_qubits)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                c = corr[i, j]
                if np.isfinite(c) and abs(c) >= threshold:
                    kg.add_edge(i, j, weight=float(c))
        return kg


def reduced_purity(psi: Statevector, keep_qubits: list[int]) -> float:
    """
    Compute Tr(rho_A^2) for the reduced density matrix rho_A on the given
    qubits, tracing out everything else.

    For a PURE global state, purity < 1 on a subsystem is proof that the
    subsystem is entangled with the rest of the register (a mixed
    marginal can only arise from entanglement across the cut, since the
    global state itself has no classical uncertainty to contribute).
    Purity == 1 means that subsystem is in a pure product state relative
    to the rest — i.e., not entangled with it.
    """
    n = psi.num_qubits
    trace_out = [q for q in range(n) if q not in keep_qubits]
    rho = partial_trace(DensityMatrix(psi), trace_out)
    return float(np.real(np.trace(rho.data @ rho.data)))


def verify_entanglement(
    x: np.ndarray,
    n_qubits: int,
    edges: list[tuple[int, int]],
    scale: tuple[float, float] | None = None,
    R: np.ndarray | None = None,
    entangler: str = "rzz",
) -> dict[tuple[int, int], float]:
    """
    For each edge (u, v) in the topology, build the encoded state and
    report the purity of qubit u's reduced density matrix. Purity < 1
    confirms u is genuinely entangled with the rest of the register
    (which, given the graph structure, is attributable to its edges).

    Returns {edge: purity_of_u}. Purity close to 1.0 = essentially not
    entangled (edge isn't doing physical work); purity well below 1.0 =
    genuinely entangled.
    """
    qc = build_feature_map(x, n_qubits, scale, edges=edges, R=R, entangler=entangler)
    psi = Statevector.from_instruction(qc)

    results = {}
    for u, v in edges:
        results[(u, v)] = reduced_purity(psi, [u])
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_qubits = 6

    x_a = rng.normal(size=32)
    x_b = x_a + rng.normal(scale=0.05, size=32)
    x_c = rng.normal(size=32) * 5

    # Need a reasonably sized batch for a meaningful correlation matrix —
    # reuse x_a/x_b/x_c plus extra synthetic samples drawn from the same
    # distribution so from_correlation() has enough data to work with.
    extra = rng.normal(size=(12, 32))
    batch = np.vstack([x_a, x_b, x_c, extra])
    R, lo, hi = fit_projection_scale(batch, n_qubits)
    scale = (lo, hi)

    print("--- Topology construction ---")
    kg_chain = KnowledgeGraph.chain(n_qubits)
    print("Chain edges:", kg_chain.edges())

    kg_corr = KnowledgeGraph.from_correlation(batch, R, n_qubits, threshold=0.5)
    print("Correlation-derived edges (seed=0):", kg_corr.edges())

    # Caveat demonstration: same batch, different R seeds -> different graphs.
    # This is NOT a bug — it's the documented confound in from_correlation()'s
    # docstring, made empirically visible here so it can't be missed.
    print("\n--- R-dependence caveat: same batch, varying only R's seed ---")
    for seed in (0, 1, 2):
        R_s, _, _ = fit_projection_scale(batch, n_qubits, seed=seed)
        kg_s = KnowledgeGraph.from_correlation(batch, R_s, n_qubits, threshold=0.5)
        print(f"  seed={seed}: {sorted(kg_s.edges())}")
    print("  Same dataset, different R -> substantially different topology.")
    print("  The correlation graph is a joint function of data AND R, not data alone.")

    print("\n--- Entanglement verification: CZ (data-independent) ---")
    purities_cz = verify_entanglement(
        x_a, n_qubits, kg_chain.edges(), scale, R, entangler="cz"
    )
    for edge, p in purities_cz.items():
        print(f"  edge {edge}: purity(qubit {edge[0]}) = {p:.4f}")
    print("  (purity < 1 observed here for these particular angles — NOT guaranteed "
          "in general: CZ|psi> stays separable whenever either qubit's Ry angle is "
          "exactly 0 or pi, since Ry(0)=I and Ry(pi) maps to a computational basis "
          "state, and CZ acting on a computational-basis qubit cannot entangle it. "
          "See the boundary-case test below for a direct demonstration. Even when "
          "entanglement IS present, recall it's identical across all inputs that "
          "produce the same angle pattern — see q_encoder.py's inertness test.)")

    print("\n--- CZ boundary case: theta=0 or pi does NOT entangle ---")
    qc_zero = QuantumCircuit(2)
    qc_zero.ry(0.0, 0)
    qc_zero.ry(1.3, 1)
    qc_zero.cz(0, 1)
    psi_zero = Statevector.from_instruction(qc_zero)
    p_zero = reduced_purity(psi_zero, [0])
    print(f"  theta_0=0:  purity(qubit 0) = {p_zero:.6f}  (expect exactly 1.0 — no entanglement)")

    qc_pi = QuantumCircuit(2)
    qc_pi.ry(np.pi, 0)
    qc_pi.ry(1.3, 1)
    qc_pi.cz(0, 1)
    psi_pi = Statevector.from_instruction(qc_pi)
    p_pi = reduced_purity(psi_pi, [0])
    print(f"  theta_0=pi: purity(qubit 0) = {p_pi:.6f}  (expect exactly 1.0 — no entanglement)")
    print("  Confirms: CZ's entangling effect is angle-dependent, not guaranteed for every edge.")

    print("\n--- Entanglement verification: RZZ (data-dependent) ---")
    purities_rzz = verify_entanglement(
        x_a, n_qubits, kg_chain.edges(), scale, R, entangler="rzz"
    )
    for edge, p in purities_rzz.items():
        print(f"  edge {edge}: purity(qubit {edge[0]}) = {p:.4f}")

    print("\n--- No entanglement baseline (empty edge list) ---")
    qc_none = build_feature_map(x_a, n_qubits, scale, edges=[], R=R)
    psi_none = Statevector.from_instruction(qc_none)
    for q in range(n_qubits):
        p = reduced_purity(psi_none, [q])
        print(f"  qubit {q} purity (no entangling layer) = {p:.6f}  (expect ~1.0)")
