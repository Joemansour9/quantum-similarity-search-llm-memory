"""
agent_memory.py — Phase 2: Agent Memory Interface

Wraps q_encoder.py / q_graph.py / q_search.py behind a store()/query()
interface, with a design that is honest about the no-cloning theorem:

    store(key, value)  -> CLASSICAL persistence only. No quantum state
                           is created or kept at write time.
    query(vector)      -> Triggers FRESH state preparation (a new
                           circuit built from classical data) + SWAP-test
                           comparison against every stored item. There is
                           no quantum "read" of a previously-stored
                           quantum state, because none is ever kept
                           around — quantum states can't be cloned or
                           persistently read without collapsing them.

*** IMPORTANT CAVEAT: the angle-scale (lo, hi) is refit on every query,
not fixed at store creation time — R itself is NOT. ***
q_search.evaluate_similarity() calls q_encoder.fit_projection_scale()
fresh across [query] + candidate_batch on every call. Checked directly
against fit_projection_scale()'s implementation: the projection matrix R
is drawn from numpy's RNG using ONLY (seed, n_qubits, embedding_dim) —
it does NOT depend on the batch's actual values at all. So calling
query() twice with the same seed produces the SAME R both times,
regardless of what's been stored in between. What DOES depend on the
batch is (lo, hi): the min/max of the projected values, used to rescale
angles into [0, pi]. As the store grows, lo/hi can shift if a new item's
projected values fall outside the previous range — and when they do,
EVERY existing item's angle encoding shifts too (min-max rescaling is
global), even though that item's own embedding never changed.
Consequences:
  - Raw fidelity values for the "same" query are NOT guaranteed to be
    numerically identical across two calls to query() if store growth
    between calls pushed lo/hi outside their previous range. If it
    didn't, values will be identical (same R, same scale) — this is
    not noise, it's a deterministic function of whether the batch's
    projected extremes changed.
  - Rankings should still be self-consistent WITHIN a single query()
    call (all candidates in that call share the same R and scale).
  - Do not assume raw fidelity numbers are comparable across separate
    query() calls unless you've confirmed lo/hi didn't move, or unless
    you've called freeze_projection() (which pins both R and scale).
"""

from __future__ import annotations

import numpy as np

from q_encoder import fit_projection_scale, fit_pca_projection_scale
from q_graph import KnowledgeGraph
from q_search import evaluate_similarity, recall_at_k, mrr, top1_match, rank_agreement


class AgentMemory:
    """
    Classical key/value memory with on-demand quantum similarity search.

    store()/delete() are pure classical dict operations — instant, cheap,
    no quantum circuits involved, no no-cloning issue.

    query() is O(N) in the number of stored items (see q_search.py's
    module docstring — no qRAM, no sub-linear claim). Each query builds
    a fresh quantum state for the query vector and for every stored
    item, then SWAP-tests them pairwise. This is the honest cost model:
    similarity is compared classically-persisted-data-in, quantum-
    comparison-out, every single time — nothing quantum is stored
    between calls.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        entangler: str = "rzz",
        topology: str = "chain",
        correlation_threshold: float = 0.5,
        method: str = "swap_test",
        shots: int = 4096,
        projection: str = "random",
    ):
        self.n_qubits = n_qubits
        self.entangler = entangler
        self.topology = topology
        self.correlation_threshold = correlation_threshold
        self.method = method
        self.shots = shots
        if projection not in ("random", "pca"):
            raise ValueError(f"Unknown projection: {projection!r}")
        self.projection = projection

        self._store: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}

        # If set via freeze_projection(), queries reuse this fixed R/scale
        # instead of refitting per-call — trades the "always fit to
        # current data" property for numerically comparable fidelities
        # across calls. See module docstring caveat.
        self._frozen_R: np.ndarray | None = None
        self._frozen_scale: tuple[float, float] | None = None
        self._frozen_edges: list[tuple[int, int]] | None = None

    def _fit_projection(self, batch: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float, float]:
        if self.projection == "random":
            return fit_projection_scale(batch, self.n_qubits, seed=seed)
        else:
            return fit_pca_projection_scale(batch, self.n_qubits)

    # ---- Classical persistence (no quantum ops) ----

    def store(self, key: str, value: np.ndarray, metadata: dict | None = None) -> None:
        """Classical write. No quantum state is created here."""
        self._store[key] = np.asarray(value, dtype=float)
        self._metadata[key] = metadata or {}

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._metadata.pop(key, None)

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> list[str]:
        return list(self._store.keys())

    # ---- Projection freezing (opt-in fix for the scale-drift caveat) ----

    def freeze_projection(self, seed: int = 0) -> None:
        """
        Fit R/scale ONCE, now, across all currently-stored items, and
        reuse them for every future query() call instead of refitting
        per-call. Trade-off: fidelity values become numerically
        comparable across queries, but the SCALE (lo, hi) will not
        reflect items added after this call as well as a fresh fit
        would — new items get encoded using angles rescaled to a range
        fit before they existed, so their angles may sit outside [0, pi]
        after clipping if their projected values fall outside the old
        (lo, hi). (R itself is unaffected either way — R depends only on
        `seed` and the embedding dimension, never on the data; see
        module docstring.) Call again after major store growth to
        re-freeze against the larger dataset's actual range.

        *** Also freezes the entangling topology (self._edges) at the
        SAME time, using the current corpus. This matters specifically
        for topology="correlation": without freezing edges too,
        "freeze_projection" would pin R/scale but NOT the graph
        structure, since KnowledgeGraph.from_correlation() recomputes
        correlations from whatever's currently stored on every query.
        Confirmed empirically: same frozen R, corpus grown by one item,
        produced a different edge set (15 edges -> 12 edges) for a
        3-candidate correlation-topology store. So "projection frozen"
        did NOT previously imply "retrieval representation frozen" for
        correlation mode — this method now freezes both together so
        that implication actually holds. For topology="chain", edges
        never depended on the corpus anyway, so this is a no-op there.
        """
        if len(self._store) < 2:
            raise ValueError("Need at least 2 stored items to fit a projection.")
        batch = np.stack(list(self._store.values()))
        R, lo, hi = self._fit_projection(batch, seed=seed)
        self._frozen_R = R
        self._frozen_scale = (lo, hi)
        self._frozen_edges = self._edges(batch, R)

    def unfreeze_projection(self) -> None:
        """Return to per-query fresh-fit behavior (the default) for R, scale, AND edges."""
        self._frozen_R = None
        self._frozen_scale = None
        self._frozen_edges = None

    # ---- Quantum comparison (on-demand, nothing persisted) ----

    def _edges(self, batch: np.ndarray, R: np.ndarray) -> list[tuple[int, int]]:
        if self.topology == "chain":
            return KnowledgeGraph.chain(self.n_qubits).edges()
        elif self.topology == "correlation":
            kg = KnowledgeGraph.from_correlation(
                batch, R, self.n_qubits, threshold=self.correlation_threshold
            )
            return kg.edges()
        else:
            raise ValueError(f"Unknown topology: {self.topology!r}")

    def query(self, vector: np.ndarray, top_k: int = 5, seed: int | None = None) -> dict:
        """
        Compare `vector` against every stored item via SWAP-test (or
        exact statevector fidelity, if method="statevector"), and
        return the top_k matches ranked by quantum fidelity.

        This is O(N) in len(self) — see module and q_search.py docstrings.
        No item is read from quantum memory; every item's classical
        embedding is re-encoded fresh for this comparison.
        """
        if len(self._store) == 0:
            return {
                "keys": [], "quantum_fidelities": np.array([]),
                "classical_cosine": np.array([]), "n_candidates": 0,
            }

        keys = list(self._store.keys())
        candidate_batch = np.stack([self._store[k] for k in keys])

        if self._frozen_R is not None:
            R, scale = self._frozen_R, self._frozen_scale
            edges = self._frozen_edges
        else:
            full_batch = np.vstack([np.asarray(vector)[None, :], candidate_batch])
            R, lo, hi = self._fit_projection(full_batch, seed=seed or 0)
            scale = (lo, hi)
            edges = self._edges(candidate_batch, R)

        result = evaluate_similarity(
            np.asarray(vector, dtype=float),
            candidate_batch,
            self.n_qubits,
            edges=edges,
            entangler=self.entangler,
            method=self.method,
            shots=self.shots,
            seed=seed,
            R=R,
            scale=scale,
        )

        top_k = min(top_k, len(keys))
        top_indices = result["quantum_rank"][:top_k]

        return {
            "keys": [keys[i] for i in top_indices],
            "metadata": [self._metadata[keys[i]] for i in top_indices],
            "quantum_fidelities": result["quantum_fidelities"][top_indices],
            "classical_cosine": result["classical_cosine"][top_indices],
            "all_keys": keys,
            "all_quantum_fidelities": result["quantum_fidelities"],
            "all_classical_cosine": result["classical_cosine"],
            "quantum_rank": result["quantum_rank"],
            "classical_rank": result["classical_rank"],
            "n_candidates": result["n_candidates"],
            "projection_frozen": self._frozen_R is not None,
        }

    def evaluate_retrieval(
        self, vector: np.ndarray, relevant_key: str, k_values: tuple[int, ...] = (1, 5), seed: int | None = None
    ) -> dict:
        """
        Convenience: run query() against the FULL store (top_k = len(self))
        and score it against a known-relevant key, using q_search.py's
        Recall@K / MRR / Top-1 / Spearman metrics. For offline evaluation
        with labeled data, not for normal retrieval use.
        """
        result = self.query(vector, top_k=len(self), seed=seed)
        keys = result["all_keys"]
        if relevant_key not in keys:
            raise ValueError(f"relevant_key {relevant_key!r} not found in store.")
        relevant_idx = keys.index(relevant_key)

        metrics = {}
        for k in k_values:
            metrics[f"recall@{k}_quantum"] = recall_at_k(result["quantum_rank"], relevant_idx, k)
            metrics[f"recall@{k}_classical"] = recall_at_k(result["classical_rank"], relevant_idx, k)
        metrics["mrr_quantum"] = mrr(result["quantum_rank"], relevant_idx)
        metrics["mrr_classical"] = mrr(result["classical_rank"], relevant_idx)
        metrics["top1_match"] = top1_match(result["quantum_rank"], result["classical_rank"])
        rho, p = rank_agreement(result["quantum_rank"], result["classical_rank"])
        metrics["spearman_rho"] = rho
        metrics["spearman_p"] = p
        return metrics


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 32

    mem = AgentMemory(n_qubits=6, topology="chain", method="swap_test", shots=4096)

    print("--- store(): classical writes, no quantum ops ---")
    base = rng.normal(size=d)
    mem.store("doc_original", base, metadata={"title": "Original document"})
    mem.store("doc_near_dup", base + rng.normal(scale=0.05, size=d), metadata={"title": "Near-duplicate"})
    mem.store("doc_unrelated_1", rng.normal(size=d) * 5, metadata={"title": "Unrelated A"})
    mem.store("doc_unrelated_2", rng.normal(size=d) * 3, metadata={"title": "Unrelated B"})
    mem.store("doc_unrelated_3", rng.normal(size=d) * 4, metadata={"title": "Unrelated C"})
    print(f"  Stored {len(mem)} items: {mem.keys()}")

    print("\n--- query(): fresh state prep + SWAP-test against all stored items ---")
    query_vec = base + rng.normal(scale=0.02, size=d)  # simulates re-querying near the original
    result = mem.query(query_vec, top_k=3, seed=0)
    for key, meta, fid, cos in zip(result["keys"], result["metadata"], result["quantum_fidelities"], result["classical_cosine"]):
        print(f"  {key:18s} ({meta['title']:20s})  quantum_fid={fid:.4f}  cosine={cos:.4f}")

    print("\n--- evaluate_retrieval(): scored against known-relevant key ---")
    metrics = mem.evaluate_retrieval(query_vec, relevant_key="doc_near_dup", k_values=(1, 3), seed=0)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Scale-drift caveat demonstration: same query, store grows between calls ---")
    # Use a NEAR-duplicate query, not an exact copy of a stored vector —
    # exact matches give fidelity=1.0 regardless of scale (trivial
    # self-fidelity), which would mask the drift this demo is meant to show.
    q_fixed = base + rng.normal(scale=0.1, size=d)
    result_before = mem.query(q_fixed, top_k=1, seed=1)
    fid_before = result_before["quantum_fidelities"][0]
    # A genuine outlier (large magnitude) is needed to actually push the
    # batch's min/max range — a typical-magnitude addition (as in the
    # first version of this demo) may fall inside the existing range and
    # change nothing, which would make this demo misleadingly show no
    # drift ever occurs. Confirmed by testing: R itself never changes
    # (seed-determined only); only (lo, hi) can shift, and only if the
    # new item's projected values exceed the existing extremes.
    outlier = rng.normal(size=d) * 50
    mem.store("doc_outlier", outlier, metadata={"title": "Extreme outlier"})
    result_after = mem.query(q_fixed, top_k=1, seed=1)
    fid_after = result_after["quantum_fidelities"][0]
    print(f"  Top-1 fidelity BEFORE adding outlier: {fid_before:.6f}")
    print(f"  Top-1 fidelity AFTER adding outlier:  {fid_after:.6f}")
    print(f"  Same top-1 key both times: {result_before['keys'][0] == result_after['keys'][0]}")
    print(f"  Values differ: {fid_before != fid_after}")
    print("  (an outlier shifting the batch's min/max range changes EVERY existing "
          "item's angle encoding, since min-max scaling is global — not a bug, "
          "but a real consequence of per-query rescaling worth knowing about)")

    print("\n--- freeze_projection(): opt-in fix for numerically comparable fidelities ---")
    mem.freeze_projection(seed=0)
    result_frozen_1 = mem.query(q_fixed, top_k=1, seed=2)
    another_outlier = rng.normal(size=d) * 60
    mem.store("doc_outlier_2", another_outlier, metadata={"title": "Another outlier"})
    result_frozen_2 = mem.query(q_fixed, top_k=1, seed=2)
    print(f"  Frozen fidelity before growth: {result_frozen_1['quantum_fidelities'][0]:.6f}")
    print(f"  Frozen fidelity after growth:  {result_frozen_2['quantum_fidelities'][0]:.6f}")
    print(f"  Identical (as expected, scale is pinned): "
          f"{result_frozen_1['quantum_fidelities'][0] == result_frozen_2['quantum_fidelities'][0]}")
    print("  (with the projection frozen, even an extreme outlier doesn't perturb "
          "existing items' encodings — contrast with the drift observed above when "
          "scale refits per-query; trade-off: doc_outlier_2 itself will be encoded "
          "using a range that doesn't account for its own extreme values)")
