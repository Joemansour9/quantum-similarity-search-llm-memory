"""
benchmark_tfidf_realdata.py — Phase 2/3: real-text generalization check

huggingface.co is not reachable here (confirmed empirically —
sentence-transformers installs fine but pretrained weight download fails
with a connection error), so dense sentence embeddings aren't available
here. This is a partial step instead: TF-IDF vectors from real,
locally-authored text across distinct topics.

This is NOT equivalent to dense sentence embeddings — TF-IDF vectors are
sparse (after conversion, high-dimensional and mostly zero), lexical
rather than semantic (no synonym understanding), and the corpus here is
small and hand-written rather than naturally occurring. But they ARE
real, non-Gaussian, anisotropic data, unlike the synthetic Gaussian
blobs used in benchmark_qubit_sweep.py and benchmark_task_difficulty.py
— so this is a genuine (if partial) test of whether those findings
generalize past synthetic data, pending real dense embeddings once
huggingface.co is reachable.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from agent_memory import AgentMemory

# 8 topics, 4 documents each -- real, distinct subject matter, hand-written
# to avoid needing any network access or pretrained models.
TOPICS = {
    "quantum_computing": [
        "Quantum computers use qubits that can exist in superposition of states.",
        "Entanglement allows correlated measurements between distant qubits.",
        "Error correction is a major challenge for near-term quantum hardware.",
        "Variational algorithms combine classical optimization with quantum circuits.",
    ],
    "cooking": [
        "Searing meat at high heat creates a flavorful crust through the Maillard reaction.",
        "Resting dough allows gluten to relax before shaping and baking bread.",
        "Balancing acid and fat is essential for a well rounded sauce.",
        "Fresh herbs should be added near the end of cooking to preserve flavor.",
    ],
    "finance": [
        "Diversification across asset classes reduces overall portfolio risk.",
        "Central banks adjust interest rates to influence inflation and employment.",
        "Compound interest grows an investment exponentially over long time horizons.",
        "Bond prices move inversely to prevailing interest rates in the market.",
    ],
    "sports": [
        "The marathon runner paced herself carefully over the final ten kilometers.",
        "A well executed pick and roll creates open shots in basketball offenses.",
        "Swimmers rely on efficient stroke technique to reduce drag in the water.",
        "The team's defense forced several turnovers in the second half.",
    ],
    "weather": [
        "A cold front moving through the region brought heavy rain and gusty winds.",
        "High pressure systems typically bring clear skies and calm conditions.",
        "Meteorologists track atmospheric moisture to forecast thunderstorm risk.",
        "Coastal areas often experience milder temperature swings than inland regions.",
    ],
    "music": [
        "The orchestra tuned to the oboe's sustained note before the performance began.",
        "Jazz improvisation relies on musicians responding to each other in real time.",
        "A minor key often conveys a more somber emotional tone than a major key.",
        "The guitarist layered rhythm and lead parts to build a fuller sound.",
    ],
    "travel": [
        "Booking flights several weeks in advance often results in lower fares.",
        "Local markets are a good way to experience a city's everyday culture.",
        "Overnight trains can save both time and the cost of a hotel stay.",
        "Travel insurance can cover unexpected medical costs while abroad.",
    ],
    "gardening": [
        "Mulching garden beds helps retain soil moisture during dry summer months.",
        "Companion planting can reduce pest pressure without using pesticides.",
        "Deadheading spent flowers encourages many plants to keep blooming.",
        "Well drained soil is essential for most root vegetables to thrive.",
    ],
}


def perturb_sentence(sentence: str, rng: np.random.Generator, drop_prob: float = 0.25) -> str:
    """
    Simulate a 'noisy query' version of a document by randomly dropping
    words — the TF-IDF analogue of the Gaussian noise added to synthetic
    embeddings in benchmark_projection.py's make_synthetic_corpus().
    """
    words = sentence.split()
    kept = [w for w in words if rng.random() > drop_prob]
    if len(kept) < 3:  # keep at least a few words so it's not degenerate
        kept = words[: max(3, len(words) // 2)]
    return " ".join(kept)


def build_corpus() -> tuple[TfidfVectorizer, dict[str, np.ndarray], list[str], dict[str, str]]:
    """
    Returns (fitted vectorizer, {key: tfidf_vector}, keys, {key: topic}).
    """
    docs, keys, topics_by_key = [], [], {}
    for topic, sentences in TOPICS.items():
        for i, sent in enumerate(sentences):
            key = f"{topic}_{i}"
            docs.append(sent)
            keys.append(key)
            topics_by_key[key] = topic

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs).toarray()  # dense for our encoder

    corpus = {key: tfidf_matrix[i] for i, key in enumerate(keys)}
    return vectorizer, corpus, keys, topics_by_key


def run_benchmark(n_qubits: int = 8, n_queries_per_topic: int = 2, seed: int = 0) -> None:
    vectorizer, corpus, keys, topics_by_key = build_corpus()
    d = next(iter(corpus.values())).shape[0]
    print(f"--- TF-IDF real-text benchmark: {len(corpus)} docs, {len(TOPICS)} topics, d={d} ---\n")

    rng = np.random.default_rng(seed)
    results = {"random": [], "pca": []}

    for projection in ("random", "pca"):
        mem = AgentMemory(n_qubits=n_qubits, topology="chain", method="statevector", projection=projection)
        for key, vec in corpus.items():
            mem.store(key, vec, metadata={"topic": topics_by_key[key]})

        for topic, sentences in TOPICS.items():
            for q_i in range(n_queries_per_topic):
                doc_idx = q_i % len(sentences)
                relevant_key = f"{topic}_{doc_idx}"
                # Perturb the ORIGINAL sentence text, then embed via the SAME
                # fitted vectorizer (transform, not fit_transform) so the
                # query lands in the same TF-IDF space as the corpus.
                perturbed = perturb_sentence(sentences[doc_idx], rng)
                query_vec = vectorizer.transform([perturbed]).toarray()[0]

                metrics = mem.evaluate_retrieval(
                    query_vec, relevant_key=relevant_key, k_values=(1, 3), seed=q_i
                )
                results[projection].append(metrics)

    for projection in ("random", "pca"):
        rows = results[projection]
        recall1_q = np.mean([r["recall@1_quantum"] for r in rows])
        recall1_c = np.mean([r["recall@1_classical"] for r in rows])
        recall3_q = np.mean([r["recall@3_quantum"] for r in rows])
        mrr_q = np.mean([r["mrr_quantum"] for r in rows])
        print(f"[{projection.upper()}]  n={len(rows)} queries")
        print(f"  Recall@1 quantum={recall1_q:.3f}  classical={recall1_c:.3f}")
        print(f"  Recall@3 quantum={recall3_q:.3f}")
        print(f"  MRR quantum={mrr_q:.3f}\n")


if __name__ == "__main__":
    run_benchmark()
    print("--- Comparison against synthetic Gaussian-blob results (n_qubits=8) ---")
    print("Synthetic (benchmark_qubit_sweep.py): random=0.680, pca=0.960")
    print("(See printed results above for the same metric on real TF-IDF text.)")
    print()
    print("--- Limitations of this test ---")
    print("- Small corpus (32 documents, 8 topics) -- not a rigorous benchmark,")
    print("  a first directional check.")
    print("- TF-IDF is lexical/sparse, not semantic/dense like real sentence")
    print("  embeddings (e.g. from a transformer model) -- still not the target")
    print("  test, just a closer approximation than Gaussian blobs.")
    print("- Word-dropping perturbation is a crude proxy for query noise --")
    print("  doesn't test paraphrase robustness, which matters more in practice.")
