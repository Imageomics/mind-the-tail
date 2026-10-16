"""Neighborhood-weighted log concept-coverage objective.

For image vectors ``x_i`` (max-pooled SAE activations), a cosine k-NN graph
``N_k(t)`` and neighborhood weight ``beta``, every candidate ``i`` receives the
concept evidence of the targets that list it as a neighbor (reverse kNN), with
each target's outgoing edge weights normalized to sum to one::

    e_i = x_i + beta * sum_{t : i in N_k(t)} w_ti x_t,   w_ti = s_ti / sum_{j in N_k(t)} s_tj

The subset objective is the (monotone submodular) log coverage
``F(S) = sum_c log(1 + sum_{i in S} e_i[c])``, maximized by lazy greedy.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse
from numba import njit, prange


@dataclasses.dataclass(frozen=True)
class HnswConfig:
    """Approximate neighbor search: sparse JL projection -> HNSW candidates -> exact rerank."""

    projection_dim: int = 512
    candidates: int = 50
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 100
    threads: int = 1
    seed: int = 0


def l2_normalize_rows(matrix: scipy.sparse.csr_matrix, eps: float = 1e-8) -> scipy.sparse.csr_matrix:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return scipy.sparse.diags(1.0 / np.maximum(norms, eps)).astype(np.float32) @ matrix


def exact_knn(vectors: scipy.sparse.csr_matrix, k: int, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` cosine neighbors (self excluded, similarities clipped at 0)."""
    n = vectors.shape[0]
    k = min(k, n - 1)
    unit = l2_normalize_rows(vectors)
    idx = np.zeros((n, k), dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        block = np.maximum((unit[start : start + chunk] @ unit.T).toarray(), 0)
        block[np.arange(len(block)), np.arange(start, start + len(block))] = -np.inf
        top = np.argpartition(-block, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(block, top, axis=1), axis=1, kind="stable")
        idx[start : start + len(block)] = np.take_along_axis(top, order, axis=1)
        sim[start : start + len(block)] = np.take_along_axis(block, idx[start : start + len(block)], axis=1)
    return idx, sim


def rerank(unit: scipy.sparse.csr_matrix, candidates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact sparse-cosine top-``k`` among each row's candidate ids."""
    n = unit.shape[0]
    idx = np.zeros((n, k), dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    for row in range(n):
        cand = np.unique(candidates[row][(candidates[row] >= 0) & (candidates[row] != row)])
        exact = np.maximum((unit[row] @ unit[cand].T).toarray().ravel(), 0)
        best = np.argsort(-exact, kind="stable")[:k]
        idx[row, : len(best)] = cand[best]
        sim[row, : len(best)] = exact[best]
    return idx, sim


def hnsw_knn(vectors: scipy.sparse.csr_matrix, k: int, cfg: HnswConfig) -> tuple[np.ndarray, np.ndarray]:
    import hnswlib
    from sklearn.random_projection import SparseRandomProjection

    n = vectors.shape[0]
    k = min(k, n - 1)
    unit = l2_normalize_rows(vectors)
    projector = SparseRandomProjection(n_components=cfg.projection_dim, density="auto", random_state=cfg.seed)
    projected = np.asarray(projector.fit_transform(unit).toarray(), dtype=np.float32)
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)

    index = hnswlib.Index(space="cosine", dim=cfg.projection_dim)
    index.init_index(max_elements=n, ef_construction=cfg.ef_construction, M=cfg.m, random_seed=cfg.seed)
    index.add_items(projected, np.arange(n), num_threads=cfg.threads)
    index.set_ef(cfg.ef_search)
    query_k = min(max(cfg.candidates, k) + 1, n)
    candidates, _ = index.knn_query(projected, k=query_k, num_threads=cfg.threads)
    return rerank(unit, candidates.astype(np.int64), k)


def reverse_row_normalized_weights(idx: np.ndarray, sim: np.ndarray, eps: float = 1e-8) -> scipy.sparse.csr_matrix:
    """Sparse ``W[i, t] = w_ti`` for every edge ``t -> i`` of the kNN graph."""
    n, k = idx.shape
    weight = sim / np.maximum(sim.sum(axis=1, keepdims=True), eps)
    targets = np.repeat(np.arange(n), k)
    return scipy.sparse.csr_matrix((weight.ravel(), (idx.ravel(), targets)), shape=(n, n), dtype=np.float32)


def neighborhood_vectors(vectors: scipy.sparse.csr_matrix, idx: np.ndarray, sim: np.ndarray, beta: float) -> scipy.sparse.csr_matrix:
    if beta == 0 or idx.shape[1] == 0:
        return vectors.tocsr()
    weights = reverse_row_normalized_weights(idx, sim)
    return (vectors + np.float32(beta) * (weights @ vectors)).tocsr().astype(np.float32)


# ---------------------------------------------------------------------------
# Compiled lazy greedy over materialized neighborhood vectors (shared with the
# large-scale path). Ties are broken towards the smaller row id.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _higher(score_a, id_a, score_b, id_b) -> bool:
    return score_a > score_b or (score_a == score_b and id_a < id_b)


@njit(cache=True)
def _sift_down(scores, ids, pos, size) -> None:
    while True:
        left = 2 * pos + 1
        if left >= size:
            return
        best = left
        if left + 1 < size and _higher(scores[left + 1], ids[left + 1], scores[left], ids[left]):
            best = left + 1
        if _higher(scores[pos], ids[pos], scores[best], ids[best]):
            return
        scores[pos], scores[best] = scores[best], scores[pos]
        ids[pos], ids[best] = ids[best], ids[pos]
        pos = best


@njit(cache=True)
def _push(scores, ids, size, score, item) -> int:
    pos = size
    while pos:
        parent = (pos - 1) // 2
        if _higher(scores[parent], ids[parent], score, item):
            break
        scores[pos], ids[pos] = scores[parent], ids[parent]
        pos = parent
    scores[pos], ids[pos] = score, item
    return size + 1


@njit(parallel=True, cache=True, fastmath=True)
def marginal_gains(rows, concept_mass, indptr, indices, data) -> np.ndarray:
    """``sum_c log1p(e_i[c] / (1 + mass[c]))`` = log-coverage gain of each row."""
    inverse = (np.float32(1.0) / (np.float32(1.0) + concept_mass)).astype(np.float32)
    out = np.empty(rows.shape[0], dtype=np.float32)
    for pos in prange(rows.shape[0]):
        row = rows[pos]
        gain = np.float32(0.0)
        for p in range(indptr[row], indptr[row + 1]):
            gain += np.log1p(np.float32(data[p] * inverse[indices[p]]))
        out[pos] = gain
    return out


@njit(cache=True, fastmath=True)
def lazy_greedy(initial_gain, target, concept_mass, indptr, indices, data):
    """Select ``target`` rows; returns (order, gains, remaining ids, n_refreshes).

    ``concept_mass`` is updated in place to the coverage of the selected prefix.
    """
    n = initial_gain.shape[0]
    scores = initial_gain.copy()
    ids = np.arange(n, dtype=np.int64)
    for root in range(n // 2 - 1, -1, -1):
        _sift_down(scores, ids, root, n)
    size = n
    inverse = (np.float32(1.0) / (np.float32(1.0) + concept_mass)).astype(np.float32)
    order = np.empty(target, dtype=np.int64)
    gains = np.empty(target, dtype=np.float32)
    selected = refreshes = 0
    while selected < target:
        item = ids[0]
        size -= 1
        scores[0], ids[0] = scores[size], ids[size]
        _sift_down(scores, ids, 0, size)
        gain = np.float32(0.0)
        for p in range(indptr[item], indptr[item + 1]):
            gain += np.log1p(np.float32(data[p] * inverse[indices[p]]))
        refreshes += 1
        upper = scores[0] if size else np.float32(-np.inf)
        if gain + np.float32(1e-8) >= upper:
            order[selected], gains[selected] = item, gain
            for p in range(indptr[item], indptr[item + 1]):
                col = indices[p]
                concept_mass[col] += data[p]
                inverse[col] = np.float32(1.0) / (np.float32(1.0) + concept_mass[col])
            selected += 1
        else:
            size = _push(scores, ids, size, gain, item)
    return order, gains, ids[:size].copy(), refreshes


def rank(vectors: scipy.sparse.csr_matrix, max_fraction: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Full ranking of ``vectors`` (already neighborhood-weighted).

    The first ``floor(max_fraction * n)`` rows follow lazy greedy; the remaining rows
    are ordered by their marginal gain w.r.t. that prefix.
    """
    vectors = vectors.tocsr()
    n = vectors.shape[0]
    indptr = vectors.indptr.astype(np.int64)
    indices = vectors.indices.astype(np.int32)
    data = vectors.data.astype(np.float32)
    mass = np.zeros(vectors.shape[1], dtype=np.float32)
    initial = marginal_gains(np.arange(n, dtype=np.int64), mass, indptr, indices, data)
    order, gains, rest, _ = lazy_greedy(initial, int(n * max_fraction), mass, indptr, indices, data)
    tail_gain = marginal_gains(rest, mass, indptr, indices, data)
    tail = np.argsort(-tail_gain, kind="stable")
    return np.concatenate([order, rest[tail]]), np.concatenate([gains, tail_gain[tail]])

