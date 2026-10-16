"""One-shot global ranking of a very large unlabelled pool (CC12M, ~11M images).

Same objective as :mod:`selection.select` (global mode); only the kNN graph is
approximated:

1. rows are L2-normalized and projected with a sparse JL matrix,
2. FAISS IVF-PQ proposes ``candidate_count`` neighbors per row,
3. the top-``k`` neighbors and their weights are recomputed with exact sparse cosine.

The row-normalized reverse-neighborhood vectors are then materialized in chunks
and the compiled lazy greedy runs over the whole pool up to ``max_fraction``; the
remaining rows are ordered by their marginal gain w.r.t. that prefix. Every stage
is cached under ``<output>/cache`` so an interrupted run resumes.

    python -m selection.large_scale --activations DIR [DIR ...] --output OUT
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import faiss
import numba
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit, prange
from scipy.sparse import load_npz

from selection.objective import lazy_greedy, marginal_gains

GRAPH_ARGS = ("projection_dim", "seed", "nlist", "pq_m", "pq_bits", "nprobe", "train_vectors", "candidate_count", "k")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", type=pathlib.Path, nargs="+", required=True,
                        help="Image-activation artifact dirs (from sae.infer), concatenated in order.")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-fraction", type=float, default=1.0)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nlist", type=int, default=16_384)
    parser.add_argument("--pq-m", type=int, default=32)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--train-vectors", type=int, default=1_000_000)
    parser.add_argument("--candidate-count", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--augment-chunk-rows", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=56)
    parser.add_argument("--force", action="store_true", help="Discard cached stages.")
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- stage 1: CSR
def build_csr(args: argparse.Namespace, cache: pathlib.Path) -> tuple[int, int]:
    """Concatenate all image shards into one int64-indptr CSR stored as .npy files."""
    info = cache / "csr.json"
    if info.exists():
        meta = json.loads(info.read_text())
        return meta["rows"], meta["features"]
    shards, rows, nnz = [], 0, 0
    for artifact in args.activations:
        table = pq.read_table(artifact / "image_shard_manifest.parquet").sort_by("shard_id")
        shards += [artifact / path for path in table["image_shard_path"].to_pylist()]
        rows += sum(table["n_images"].to_pylist())
        nnz += sum(table["image_nnz"].to_pylist())
    features = load_npz(shards[0]).shape[1]
    indptr = np.lib.format.open_memmap(cache / "indptr.npy", "w+", np.int64, (rows + 1,))
    indices = np.lib.format.open_memmap(cache / "indices.npy", "w+", np.int32, (nnz,))
    data = np.lib.format.open_memmap(cache / "data.npy", "w+", np.float32, (nnz,))
    indptr[0] = row = pos = 0
    for path in shards:
        matrix = load_npz(path).tocsr()
        indices[pos : pos + matrix.nnz] = matrix.indices
        data[pos : pos + matrix.nnz] = matrix.data
        indptr[row + 1 : row + 1 + matrix.shape[0]] = matrix.indptr[1:] + pos
        row, pos = row + matrix.shape[0], pos + matrix.nnz
    for array in (indptr, indices, data):
        array.flush()
    info.write_text(json.dumps({"rows": rows, "nnz": nnz, "features": features}))
    log(f"CSR: {rows:,} rows, {nnz:,} nnz")
    return rows, features


def load_csr(cache: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.load(cache / f"{name}.npy", mmap_mode="r") for name in ("indptr", "indices", "data"))


# ------------------------------------------------------ stage 2: JL projection
@njit(parallel=True, cache=True)
def sparse_jl_project(indptr, indices, data, proj_indptr, proj_indices, proj_data, out) -> None:
    """``out[i] = normalize(normalize(x_i) @ P)`` for a sparse projection matrix ``P`` (CSC)."""
    for row in prange(indptr.shape[0] - 1):
        norm = np.float32(0.0)
        for p in range(indptr[row], indptr[row + 1]):
            norm += data[p] * data[p]
        inv = np.float32(1.0 / math.sqrt(max(float(norm), 1e-20)))
        for p in range(indptr[row], indptr[row + 1]):
            value = data[p] * inv
            col = indices[p]
            for q in range(proj_indptr[col], proj_indptr[col + 1]):
                out[row, proj_indices[q]] += value * proj_data[q]
        norm = np.float32(0.0)
        for c in range(out.shape[1]):
            norm += out[row, c] * out[row, c]
        inv = np.float32(1.0 / math.sqrt(max(float(norm), 1e-20)))
        for c in range(out.shape[1]):
            out[row, c] *= inv


def build_projection(args, cache, rows, features) -> np.ndarray:
    path = cache / "projection.npy"
    if not path.exists():
        from sklearn.random_projection import SparseRandomProjection

        projector = SparseRandomProjection(n_components=args.projection_dim, density="auto", random_state=args.seed)
        components = projector.fit(np.zeros((1, features), dtype=np.float32)).components_.tocsc().astype(np.float32)
        tmp = cache / "projection.tmp.npy"
        out = np.lib.format.open_memmap(tmp, "w+", np.float32, (rows, args.projection_dim))
        sparse_jl_project(*load_csr(cache), components.indptr.astype(np.int64),
                          components.indices.astype(np.int32), components.data, out)
        out.flush()
        tmp.replace(path)
        log("projection done")
    return np.load(path, mmap_mode="r")


# ------------------------------------------------ stage 3: IVF-PQ candidates
@njit(parallel=True, cache=True)
def strip_self(found, start, keep, out) -> None:
    """Copy each query's first ``keep`` non-self hits into ``out``; pad with -1."""
    for local in prange(found.shape[0]):
        row, filled = start + local, 0
        for hit in found[local]:
            if hit >= 0 and hit != row and filled < keep:
                out[row, filled] = hit
                filled += 1
        for pos in range(filled, keep):
            out[row, pos] = -1


def build_candidates(args, cache, projected: np.ndarray) -> np.ndarray:
    path = cache / "candidates.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    rows = projected.shape[0]
    rng = np.random.default_rng(args.seed)
    train = np.asarray(projected[np.sort(rng.choice(rows, min(rows, args.train_vectors), replace=False))])
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(args.projection_dim), args.projection_dim, args.nlist, args.pq_m, args.pq_bits)
    index.train(train)
    for start in range(0, rows, args.batch_size):
        index.add(np.asarray(projected[start : start + args.batch_size]))
    index.nprobe = args.nprobe
    log("IVF-PQ index built")
    tmp = cache / "candidates.tmp.npy"
    out = np.lib.format.open_memmap(tmp, "w+", np.int32, (rows, args.candidate_count))
    for start in range(0, rows, args.batch_size):
        _, found = index.search(np.asarray(projected[start : start + args.batch_size]), args.candidate_count + 1)
        strip_self(found, start, args.candidate_count, out)
    out.flush()
    tmp.replace(path)
    log("candidates done")
    return np.load(path, mmap_mode="r")


# ------------------------------------------------------ stage 4: exact rerank
@njit(parallel=True)  # thread-local buffers prevent caching
def exact_rerank(indptr, indices, data, candidates, norms, n_features, out_idx, out_sim) -> None:
    """Exact sparse-cosine top-k among candidates, using a thread-local dense row lookup."""
    k = out_idx.shape[1]
    lookup = np.zeros((numba.get_num_threads(), n_features), dtype=np.float32)
    marks = np.zeros((numba.get_num_threads(), n_features), dtype=np.int64)
    for row in prange(candidates.shape[0]):
        thread, stamp = numba.get_thread_id(), row + 1
        for p in range(indptr[row], indptr[row + 1]):
            lookup[thread, indices[p]] = data[p]
            marks[thread, indices[p]] = stamp
        best_ids = np.full(k, -1, dtype=np.int32)
        best_sims = np.full(k, np.float32(-1.0), dtype=np.float32)
        for other in candidates[row]:
            if other < 0:
                continue
            dot = np.float32(0.0)
            for p in range(indptr[other], indptr[other + 1]):
                if marks[thread, indices[p]] == stamp:
                    dot += lookup[thread, indices[p]] * data[p]
            sim = max(np.float32(0.0), dot / max(norms[row] * norms[other], np.float32(1e-20)))
            if sim > best_sims[k - 1]:
                pos = k - 1
                while pos > 0 and sim > best_sims[pos - 1]:
                    best_sims[pos], best_ids[pos] = best_sims[pos - 1], best_ids[pos - 1]
                    pos -= 1
                best_sims[pos], best_ids[pos] = sim, other
        out_idx[row], out_sim[row] = best_ids, best_sims


@njit(parallel=True, cache=True)
def row_norms(indptr, data) -> np.ndarray:
    out = np.empty(indptr.shape[0] - 1, dtype=np.float32)
    for row in prange(out.shape[0]):
        total = np.float32(0.0)
        for p in range(indptr[row], indptr[row + 1]):
            total += data[p] * data[p]
        out[row] = math.sqrt(total)
    return out


def build_graph(args, cache, candidates, features) -> tuple[np.ndarray, np.ndarray]:
    idx_path, sim_path = cache / "neighbors.npy", cache / "neighbor_similarity.npy"
    if not sim_path.exists():
        indptr, indices, data = load_csr(cache)
        rows = candidates.shape[0]
        norms = row_norms(indptr, data)
        idx = np.lib.format.open_memmap(idx_path, "w+", np.int32, (rows, args.k))
        tmp = cache / "neighbor_similarity.tmp.npy"
        sim = np.lib.format.open_memmap(tmp, "w+", np.float32, (rows, args.k))
        exact_rerank(indptr, indices, data, candidates, norms, features, idx, sim)
        idx.flush(), sim.flush()
        tmp.replace(sim_path)
        log("exact kNN graph done")
    return np.load(idx_path, mmap_mode="r"), np.load(sim_path, mmap_mode="r")


# ---------------------------------------- stage 5: neighborhood-weighted rows
def reverse_graph(idx: np.ndarray, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR over candidates: for candidate ``i``, targets ``t`` with ``i in N_k(t)`` and weight ``w_ti``."""
    rows, k = idx.shape
    weight = (sim / np.maximum(sim.sum(axis=1, keepdims=True), np.float32(1e-8))).astype(np.float32).ravel()
    flat = np.asarray(idx).ravel()
    valid = flat >= 0
    order = np.flatnonzero(valid)[np.argsort(flat[valid], kind="stable")]
    indptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(flat[valid], minlength=rows), out=indptr[1:])
    return indptr, (order // k).astype(np.int32), weight[order]


@njit(parallel=True)  # thread-local buffers prevent caching
def augmented_rows(start, end, indptr, indices, data, rev_indptr, rev_target, rev_weight, beta, n_features,
                   out_indptr, out_indices, out_data, count_only) -> np.ndarray:
    """Rows ``e_i = x_i + beta * sum_t w_ti x_t`` for ``i`` in ``[start, end)``.

    With ``count_only`` returns the number of distinct columns per row; otherwise
    writes the rows into ``out_*`` at the offsets given by ``out_indptr``.
    """
    threads = numba.get_num_threads()
    marks = np.zeros((threads, n_features), dtype=np.int64)
    values = np.zeros((threads, n_features), dtype=np.float32)
    touched = np.empty((threads, n_features), dtype=np.int32)
    counts = np.empty(end - start, dtype=np.int64)
    for local in prange(end - start):
        thread, stamp, row = numba.get_thread_id(), local + 1, start + local
        count = 0
        for source in range(-1, rev_indptr[row + 1] - rev_indptr[row]):
            if source < 0:
                other, scale = row, np.float32(1.0)
            else:
                edge = rev_indptr[row] + source
                other, scale = rev_target[edge], beta * rev_weight[edge]
            for p in range(indptr[other], indptr[other + 1]):
                col = indices[p]
                if marks[thread, col] != stamp:
                    marks[thread, col] = stamp
                    touched[thread, count] = col
                    values[thread, col] = scale * data[p]
                    count += 1
                else:
                    values[thread, col] += scale * data[p]
        counts[local] = count
        if not count_only:
            base = out_indptr[local]
            for c in range(count):
                col = touched[thread, c]
                out_indices[base + c] = col
                out_data[base + c] = values[thread, col]
    return counts


def build_augmented(args, cache, idx, sim, features) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = f"beta{args.beta:g}"
    paths = [cache / f"augmented_{label}_{name}" for name in ("indptr.npy", "indices.bin", "data.bin")]
    info = cache / f"augmented_{label}.json"
    if not info.exists():
        indptr, indices, data = load_csr(cache)
        rev = reverse_graph(idx, sim)
        rows = idx.shape[0]
        out_indptr = np.lib.format.open_memmap(paths[0], "w+", np.int64, (rows + 1,))
        out_indptr[0] = offset = 0
        empty_i, empty_f = np.empty(0, np.int32), np.empty(0, np.float32)
        with open(paths[1], "wb") as index_file, open(paths[2], "wb") as value_file:
            for start in range(0, rows, args.augment_chunk_rows):
                end = min(start + args.augment_chunk_rows, rows)
                common = (start, end, indptr, indices, data, *rev, np.float32(args.beta), features)
                counts = augmented_rows(*common, np.zeros(1, np.int64), empty_i, empty_f, True)
                local = np.concatenate([[0], np.cumsum(counts)])
                chunk_indices, chunk_data = np.empty(local[-1], np.int32), np.empty(local[-1], np.float32)
                augmented_rows(*common, local, chunk_indices, chunk_data, False)
                chunk_indices.tofile(index_file)
                chunk_data.tofile(value_file)
                out_indptr[start + 1 : end + 1] = local[1:] + offset
                offset += int(local[-1])
        out_indptr.flush()
        info.write_text(json.dumps({"nnz": offset}))
        log(f"augmented rows done ({offset:,} nnz)")
    nnz = json.loads(info.read_text())["nnz"]
    return (np.load(paths[0], mmap_mode="r"), np.memmap(paths[1], np.int32, "r", shape=(nnz,)),
            np.memmap(paths[2], np.float32, "r", shape=(nnz,)))


# ------------------------------------------------------ stage 6: greedy + tail
def rank_pool(args, rows, features, indptr, indices, data) -> tuple[np.ndarray, dict]:
    mass = np.zeros(features, dtype=np.float32)
    ids = np.arange(rows, dtype=np.int64)
    initial = marginal_gains(ids, mass, indptr, indices, data)
    target = int(rows * args.max_fraction)
    started = time.monotonic()
    order, gains, rest, refreshes = lazy_greedy(initial, target, mass, indptr, indices, data)
    tail_gain = marginal_gains(rest, mass, indptr, indices, data)
    ranking = np.concatenate([order, rest[np.argsort(-tail_gain, kind="stable")]])
    if np.unique(ranking).size != rows:
        raise RuntimeError("ranking is not a permutation")
    stats = {"greedy_prefix": target, "refreshes": int(refreshes), "greedy_seconds": time.monotonic() - started}
    return ranking, stats


def write_ranking(args, ranking: np.ndarray) -> pathlib.Path:
    manifests = [pq.read_table(a / "sample_manifest.parquet", columns=["example_idx", "image_path"]).sort_by("example_idx")
                 for a in args.activations]
    manifest = pa.concat_tables([m.drop(["example_idx"]) for m in manifests])
    ranked = manifest.take(pa.array(ranking))
    ranked = ranked.append_column("example_idx", pa.array(ranking, pa.int64()))
    ranked = ranked.append_column("rank_global", pa.array(np.arange(len(ranking), dtype=np.int64)))
    path = args.output / "ranked_examples.parquet"
    pq.write_table(ranked.select(["rank_global", "example_idx", "image_path"]), path, compression="zstd")
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.projection_dim % args.pq_m or args.candidate_count < args.k:
        raise ValueError("need projection_dim % pq_m == 0 and candidate_count >= k")
    cache = args.output / "cache"
    signature = {name: getattr(args, name) for name in GRAPH_ARGS} | {"activations": [str(a) for a in args.activations]}
    sig_path = cache / "graph_args.json"
    if args.force and cache.exists():
        for path in cache.iterdir():
            path.unlink()
    cache.mkdir(parents=True, exist_ok=True)
    if sig_path.exists() and json.loads(sig_path.read_text()) != signature:
        raise ValueError(f"{cache} was built with different graph arguments; pass --force or a new --output.")
    sig_path.write_text(json.dumps(signature, indent=2))
    faiss.omp_set_num_threads(args.threads)
    numba.set_num_threads(args.threads)

    rows, features = build_csr(args, cache)
    candidates = build_candidates(args, cache, build_projection(args, cache, rows, features))
    idx, sim = build_graph(args, cache, candidates, features)
    ranking, stats = rank_pool(args, rows, features, *build_augmented(args, cache, idx, sim, features))
    np.save(args.output / "ranked_example_indices.npy", ranking)
    path = write_ranking(args, ranking)
    summary = {"rows": rows, "beta": args.beta, "k": args.k, "max_fraction": args.max_fraction, **stats,
               "ranked_examples": str(path)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(json.dumps(summary))


if __name__ == "__main__":
    main()
