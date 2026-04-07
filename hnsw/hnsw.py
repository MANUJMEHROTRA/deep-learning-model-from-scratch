import numpy as np
import heapq
from typing import Optional


class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) graph for Approximate
    Nearest Neighbour (ANN) search.

    Based on: Malkov & Yashunin, "Efficient and robust approximate nearest
    neighbor search using Hierarchical Navigable Small World graphs", 2018.
    (https://arxiv.org/abs/1603.09320)

    Parameters
    ----------
    M : int
        Maximum number of bidirectional links per node at layers > 0.
        Higher M → better recall, more memory and slower insert.
        Recommended range: 5 – 48.  Paper default: 16.
    ef_construction : int
        Size of the dynamic candidate list during index construction.
        Higher ef_construction → better recall at build time, slower inserts.
        Must be >= M.  Paper default: 200.
    M0 : int | None
        Maximum connections at layer 0 (default: 2 * M).
        Layer 0 has denser connectivity to ensure recall near the query.
    mL : float | None
        Level-generation multiplier.  Defaults to 1 / ln(M), which gives
        an expected O(log n) number of layers (same as skip lists).
    distance : str
        Distance metric — 'euclidean' (L2) or 'cosine'.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        M0: Optional[int] = None,
        mL: Optional[float] = None,
        distance: str = "euclidean",
    ):
        if distance not in ("euclidean", "cosine"):
            raise ValueError("distance must be 'euclidean' or 'cosine'")
        self.M = M
        self.ef_construction = max(ef_construction, M)
        self.M0 = M0 if M0 is not None else 2 * M
        self.mL = mL if mL is not None else 1.0 / np.log(M)
        self.distance = distance

        # Core data structures
        self._vectors: list[np.ndarray] = []   # raw stored vectors
        self._levels: list[int] = []            # max layer each node lives on
        # graph[layer][node_id] = list[int] of neighbour node ids
        self._graph: list[dict[int, list[int]]] = []
        self._entry_point: Optional[int] = None
        self._max_level: int = -1

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.distance == "euclidean":
            diff = a - b
            return float(np.dot(diff, diff) ** 0.5)
        else:  # cosine distance = 1 - cosine_similarity
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0.0:
                return 1.0
            return float(1.0 - np.dot(a, b) / denom)

    # ------------------------------------------------------------------
    # Level generation
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """
        Sample a level for a new node from a geometric distribution.
        Mirrors the skip-list construction: P(level >= l) = (1/M)^l.
        """
        return int(-np.log(np.random.random() + 1e-10) * self.mL)

    # ------------------------------------------------------------------
    # Core search primitive — SEARCH-LAYER (Algorithm 2 in the paper)
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        query: np.ndarray,
        entry_point: int,
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """
        Greedy beam search within a single layer.

        Returns a list of (distance, node_id) pairs — the ef closest nodes
        found, sorted by distance ascending.

        Data structures:
          candidates  — min-heap of (dist, id): points to explore next
          found       — max-heap of (-dist, id): best ef points seen so far
          visited     — set of already-processed node ids
        """
        d_ep = self._dist(query, self._vectors[entry_point])

        # Min-heap: (dist, id) — pop gives closest candidate
        candidates: list[tuple[float, int]] = [(d_ep, entry_point)]

        # Max-heap via negation: (-dist, id) — pop gives FARTHEST in found
        found: list[tuple[float, int]] = [(-d_ep, entry_point)]

        visited: set[int] = {entry_point}

        while candidates:
            d_c, c = heapq.heappop(candidates)   # closest unexplored candidate

            # Farthest distance currently in found
            d_f = -found[0][0]

            # If the closest candidate is farther than the farthest found,
            # no candidate can improve found → terminate
            if d_c > d_f:
                break

            layer_neighbors = self._graph[layer].get(c, [])

            for nb in layer_neighbors:
                if nb in visited:
                    continue
                visited.add(nb)

                d_nb = self._dist(query, self._vectors[nb])
                d_f = -found[0][0]   # refresh (found may have changed)

                if d_nb < d_f or len(found) < ef:
                    heapq.heappush(candidates, (d_nb, nb))
                    heapq.heappush(found, (-d_nb, nb))

                    if len(found) > ef:
                        heapq.heappop(found)   # remove farthest

        # Convert max-heap to ascending list
        results = [(-neg_d, node_id) for neg_d, node_id in found]
        results.sort(key=lambda x: x[0])
        return results

    # ------------------------------------------------------------------
    # Neighbour selection — SELECT-NEIGHBORS-SIMPLE (Algorithm 3)
    # ------------------------------------------------------------------

    def _select_neighbors(
        self,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        """
        Return the ids of the M nearest nodes from candidates
        (already sorted ascending by distance).
        """
        return [node_id for _, node_id in candidates[:M]]

    # ------------------------------------------------------------------
    # INSERT (Algorithm 1 in the paper)
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray) -> "HNSW":
        """
        Insert a single vector into the index.

        Steps
        -----
        1. Sample a random level l for the new node.
        2. From the top of the graph down to l+1, greedily descend (ef=1)
           to find the best entry point for the insertion layers.
        3. From level l down to 0, run full beam search (ef=ef_construction),
           select M neighbours, add bidirectional links, and prune if needed.
        4. If l > current max level, update the global entry point.
        """
        vector = np.array(vector, dtype=float)
        idx = len(self._vectors)
        self._vectors.append(vector)

        level = self._random_level()
        self._levels.append(level)

        # Extend graph layer list
        while len(self._graph) <= level:
            self._graph.append({})
        for l in range(level + 1):
            self._graph[l][idx] = []

        # ── First node: just set entry point ────────────────────────────
        if self._entry_point is None:
            self._entry_point = idx
            self._max_level = level
            return self

        ep = self._entry_point

        # ── Phase 1: descend from max_level to level+1 (ef=1 greedy) ───
        for l in range(self._max_level, level, -1):
            if l >= len(self._graph):
                continue
            found = self._search_layer(vector, ep, ef=1, layer=l)
            ep = found[0][1]   # nearest point at this layer

        # ── Phase 2: insert from level down to 0 ────────────────────────
        for l in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(vector, ep, ef=self.ef_construction, layer=l)

            Mmax = self.M0 if l == 0 else self.M
            neighbors = self._select_neighbors(found, Mmax)

            # Add outgoing edges from idx
            self._graph[l][idx] = neighbors

            # Add back-edges; prune neighbours that exceed Mmax
            for nb in neighbors:
                self._graph[l].setdefault(nb, []).append(idx)

                if len(self._graph[l][nb]) > Mmax:
                    # Keep only the Mmax nearest to nb
                    nb_vec = self._vectors[nb]
                    ranked = sorted(
                        self._graph[l][nb],
                        key=lambda c: self._dist(nb_vec, self._vectors[c]),
                    )
                    self._graph[l][nb] = ranked[:Mmax]

            # Best point found becomes entry point for the next (lower) layer
            ep = found[0][1]

        # ── Update global entry point if new node lives higher ───────────
        if level > self._max_level:
            self._entry_point = idx
            self._max_level = level

        return self

    # ------------------------------------------------------------------
    # K-NN SEARCH (Algorithm 5 in the paper)
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 1,
        ef: Optional[int] = None,
    ) -> tuple[list[int], list[float]]:
        """
        Find the k approximate nearest neighbours of query.

        Parameters
        ----------
        query : array-like of shape (d,)
        k     : number of neighbours to return
        ef    : search-time candidate list size (>= k).
                Larger ef → higher recall, slower search.
                Defaults to max(k, ef_construction).

        Returns
        -------
        indices   : list[int]  — indices of k nearest vectors
        distances : list[float]
        """
        if self._entry_point is None:
            raise RuntimeError("Index is empty. Call add() or build() first.")

        if ef is None:
            ef = max(k, self.ef_construction)

        query = np.array(query, dtype=float)
        ep = self._entry_point

        # Greedy descent through layers above 0
        for l in range(self._max_level, 0, -1):
            if l >= len(self._graph):
                continue
            found = self._search_layer(query, ep, ef=1, layer=l)
            ep = found[0][1]

        # Full beam search at layer 0
        found = self._search_layer(query, ep, ef=ef, layer=0)

        top_k = found[:k]
        indices = [node_id for _, node_id in top_k]
        distances = [d for d, _ in top_k]
        return indices, distances

    # ------------------------------------------------------------------
    # Batch build
    # ------------------------------------------------------------------

    def build(self, X: np.ndarray) -> "HNSW":
        """Insert all rows of X into the index."""
        for vec in X:
            self.add(vec)
        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary of the index structure."""
        n = len(self._vectors)
        layer_sizes = {l: len(nodes) for l, nodes in enumerate(self._graph)}
        avg_degree_l0 = (
            np.mean([len(nb) for nb in self._graph[0].values()])
            if self._graph
            else 0.0
        )
        return {
            "n_vectors": n,
            "n_layers": self._max_level + 1,
            "layer_sizes": layer_sizes,
            "avg_degree_layer0": float(avg_degree_l0),
            "entry_point": self._entry_point,
            "max_level": self._max_level,
        }


# ---------------------------------------------------------------------------
# Recall evaluation helper
# ---------------------------------------------------------------------------

def compute_recall(
    index: HNSW,
    queries: np.ndarray,
    ground_truth: np.ndarray,   # (n_queries, k) true nearest-neighbour ids
    k: int,
    ef: int,
) -> float:
    """
    Recall@k: fraction of true top-k neighbours returned by the ANN search.
    """
    hits = 0
    total = queries.shape[0] * k
    for i, q in enumerate(queries):
        approx_ids, _ = index.search(q, k=k, ef=ef)
        hits += len(set(approx_ids) & set(ground_truth[i]))
    return hits / total


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Build index ────────────────────────────────────────────────────────
    D = 32          # vector dimensionality
    N = 2_000       # index size
    N_QUERY = 200   # number of queries
    K = 10          # neighbours to retrieve

    X = rng.standard_normal((N, D)).astype(np.float32)

    print(f"Building HNSW index (N={N}, D={D}, M=16, ef_construction=100)...")
    index = HNSW(M=16, ef_construction=100, distance="euclidean")
    index.build(X)
    s = index.stats()
    print(f"  Layers             : {s['n_layers']}")
    print(f"  Layer sizes        : {s['layer_sizes']}")
    print(f"  Avg degree (L0)    : {s['avg_degree_layer0']:.1f}")

    # Ground truth via brute-force ───────────────────────────────────────
    queries = rng.standard_normal((N_QUERY, D)).astype(np.float32)

    # Pairwise distances: (N_QUERY, N)
    diff = queries[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    ground_truth = np.argsort(dists, axis=1)[:, :K]    # (N_QUERY, K)

    # Evaluate at different ef values ────────────────────────────────────
    print(f"\nRecall@{K} vs ef (search-time candidate list size):")
    print(f"  {'ef':>6}  {'Recall@K':>10}")
    print(f"  {'──':>6}  {'────────':>10}")
    for ef in [10, 20, 50, 100, 200]:
        recall = compute_recall(index, queries, ground_truth, k=K, ef=ef)
        print(f"  {ef:>6}  {recall:>10.4f}")

    # Single query example ───────────────────────────────────────────────
    q = queries[0]
    ids, dists_approx = index.search(q, k=5, ef=50)
    true_ids = ground_truth[0].tolist()
    print(f"\nQuery 0 — top-5 approximate neighbours : {ids}")
    print(f"Query 0 — top-5 true      neighbours : {true_ids[:5]}")