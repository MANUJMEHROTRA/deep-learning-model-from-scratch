# ANN via HNSW — from scratch with NumPy

## Overview

**Approximate Nearest Neighbour (ANN)** search trades a small amount of recall for dramatically faster query times compared to exact nearest-neighbour search. This implementation uses the **Hierarchical Navigable Small World (HNSW)** graph algorithm, the state-of-the-art method behind modern vector databases (Pinecone, Weaviate, pgvector, FAISS, etc.).

**Reference**: Malkov & Yashunin, *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"*, IEEE TPAMI 2020. ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320))

---

## The Problem: Why Not Exact Search?

Exact k-NN requires computing the distance from a query to every stored vector — **O(n · d)** per query. At scale this is prohibitive:

| Scale | Vectors | Dims | Brute-force latency |
|---|---|---|---|
| Small | 10 K | 128 | ~2 ms |
| Medium | 1 M | 512 | ~500 ms |
| Large | 1 B | 1536 | ~10 minutes |

HNSW achieves **O(log n)** average query time with >95% recall.

---

## Navigable Small World Graphs

The foundation of HNSW is the **Navigable Small World (NSW)** graph. Each node (vector) is connected to a fixed number of its nearest neighbours. A greedy search traverses the graph from an entry point, always moving to the neighbour closest to the query:

```
Start at entry point e
Repeat:
  For each neighbour n of current node c:
    If dist(n, query) < dist(c, query):
      Move to n  (greedy step)
Until no improvement possible
```

A plain NSW graph can get trapped in local minima. HNSW solves this with a **hierarchical layer structure**.

---

## HNSW Architecture

```
Layer 2  ●────────────────────●              (few long-range links)
            ╲                ╱
Layer 1  ●───●──────────●───●──●            (medium links)
          ╲  ╲         ╱   ╱  ╱
Layer 0  ●─●──●──●──●──●──●──●──●──●       (all nodes, short links)
```

- **Layer 0** contains every node and has the densest local connectivity.
- **Higher layers** contain exponentially fewer nodes (like a skip list) and act as "highways" for long-range navigation.
- Search starts at the top layer and descends, refining the candidate set at each layer.

The probability that a node appears at layer l is:

```
P(level ≥ l) = (1/M)^l
```

giving O(log n) expected layers.

---

## Mathematical Foundation

### Level Generation

A new node's maximum layer is sampled from a geometric distribution:

```
level = ⌊ -ln(uniform(0,1)) · mL ⌋

mL = 1 / ln(M)    (default)
```

This mirrors the skip-list construction and ensures E[n_layers] = O(log n).

### SEARCH-LAYER (Algorithm 2)

The core primitive. Given a query **q**, entry point **ep**, candidate list size **ef**, and layer **l**:

```
v ← {ep}                           // visited set
C ← min-heap {(dist(ep,q), ep)}    // candidates to explore
W ← max-heap {(dist(ep,q), ep)}    // best ef points found

while C not empty:
  c ← extract_min(C)              // closest unexplored candidate
  f ← peek_max(W)                 // farthest point in W

  if dist(c, q) > dist(f, q):
    break                          // W cannot improve further

  for each neighbour e of c at layer l:
    if e ∉ v:
      v ← v ∪ {e}
      if dist(e, q) < dist(f, q) or |W| < ef:
        C ← C ∪ {(dist(e,q), e)}
        W ← W ∪ {(dist(e,q), e)}
        if |W| > ef:
          remove farthest from W

return W
```

**Why this works**: The algorithm maintains a set of the `ef` globally closest points found so far (`W`) while exploring through `C`. Termination is exact within the layer: once the closest candidate is farther than the farthest known point, no further improvement is possible.

### INSERT (Algorithm 1)

```
INSERT(q, M, ef_construction):
  level l ← random_level()
  ep ← global entry point

  // Phase 1: fast descent above insertion level
  for lc from max_level down to l+1:
    W ← SEARCH-LAYER(q, ep, ef=1, lc)
    ep ← nearest element in W

  // Phase 2: insert and connect at each of the node's layers
  for lc from min(l, max_level) down to 0:
    W ← SEARCH-LAYER(q, ep, ef=ef_construction, lc)
    Mmax ← M0 if lc==0 else M
    neighbours ← SELECT-NEIGHBORS(W, Mmax)   // M nearest from W
    add bidirectional edges: q ↔ neighbours
    for each nb in neighbours:
      if |neighbours(nb)| > Mmax:
        prune nb's connections to Mmax nearest
    ep ← nearest element in W

  if l > max_level:
    entry_point ← q
    max_level ← l
```

**Phase 1** navigates from the top of the graph down to the new node's level using a fast greedy walk (ef=1 is exact within a layer for the single best candidate).

**Phase 2** does a full beam search at each of the node's layers to find the best neighbours and wires bidirectional edges.

### K-NN-SEARCH (Algorithm 5)

```
SEARCH(q, K, ef):
  ep ← global entry point

  for lc from max_level down to 1:
    W ← SEARCH-LAYER(q, ep, ef=1, lc)
    ep ← nearest element in W

  W ← SEARCH-LAYER(q, ep, ef=ef, lc=0)
  return K nearest elements from W
```

---

## Implementation Details

File: [hnsw.py](hnsw.py)

| Component | Description |
|---|---|
| `_dist(a, b)` | Euclidean or cosine distance between two vectors |
| `_random_level()` | Geometric level sampling via `-log(uniform) × mL` |
| `_search_layer(q, ep, ef, layer)` | Core beam search — returns sorted list of (dist, id) |
| `_select_neighbors(candidates, M)` | Return M nearest from already-sorted candidate list |
| `add(vector)` | Insert a single vector (full Algorithm 1) |
| `build(X)` | Batch insert all rows of X |
| `search(query, k, ef)` | ANN query (Algorithm 5) |
| `stats()` | Index diagnostics: layer sizes, average degree, etc. |
| `compute_recall(...)` | Module-level helper: Recall@K vs. brute-force ground truth |

### Heap Representation

Python's `heapq` is a **min-heap**. The `found` (W) set must behave as a **max-heap** (to efficiently remove the farthest element). This is achieved by storing **negated distances**:

```python
# found stores (-distance, node_id)
# heappop(found) removes the element with smallest -dist = farthest point ✓
# found[0] is the element with smallest -dist = farthest point ✓

d_f = -found[0][0]   # farthest distance in W
```

---

## Step-by-Step Walkthrough

```
BUILD PHASE (add each vector):

For vector q at index i:
  1. Sample level l = ⌊ -ln(rand) / ln(M) ⌋
  2. Extend graph layers if l > current max_level
  3. Phase 1 — descend from top to l+1 with ef=1:
       ep ← greedy nearest at each layer
  4. Phase 2 — from l down to 0:
       candidates ← SEARCH-LAYER(q, ep, ef=ef_construction, layer)
       select M nearest as neighbours
       add edges q→nb and nb→q
       prune nb's connections if > Mmax
       ep ← nearest in candidates (for next layer)
  5. If l > max_level: update entry point

SEARCH PHASE (query):

  1. Start from global entry point
  2. Greedy descent from max_level to 1 (ef=1)
  3. Full beam search at layer 0 with ef=ef_search
  4. Return K nearest from beam search result
```

---

## Usage

```python
import numpy as np
from hnsw import HNSW, compute_recall

# Build index
index = HNSW(M=16, ef_construction=200, distance='euclidean')
index.build(X_train)        # or: for v in X_train: index.add(v)

# Query
ids, dists = index.search(query_vector, k=10, ef=50)

# Index statistics
print(index.stats())

# Evaluate recall vs. brute-force ground truth
recall = compute_recall(index, queries, ground_truth_ids, k=10, ef=100)
print(f"Recall@10: {recall:.4f}")
```

Run the built-in demo (N=2000, D=32, evaluates Recall@10 at multiple ef values):

```bash
python hnsw.py
```

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `M` | `16` | Connections per node per layer. Higher → better recall, more memory. Range: 5–48 |
| `ef_construction` | `200` | Candidate list during build. Higher → better index quality, slower build |
| `M0` | `2 * M` | Connections at layer 0. More connections at the densest layer improves recall |
| `mL` | `1 / ln(M)` | Level generation multiplier. Controls average number of layers |
| `ef` (search) | `max(k, ef_construction)` | Candidate list during search. Key recall/speed trade-off knob |

### Recall vs. Speed Trade-Off

`ef` at search time is the primary knob:

```
ef = k     → fastest, lowest recall  (~70-85%)
ef = 50    → balanced               (~90-95%)
ef = 200   → near-exact             (~98-99%)
ef = ∞     → exact within graph     (bounded by index quality)
```

Recall is also bounded by index quality (set by `M` and `ef_construction`).

---

## Complexity

| Operation | Time Complexity | Notes |
|---|---|---|
| Insert one vector | O(M · d · log n) expected | Dominated by layer-0 beam search |
| Build n vectors | O(n · M · d · log n) | |
| Search (single query) | O(M · d · log n) expected | For fixed ef |
| Memory | O(n · M · log n) | |

where *n* = index size, *d* = vector dimensions, *M* = max connections.

Compared to brute-force O(n · d) search, HNSW achieves **O(log n)** query time — effectively constant cost as n grows.

---

## Recall@K Definition

```
Recall@K = |ANN top-K  ∩  True top-K| / K
```

A Recall@10 of 0.95 means that on average, 9.5 out of the 10 true nearest neighbours are returned by the approximate search.

---

## Practical Tips

- **Normalise vectors** before using cosine distance (then cosine = Euclidean on the unit sphere).
- **Increase `ef_construction`** if index quality (recall ceiling) is poor.
- **Increase `ef` at search time** to improve recall without rebuilding the index.
- **Increase `M`** if recall is capped even at high `ef` (the graph connections are the bottleneck).
- For very high-dimensional data (d > 512), consider dimensionality reduction (PCA) first.