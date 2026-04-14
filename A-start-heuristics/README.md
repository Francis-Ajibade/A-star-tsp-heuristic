# A* Search for the Traveling Salesman Problem
### Designing, Implementing, and Evaluating Heuristics

> **Course:** CS — Artificial Intelligence  
> **Topic:** Informed Search and Heuristic Design  
> **Result:** MST heuristic reduces nodes expanded by up to 92% while guaranteeing the optimal tour

---

## Overview

This project investigates how heuristic quality affects A* search efficiency on the Traveling Salesman Problem (TSP). Three heuristics of increasing informativeness were designed, implemented, and evaluated:

| Heuristic | h(n) | Admissible | Consistent | Nodes (n=8) |
|-----------|-------|:---:|:---:|---:|
| Null (UCS) | 0 | ✓ | ✓ | 450 |
| Min-edge | remaining × global min edge | ✓ | ✗ | 444 |
| **MST** | MST cost + entry + return edges | ✓ | ✓ | **17** |

Both admissibility and consistency were verified computationally across all 33 reachable states and 52 state-to-successor edges in a 5-city instance — zero violations found. Testing on the TSPLIB95 benchmark `wi29` (29 cities, Western Sahara) confirmed MST solves the instance in 268ms with 2,166 nodes, while the Null heuristic completed in 2,206ms but returned a **suboptimal** tour — demonstrating that admissibility is necessary not just for optimality, but to prevent A* from confidently returning wrong answers.

---

## Repository Structure

```
A-start-heuristics/
│
├── code/
│   ├── phase1_foundation/
│   │   ├── tsp.py              # City class, distance matrix, TSPLIB95 loader
│   │   ├── state.py            # TSPState — frozenset visited set, expand(), hash
│   │   ├── baseline.py         # Brute-force exact solver via itertools.permutations
│   │   └── verify.py           # 40+ assertion test suite for Phase 1
│   │
│   ├── phase2_astar/
│   │   ├── astar.py            # Core A* loop with lazy-deletion heap
│   │   ├── frontier.py         # Frontier + VisitedSet classes with memory stats
│   │   └── test_astar.py       # 59-assertion test suite for Phase 2
│   │
│   └── phase3_heuristics/
│       ├── heuristic_mst.py    # Prim's MST heuristic (primary contribution)
│       ├── heuristic_nn.py     # Nearest-neighbour (inadmissible — comparison only)
│       └── heuristic_combo.py  # max(MST, min-edge) combination
│
├── experiments/
│   └── phase4_experiments/
│       ├── admissibility.ipynb # Admissibility and consistency proofs + charts
│       └── dataset_viz.ipynb   # wi29 dataset visualisation and comparison
│
├── data/
│   └── wi29.tsp                # TSPLIB95 Western Sahara 29-city benchmark
│
└── results/
    └── charts/                 # All generated PNG charts (see Figures below)
```

---

## Key Results

### Random Instances — Nodes Expanded

| Cities (n) | Null (UCS) | Min-edge | MST | MST saving |
|:---:|---:|---:|---:|---:|
| 5 | 34 | 25 | 9 | 73.5% |
| 6 | 82 | 53 | 9 | 89.0% |
| 7 | 194 | 181 | 14 | 92.8% |
| 8 | 450 | 444 | 17 | 96.2% |

All heuristics returned the exact same optimal tour cost on every instance.

### Real Dataset — wi29 Western Sahara (TSPLIB95)

| Heuristic | Nodes | Time | Tour cost | Optimal? |
|-----------|------:|-----:|----------:|:---:|
| Null (UCS) | 50,538 | 2,206ms | 35.6519 | ✗ suboptimal |
| Min-edge | 53,227 | 4,592ms | 30.8785 | ✓ |
| **MST** | **2,166** | **268ms** | **30.8785** | **✓** |

The Null heuristic returned a wrong answer — not a timeout. It completed its search but settled on a suboptimal path 15.2% longer than the optimal, demonstrating that admissibility is not just a theoretical nicety.

---

## Implementation Notes

### Why lazy deletion?

The A* heap uses lazy deletion rather than decrease-key. When a better path to a state is found, a new entry is pushed and the old one is left in the heap. Stale entries are detected and discarded on pop by comparing `state.g_cost` against the `best_g` dictionary. This keeps `push()` at O(log n) and avoids the O(n) cost of heap search-and-remove.

### Why frozenset for the visited set?

`TSPState` stores visited cities as a `frozenset[int]`. This makes states hashable so they can be used as dictionary keys in the `best_g` table — which is how A* detects that two different paths have reached the same search position.

### The goal detection fix

A naive TSP A* implementation fires the goal test on the first all-visited state popped from the heap. This is incorrect because total tour cost is `g_cost + return_edge`, not `g_cost` alone. A path with lower `g_cost` may have a longer return edge and produce a worse total. The correct implementation treats the return edge as one more expansion: when all cities are visited, a terminal state with `g_cost = g + return_edge` is pushed onto the heap. The first terminal state popped is the optimal complete tour.

### The Prim's MST seed-node bug

An earlier version of `prim_mst_cost()` initialised `min_edge[v] = inf` for all nodes and then set the seed node's distance to 0 — but never relaxed the seed's outgoing edges before the main loop. This caused the first iteration to pick an arbitrary node with `inf` cost rather than the cheapest-connected one, producing incorrect MST costs and making the heuristic return 0.0 for all states. The fix: initialise `min_edge[v] = dist[seed][v]` for all `v != seed` before the loop begins.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Francis-Ajibade/A-star-tsp-heuristic.git
cd A-star-tsp-heuristic

# Install dependencies
pip install matplotlib numpy

# Run the Phase 1 foundation tests
python code/phase1_foundation/verify.py

# Run the Phase 2 A* test suite
python code/phase2_astar/test_astar.py

# Run the MST heuristic smoke test
python code/phase3_heuristics/heuristic_mst.py

# Run the full heuristic comparison
python code/phase3_heuristics/heuristic_combo.py
```

To reproduce the admissibility charts:

```bash
cd experiments/phase4_experiments
jupyter notebook admissibility.ipynb
```

> **Note:** Run the notebook from the `phase4_experiments/` directory. The path setup cell uses `os.path.abspath('..')` to locate the source modules, which assumes that working directory.

---

## Heuristic Design

### MST Heuristic — `heuristic_mst.py`

```
h(n) = prim_mst_cost(unvisited cities)
     + min edge from current city into unvisited set
     + min edge from any unvisited city back to start
```

**Admissibility proof (Held & Karp, 1962):**  
Any Hamiltonian path through k unvisited cities must use at least as many edges as the MST connecting them. The entry and return edges are mandatory distinct edges in the optimal remaining path. Since all three components are lower bounds on distinct parts of the optimal remaining cost, their sum never exceeds the true remaining cost. ✓

**Consistency proof:**  
By the cut property of spanning trees: removing a city from the unvisited set reduces the MST cost by at most the weight of the cheapest connecting edge — which is exactly the edge just traversed. Therefore `h(n) <= edge_cost + h(n')` for every successor. ✓

### Nearest-Neighbour — `heuristic_nn.py`

Greedily picks the closest unvisited city at each step. **Not admissible** — included as a comparison case to show that inadmissibility causes A* to expand fewer nodes but return suboptimal tours. This is the "striking failure" result: NN looks efficient by node count but the answers are wrong.

### Combo — `heuristic_combo.py`

`max(MST, min-edge)`. Always admissible since max of two admissible heuristics is admissible. Rarely improves on MST alone in practice because MST already dominates min-edge almost everywhere.

---

## Theoretical Background

**Why TSP is hard:**  
TSP is NP-hard. The number of possible routes grows factorially — 10 cities gives over 3.6 million permutations, 20 cities gives over 2 trillion. No exact algorithm can evaluate every possibility beyond a modest city count.

**Why A* helps:**  
A* uses a heuristic estimate `h(n)` to prioritise promising partial routes. With a tight admissible heuristic, most of the search space can be safely pruned without risking an incorrect answer.

**The hard limit:**  
Even with MST reducing nodes by over 96%, A* hits the exponential wall at approximately 12–15 cities. TSP is NP-hard and no heuristic choice changes the asymptotic complexity — only the constant factor. This is documented in the exponential failure chart in `results/charts/`.

---

## Related Work

| Approach | Type | Optimal? | Scales to |
|----------|------|:--------:|----------:|
| Held-Karp DP [1] | Exact | ✓ | ~20 cities |
| **A* + MST (this project)** | **Exact** | **✓** | **~12–15 cities** |
| Christofides [2] | 1.5× approx | ✗ | thousands |
| LKH-3 [3] | Near-optimal | ✗ | 100,000+ cities |
| Attention Model [4] | Learned | ✗ | 100+ cities |

This project prioritises exactness and interpretability over scale. The full proof chain — from implementation through computational verification to experimental results — is the contribution, not the absolute scale.

---

## References

[1] M. Held and R. M. Karp, "A Dynamic Programming Approach to Sequencing Problems," *J. Soc. Ind. Appl. Math.*, vol. 10, no. 1, pp. 196–210, 1962. doi: 10.1137/0110015

[2] N. Christofides, "Worst-Case Analysis of a New Heuristic for the Travelling Salesman Problem," *Oper. Res. Forum*, vol. 3, no. 1, p. 20, 2022. doi: 10.1007/s43069-021-00101-z

[3] K. Helsgaun, "An effective implementation of the Lin–Kernighan traveling salesman heuristic," *Eur. J. Oper. Res.*, vol. 126, no. 1, pp. 106–130, 2000. doi: 10.1016/S0377-2217(99)00284-2

[4] W. Kool, H. van Hoof, and M. Welling, "Attention, Learn to Solve Routing Problems!" *ICLR 2019*. arXiv: 1803.08475

[5] G. Reinelt, "TSPLIB — A Traveling Salesman Problem Library," *ORSA J. Comput.*, vol. 3, no. 4, pp. 376–384, 1991. doi: 10.1287/ijoc.3.4.376

[6] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Pearson, 2020.

---

## Dataset

The `wi29` benchmark is sourced from TSPLIB95 — the standard academic benchmark library for TSP research:

- **Source:** http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- **Direct download:** http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/wi29.tsp.gz
- **Description:** 29 cities at real geographic coordinates in Western Sahara (Gr, 1977)
- **Coordinate type:** EUC_2D (normalised to [0, 100] for consistency with random instances)

The `tsp.py` module includes `download_wi29()` which fetches and decompresses the file automatically, with an embedded fallback if the TSPLIB server is unreachable.

---

*CS — Artificial Intelligence · 2025*
