"""
baseline.py — Brute-force exact TSP solver for ground-truth verification.

Phase 1 Foundation — A* Search for TSP
Repository: https://github.com/Francis-Ajibade/A-star-tsp-heuristic

Provides:
    brute_force_tsp()   — enumerate all permutations and return the optimal tour
    tour_cost()         — sum edge costs along a given path
    all_optimal_tours() — return every permutation that achieves the optimal cost
    solve_and_time()    — timed wrapper that returns a result dict for experiments

The brute-force solver exists solely to provide a known-correct answer against
which A* results can be validated. It becomes impractical beyond ~11 cities
(10! = 3,628,800 permutations), which is precisely why A* with a good heuristic
is necessary for larger instances.
"""

from __future__ import annotations
import sys
import os
import math
import time
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tsp import City, build_distance_matrix


# ── Exact solver ──────────────────────────────────────────────────────────────

def brute_force_tsp(
    dist: list[list[float]],
    start: int = 0,
) -> tuple[list[int], float]:
    """
    Solve TSP exactly by evaluating every permutation of cities.

    The start city is fixed and all other cities are permuted. Fixing the
    start avoids counting rotations of the same tour multiple times
    (A→B→C→A is the same tour as B→C→A→B).

    Time complexity:  O(n!)   — feasible only up to ~11 cities
    Space complexity: O(n)    — only the current best path is retained

    Args:
        dist:  n×n distance matrix from build_distance_matrix().
        start: index of the fixed start city, defaults to 0.

    Returns:
        Tuple (best_path, best_cost) where best_path is a list of city
        indices starting and ending at start.
    """
    n = len(dist)

    # Degenerate cases
    if n == 1:
        return [start, start], 0.0
    if n == 2:
        other = 1 - start
        cost  = dist[start][other] + dist[other][start]
        return [start, other, start], cost

    others     = [i for i in range(n) if i != start]
    best_cost  = float("inf")
    best_path  = None

    for perm in permutations(others):
        path = [start] + list(perm) + [start]
        cost = tour_cost(path, dist)
        if cost < best_cost:
            best_cost = cost
            best_path = path

    return best_path, best_cost


# ── Tour cost utility ─────────────────────────────────────────────────────────

def tour_cost(path: list[int], dist: list[list[float]]) -> float:
    """
    Compute the total travel cost of a complete tour.

    Args:
        path: ordered city indices; must start and end at the same city.
        dist: n×n distance matrix.

    Returns:
        Sum of edge costs along the tour.
    """
    return sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))


# ── All optimal tours ─────────────────────────────────────────────────────────

def all_optimal_tours(
    dist: list[list[float]],
    start: int = 0,
) -> tuple[list[list[int]], float]:
    """
    Return every tour that achieves the optimal cost, not only the first found.

    Multiple tours can share the same minimum cost when a problem instance
    has symmetric structure. This is useful when validating A* results: the
    A* path need not match the brute-force path exactly — it only needs to
    match the optimal cost.

    Args:
        dist:  n×n distance matrix.
        start: fixed start city index, defaults to 0.

    Returns:
        Tuple (list_of_optimal_paths, optimal_cost).
    """
    n          = len(dist)
    others     = [i for i in range(n) if i != start]
    best_cost  = float("inf")
    best_paths: list[list[int]] = []

    for perm in permutations(others):
        path = [start] + list(perm) + [start]
        cost = tour_cost(path, dist)
        if cost < best_cost - 1e-9:
            best_cost  = cost
            best_paths = [path]
        elif abs(cost - best_cost) < 1e-9:
            best_paths.append(path)

    return best_paths, best_cost


# ── Timed experiment wrapper ──────────────────────────────────────────────────

def solve_and_time(
    dist: list[list[float]],
    start: int = 0,
) -> dict:
    """
    Run brute_force_tsp and record wall-clock runtime and permutation count.

    Returns a standardised result dict compatible with the Phase 4 experiment
    logging format. The nodes_explored field counts (n-1)! permutations,
    which matches the number of complete candidate tours evaluated.

    Args:
        dist:  n×n distance matrix.
        start: fixed start city, defaults to 0.

    Returns:
        {
            "path"          : list[int],   — optimal tour with return edge
            "cost"          : float,        — optimal tour cost
            "runtime_ms"    : float,        — wall-clock time in milliseconds
            "nodes_explored": int,          — (n-1)! permutations evaluated
            "n_cities"      : int,          — number of cities
        }
    """
    n      = len(dist)
    n_perm = math.factorial(n - 1) if n > 1 else 1

    t0         = time.perf_counter()
    path, cost = brute_force_tsp(dist, start=start)
    runtime_ms = (time.perf_counter() - t0) * 1000

    return {
        "path"          : path,
        "cost"          : cost,
        "runtime_ms"    : runtime_ms,
        "nodes_explored": n_perm,
        "n_cities"      : n,
    }


# ── Display helper ────────────────────────────────────────────────────────────

def print_result(
    result: dict,
    cities: list[City] | None = None,
) -> None:
    """
    Print a formatted summary of a solve_and_time() result.

    Args:
        result: dict returned by solve_and_time().
        cities: optional list of City objects; when provided, city names
                are printed instead of numeric indices.
    """
    path = result["path"]
    if cities:
        path_str = " → ".join(cities[i].name for i in path)
    else:
        path_str = " → ".join(str(i) for i in path)

    print(f"  Optimal tour : {path_str}")
    print(f"  Total cost   : {result['cost']:.4f}")
    print(f"  Runtime      : {result['runtime_ms']:.3f} ms")
    print(f"  Permutations : {result['nodes_explored']:,}")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tsp import generate_random_cities

    # ── Test 1: unit square — optimal perimeter = 4.0 ─────────────────────────
    print("=" * 55)
    print("Test 1 — 4 cities (unit square, optimal = 4.0)")
    print("=" * 55)

    cities_4 = [
        City("A", 0, 0), City("B", 1, 0),
        City("C", 1, 1), City("D", 0, 1),
    ]
    dist_4  = build_distance_matrix(cities_4)
    result  = solve_and_time(dist_4)
    print_result(result, cities_4)
    assert abs(result["cost"] - 4.0) < 1e-6, f"Expected 4.0, got {result['cost']:.6f}"
    print("  Assertion passed: cost == 4.0\n")

    # ── Test 2: 5-city standard instance ──────────────────────────────────────
    print("=" * 55)
    print("Test 2 — 5 cities")
    print("=" * 55)

    cities_5 = [
        City("A", 0, 0), City("B", 2, 4), City("C", 5, 2),
        City("D", 6, 6), City("E", 1, 7),
    ]
    dist_5  = build_distance_matrix(cities_5)
    result5 = solve_and_time(dist_5)
    print_result(result5, cities_5)

    opt_tours, opt_cost = all_optimal_tours(dist_5)
    print(f"  Optimal tours found : {len(opt_tours)}")
    print()

    # ── Test 3: runtime scaling — illustrates O(n!) growth ────────────────────
    print("=" * 55)
    print("Test 3 — Runtime scaling (n = 5 to 10)")
    print("=" * 55)

    for n in range(5, 11):
        cities_n = generate_random_cities(n, seed=42)
        dist_n   = build_distance_matrix(cities_n)
        r        = solve_and_time(dist_n)
        print(
            f"  n={n:2d} | "
            f"cost={r['cost']:8.3f} | "
            f"perms={r['nodes_explored']:>8,} | "
            f"time={r['runtime_ms']:7.2f} ms"
        )

    print()
    print("  Brute force becomes impractical beyond ~11 cities.")
    print("  This is the motivation for A* with admissible heuristics.")