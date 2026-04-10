from __future__ import annotations
import sys
import os
import time
from itertools import permutations

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))
from tsp import City, build_distance_matrix


# ── Core solver ───────────────────────────────────────────────────────────────

def brute_force_tsp(
    dist: list[list[float]],
    start: int = 0,
) -> tuple[list[int], float]:
    """
    Solve TSP exactly by evaluating every possible permutation of cities.

    Fixes the start city and permutes all others — this avoids counting
    rotations of the same tour multiple times (A→B→C→A == B→C→A→B).

    Time complexity : O(n!)  — only practical up to ~11 cities
    Space complexity: O(n)   — only the best path is stored

    Args:
        dist  : n×n distance matrix from build_distance_matrix()
        start : index of the starting city (default 0)

    Returns:
        (best_path, best_cost) where best_path includes the return to start
    """
    n = len(dist)

    if n == 1:
        return [start, start], 0.0

    if n == 2:
        other = 1 - start
        cost = dist[start][other] + dist[other][start]
        return [start, other, start], cost

    others = [i for i in range(n) if i != start]
    best_cost = float('inf')
    best_path = None

    for perm in permutations(others):
        path = [start] + list(perm) + [start]
        cost = tour_cost(path, dist)
        if cost < best_cost:
            best_cost = cost
            best_path = path

    return best_path, best_cost


# ── Tour cost helper ──────────────────────────────────────────────────────────

def tour_cost(path: list[int], dist: list[list[float]]) -> float:
    """
    Compute the total cost of a complete tour.

    Args:
        path: ordered list of city indices, must start and end at same city
        dist: n×n distance matrix

    Returns:
        sum of edge costs along the tour
    """
    return sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))


# ── All optimal tours ─────────────────────────────────────────────────────────

def all_optimal_tours(
    dist: list[list[float]],
    start: int = 0,
) -> tuple[list[list[int]], float]:
    """
    Return ALL tours that achieve the optimal cost, not just the first one.
    Useful when verifying that A* finds a valid optimum (not necessarily
    the same path, but the same cost).

    Returns:
        (list of optimal paths, optimal cost)
    """
    n = len(dist)
    others = [i for i in range(n) if i != start]
    best_cost = float('inf')
    best_paths = []

    for perm in permutations(others):
        path = [start] + list(perm) + [start]
        cost = tour_cost(path, dist)
        if cost < best_cost - 1e-9:
            best_cost = cost
            best_paths = [path]
        elif abs(cost - best_cost) < 1e-9:
            best_paths.append(path)

    return best_paths, best_cost


# ── Benchmarking wrapper ──────────────────────────────────────────────────────

def solve_and_time(
    dist: list[list[float]],
    start: int = 0,
) -> dict:
    """
    Run brute_force_tsp and record wall-clock runtime and nodes evaluated.
    Returns a results dict ready to hand to the experiment logger in Phase 4.

    Returns:
        {
            "path"          : list[int],
            "cost"          : float,
            "runtime_ms"    : float,
            "nodes_explored": int,      # = (n-1)! permutations evaluated
            "n_cities"      : int,
        }
    """
    import math
    n = len(dist)
    n_perms = math.factorial(n - 1) if n > 1 else 1

    t0 = time.perf_counter()
    path, cost = brute_force_tsp(dist, start=start)
    runtime_ms = (time.perf_counter() - t0) * 1000

    return {
        "path"          : path,
        "cost"          : cost,
        "runtime_ms"    : runtime_ms,
        "nodes_explored": n_perms,
        "n_cities"      : n,
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(
    result: dict,
    cities: list[City] | None = None,
) -> None:
    """
    Print a formatted solution summary.
    If cities is provided, shows city names instead of indices.
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

    # ── Test 1: tiny 4-city hand-verifiable instance ──────────────────────────
    print("=" * 55)
    print("Test 1 — 4 cities (hand-verifiable)")
    print("=" * 55)

    cities_4 = [
        City("A", 0, 0),
        City("B", 1, 0),
        City("C", 1, 1),
        City("D", 0, 1),
    ]
    dist_4 = build_distance_matrix(cities_4)
    result = solve_and_time(dist_4)
    print_result(result, cities_4)

    # The square has perimeter 4.0 — optimal tour must equal 4.0
    assert abs(result["cost"] - 4.0) < 1e-6, "Expected cost 4.0 for unit square"
    print("  Assertion passed: cost == 4.0\n")

    # ── Test 2: 5-city standard instance ─────────────────────────────────────
    print("=" * 55)
    print("Test 2 — 5 cities (standard test instance)")
    print("=" * 55)

    cities_5 = [
        City("A", 0, 0),
        City("B", 2, 4),
        City("C", 5, 2),
        City("D", 6, 6),
        City("E", 1, 7),
    ]
    dist_5 = build_distance_matrix(cities_5)
    result5 = solve_and_time(dist_5)
    print_result(result5, cities_5)

    optimal_tours, opt_cost = all_optimal_tours(dist_5)
    print(f"  Optimal tours found: {len(optimal_tours)}")
    print()

    # ── Test 3: scaling behaviour ─────────────────────────────────────────────
    print("=" * 55)
    print("Test 3 — runtime scaling (5 → 10 cities)")
    print("=" * 55)

    from tsp import generate_random_cities

    for n in [5, 6, 7, 8, 9, 10]:
        cities_n = generate_random_cities(n, seed=42)
        dist_n = build_distance_matrix(cities_n)
        r = solve_and_time(dist_n)
        print(
            f"  n={n:2d} | "
            f"cost={r['cost']:8.3f} | "
            f"perms={r['nodes_explored']:>8,} | "
            f"time={r['runtime_ms']:7.2f} ms"
        )

    print()
    print("Brute-force becomes impractical beyond ~11 cities.")
    print("This is exactly why we need A* + good heuristics.")