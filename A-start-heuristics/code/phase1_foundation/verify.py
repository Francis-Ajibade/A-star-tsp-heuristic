"""
verify.py — Automated verification suite for Phase 1 foundation modules.

Phase 1 Foundation — A* Search for TSP
Repository: https://github.com/Francis-Ajibade/A-star-tsp-heuristic

Runs five test suites covering:
    1. Distance matrix properties (symmetry, zero diagonal, known values)
    2. TSPState representation (expansion, hashing, goal detection)
    3. Brute-force baseline solver correctness
    4. Edge cases (1-city and 2-city degenerate instances)
    5. Full pipeline integration (state space size, tour validity)

Exit code 0 if all tests pass, 1 if any fail.

Usage:
    python verify.py
"""

from __future__ import annotations
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))

from tsp      import City, build_distance_matrix, euclidean, manhattan, generate_random_cities
from state    import TSPState, make_start_state
from baseline import brute_force_tsp, tour_cost, all_optimal_tours, solve_and_time


# ── Output helpers ────────────────────────────────────────────────────────────

PASS = "  PASS"
FAIL = "  FAIL"
SEP  = "=" * 55


def check(label: str, condition: bool, detail: str = "") -> bool:
    """
    Print a PASS/FAIL line for one assertion.

    Args:
        label:     short description of the check.
        condition: True if the check passes.
        detail:    optional extra context printed in parentheses on failure.

    Returns:
        The value of condition, so callers can accumulate failure counts.
    """
    status = PASS if condition else FAIL
    line   = f"{status} | {label}"
    if detail and not condition:
        line += f"  ({detail})"
    print(line)
    return condition


def section(title: str) -> None:
    """Print a section separator with a title."""
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── Test suite 1: Distance matrix ─────────────────────────────────────────────

def test_distance_matrix() -> int:
    """
    Verify distance matrix properties on a hand-crafted 3-city instance.

    Checks:
        - Zero diagonal: dist[i][i] == 0 for all i
        - Symmetry: dist[i][j] == dist[j][i] for all i, j
        - Known values: 3-4-5 right triangle gives exact integer distances
        - Triangle inequality holds
        - Manhattan metric produces correct L1 distances
    """
    section("1. Distance matrix")
    failures = 0

    # 3-4-5 right triangle — all distances are exact integers or 5.0
    cities = [City("A", 0, 0), City("B", 3, 0), City("C", 3, 4)]
    dist   = build_distance_matrix(cities)
    n      = len(cities)

    for i in range(n):
        ok = check(f"dist[{i}][{i}] == 0", dist[i][i] == 0.0)
        failures += 0 if ok else 1

    for i in range(n):
        for j in range(n):
            ok = check(
                f"symmetric: dist[{i}][{j}] == dist[{j}][{i}]",
                abs(dist[i][j] - dist[j][i]) < 1e-9,
            )
            failures += 0 if ok else 1

    ok = check("dist A→B == 3.0", abs(dist[0][1] - 3.0) < 1e-9, f"got {dist[0][1]:.4f}")
    failures += 0 if ok else 1
    ok = check("dist B→C == 4.0", abs(dist[1][2] - 4.0) < 1e-9, f"got {dist[1][2]:.4f}")
    failures += 0 if ok else 1
    ok = check("dist A→C == 5.0", abs(dist[0][2] - 5.0) < 1e-9, f"got {dist[0][2]:.4f}")
    failures += 0 if ok else 1

    ok = check(
        "triangle inequality: dist[A][C] <= dist[A][B] + dist[B][C]",
        dist[0][2] <= dist[0][1] + dist[1][2] + 1e-9,
    )
    failures += 0 if ok else 1

    dist_m = build_distance_matrix(cities, metric=manhattan)
    ok = check(
        "manhattan dist A→C == 7.0",
        abs(dist_m[0][2] - 7.0) < 1e-9,
        f"got {dist_m[0][2]:.4f}",
    )
    failures += 0 if ok else 1

    return failures


# ── Test suite 2: State representation ────────────────────────────────────────

def test_state_representation() -> int:
    """
    Verify TSPState construction, expansion, goal detection, and hashing.

    Checks:
        - Start state has correct initial values
        - Expansion produces the correct number and type of successors
        - g_cost updates correctly through edge traversal
        - Visited set grows correctly on expansion
        - No city is revisited once marked visited
        - Goal detection triggers at the right count
        - final_cost and full_tour produce correct output
        - Equal states (same city + visited, different g/path) share hash
        - States are usable as dictionary keys (required for A* best_g table)
    """
    section("2. State representation")
    failures = 0

    dist = [
        [0.0, 1.0, 4.0, 3.0],
        [1.0, 0.0, 2.0, 5.0],
        [4.0, 2.0, 0.0, 1.0],
        [3.0, 5.0, 1.0, 0.0],
    ]

    start = make_start_state(start_city=0)

    ok = check("start.current_city == 0",       start.current_city == 0)
    failures += 0 if ok else 1
    ok = check("start.visited == {0}",          start.visited == frozenset({0}))
    failures += 0 if ok else 1
    ok = check("start.g_cost == 0.0",           start.g_cost == 0.0)
    failures += 0 if ok else 1
    ok = check("start.path == [0]",             start.path == [0])
    failures += 0 if ok else 1
    ok = check("start is not goal (n=4)",       not start.is_goal(n=4))
    failures += 0 if ok else 1

    successors = start.expand(dist)
    ok = check("3 successors from city 0",      len(successors) == 3, f"got {len(successors)}")
    failures += 0 if ok else 1

    cities_reached = {s.current_city for s in successors}
    ok = check("successors reach cities 1, 2, 3", cities_reached == {1, 2, 3})
    failures += 0 if ok else 1

    s1 = next(s for s in successors if s.current_city == 1)
    ok = check("g_cost 0→1 == 1.0",            abs(s1.g_cost - 1.0) < 1e-9, f"got {s1.g_cost}")
    failures += 0 if ok else 1
    ok = check("visited after 0→1 == {0, 1}",  s1.visited == frozenset({0, 1}))
    failures += 0 if ok else 1
    ok = check("path after 0→1 == [0, 1]",     s1.path == [0, 1])
    failures += 0 if ok else 1

    s1_successors = s1.expand(dist)
    ok = check("no revisit of cities 0 or 1",  all(s.current_city not in {0, 1} for s in s1_successors))
    failures += 0 if ok else 1

    goal = TSPState(
        current_city=3,
        visited=frozenset({0, 1, 2, 3}),
        g_cost=4.0,
        path=[0, 1, 2, 3],
    )
    ok = check("goal detected when all 4 cities visited", goal.is_goal(n=4))
    failures += 0 if ok else 1

    expected_final = 4.0 + dist[3][0]
    ok = check(
        "final_cost == g_cost + return edge",
        abs(goal.final_cost(dist) - expected_final) < 1e-9,
        f"got {goal.final_cost(dist):.4f}, expected {expected_final:.4f}",
    )
    failures += 0 if ok else 1

    tour = goal.full_tour()
    ok = check("full_tour starts and ends at city 0", tour[0] == 0 and tour[-1] == 0)
    failures += 0 if ok else 1

    duplicate = TSPState(
        current_city=3,
        visited=frozenset({0, 1, 2, 3}),
        g_cost=99.0,            # different cost, same identity
        path=[0, 3, 2, 1, 3],  # different path, same identity
    )
    ok = check("equal states with different g_cost", goal == duplicate)
    failures += 0 if ok else 1
    ok = check("same hash for equal states",          hash(goal) == hash(duplicate))
    failures += 0 if ok else 1

    different = TSPState(
        current_city=2,
        visited=frozenset({0, 1, 2, 3}),
        g_cost=4.0,
        path=[0, 1, 3, 2],
    )
    ok = check("different current_city → not equal", goal != different)
    failures += 0 if ok else 1

    best_g: dict[TSPState, float] = {goal: goal.g_cost}
    ok = check("state usable as dict key (A* best_g table)", goal in best_g)
    failures += 0 if ok else 1

    return failures


# ── Test suite 3: Baseline solver ─────────────────────────────────────────────

def test_baseline_solver() -> int:
    """
    Verify brute_force_tsp returns correct optimal costs on known instances.

    Checks:
        - Unit square: perimeter = 4.0
        - Collinear 3 cities: forced back-and-forth = 4.0
        - 5-city instance: tour validity and internal consistency
        - all_optimal_tours agrees with brute_force_tsp on cost
        - solve_and_time returns all required keys
        - nodes_explored == (n-1)! permutations
    """
    section("3. Brute-force baseline solver")
    failures = 0

    # Unit square — optimal perimeter = 4.0
    cities_sq = [
        City("A", 0, 0), City("B", 1, 0),
        City("C", 1, 1), City("D", 0, 1),
    ]
    dist_sq    = build_distance_matrix(cities_sq)
    path, cost = brute_force_tsp(dist_sq)

    ok = check("unit square optimal cost == 4.0",
               abs(cost - 4.0) < 1e-6, f"got {cost:.6f}")
    failures += 0 if ok else 1
    ok = check("tour starts and ends at city 0",
               path[0] == 0 and path[-1] == 0)
    failures += 0 if ok else 1
    ok = check("tour visits all 4 cities",
               len(set(path)) == 4)
    failures += 0 if ok else 1
    ok = check("tour_cost is consistent with returned cost",
               abs(tour_cost(path, dist_sq) - cost) < 1e-9)
    failures += 0 if ok else 1

    # Collinear 3 cities — only route is A→B→C→A or reverse, cost = 2+2 = 4.0
    cities_line  = [City("A", 0, 0), City("B", 1, 0), City("C", 2, 0)]
    dist_line    = build_distance_matrix(cities_line)
    _, cost_line = brute_force_tsp(dist_line)
    ok = check("collinear 3-city optimal cost == 4.0",
               abs(cost_line - 4.0) < 1e-6, f"got {cost_line:.6f}")
    failures += 0 if ok else 1

    # 5-city standard instance
    cities_5 = [
        City("A", 0, 0), City("B", 2, 4), City("C", 5, 2),
        City("D", 6, 6), City("E", 1, 7),
    ]
    dist_5       = build_distance_matrix(cities_5)
    path5, cost5 = brute_force_tsp(dist_5)

    ok = check("5-city tour visits all cities",
               len(set(path5)) == 5)
    failures += 0 if ok else 1
    ok = check("5-city tour cost > 0", cost5 > 0)
    failures += 0 if ok else 1
    ok = check("5-city tour_cost consistent with returned cost",
               abs(tour_cost(path5, dist_5) - cost5) < 1e-9)
    failures += 0 if ok else 1

    opt_tours, opt_cost = all_optimal_tours(dist_5)
    ok = check("all_optimal_tours agrees on cost",
               abs(opt_cost - cost5) < 1e-9,
               f"brute={cost5:.4f}  all_opt={opt_cost:.4f}")
    failures += 0 if ok else 1
    ok = check("at least one optimal tour returned",
               len(opt_tours) >= 1)
    failures += 0 if ok else 1

    result = solve_and_time(dist_5)
    for key in ("path", "cost", "runtime_ms", "nodes_explored", "n_cities"):
        ok = check(f"solve_and_time contains key '{key}'", key in result)
        failures += 0 if ok else 1

    ok = check(
        "nodes_explored == (n-1)! == 24",
        result["nodes_explored"] == math.factorial(4),
        f"got {result['nodes_explored']}",
    )
    failures += 0 if ok else 1

    return failures


# ── Test suite 4: Edge cases ──────────────────────────────────────────────────

def test_edge_cases() -> int:
    """
    Verify correct handling of degenerate 1-city and 2-city instances,
    and confirm reproducibility of the random city generator.

    Checks:
        - 1-city instance: cost = 0.0, state is immediately at goal
        - 2-city instance: only tour is A→B→A, cost = 2 × dist(A, B)
        - Same seed always produces identical city positions
        - Different seeds produce at least one differing city position
    """
    section("4. Edge cases")
    failures = 0

    # 1 city
    dist_1    = build_distance_matrix([City("A", 0, 0)])
    _, cost_1 = brute_force_tsp(dist_1)
    ok = check("1-city cost == 0.0", abs(cost_1 - 0.0) < 1e-9, f"got {cost_1}")
    failures += 0 if ok else 1

    s1 = make_start_state(0)
    ok = check("1-city state is immediately at goal", s1.is_goal(n=1))
    failures += 0 if ok else 1

    # 2 cities: A(0,0) B(3,4) — dist = 5.0, round trip = 10.0
    dist_2    = build_distance_matrix([City("A", 0, 0), City("B", 3, 4)])
    _, cost_2 = brute_force_tsp(dist_2)
    ok = check("2-city cost == 10.0 (round trip of dist=5.0)",
               abs(cost_2 - 10.0) < 1e-6, f"got {cost_2:.4f}")
    failures += 0 if ok else 1

    # Reproducibility
    run_a = generate_random_cities(5, seed=99)
    run_b = generate_random_cities(5, seed=99)
    ok = check("same seed produces identical cities",
               all(a.x == b.x and a.y == b.y for a, b in zip(run_a, run_b)))
    failures += 0 if ok else 1

    run_c = generate_random_cities(5, seed=1)
    ok = check("different seeds produce at least one differing city",
               any(a.x != c.x or a.y != c.y for a, c in zip(run_a, run_c)))
    failures += 0 if ok else 1

    return failures


# ── Test suite 5: Full pipeline integration ───────────────────────────────────

def test_integration() -> int:
    """
    End-to-end pipeline check: build cities → distance matrix → state space
    exploration → brute-force optimal tour.

    Checks:
        - State space for n=5 contains exactly 33 unique (city, visited) pairs.
          City 0 is marked visited in the start state and excluded from all
          successors, so the reachable pairs are:
              size=1: 1  (start only)
              size=2: C(4,1)*1 = 4
              size=3: C(4,2)*2 = 12
              size=4: C(4,3)*3 = 12
              size=5: C(4,4)*4 = 4
              total  = 33
        - Brute-force tour cost recomputed from the path matches the returned value
        - Tour length is n+1 (n cities plus the return to start)
        - Tour visits every city index exactly once (no repeats except start/end)
    """
    section("5. Integration — full pipeline")
    failures = 0

    cities = [
        City("A", 0, 0), City("B", 2, 4), City("C", 5, 2),
        City("D", 6, 6), City("E", 1, 7),
    ]
    dist = build_distance_matrix(cities)

    # BFS to enumerate the full state space without A* — checks state graph size
    start     = make_start_state(0)
    frontier  = [start]
    seen: set[tuple] = set()
    n_states  = 0

    while frontier:
        state = frontier.pop()
        key   = (state.current_city, state.visited)
        if key in seen:
            continue
        seen.add(key)
        n_states += 1
        if not state.is_goal(n=5):
            frontier.extend(state.expand(dist))

    ok = check(
        "state space size == 33 for n=5",
        n_states == 33,
        f"got {n_states}",
    )
    failures += 0 if ok else 1

    # Brute-force optimal tour
    path, cost  = brute_force_tsp(dist)
    recomputed  = tour_cost(path, dist)

    ok = check("recomputed tour cost matches returned cost",
               abs(recomputed - cost) < 1e-9)
    failures += 0 if ok else 1
    ok = check("tour length == n+1 (includes return to start)",
               len(path) == len(cities) + 1)
    failures += 0 if ok else 1
    ok = check("tour visits every city index exactly once",
               sorted(set(path)) == list(range(len(cities))))
    failures += 0 if ok else 1

    print(f"\n  Optimal tour : {' → '.join(cities[i].name for i in path)}")
    print(f"  Total cost   : {cost:.4f}")

    return failures


# ── Test runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nPhase 1 Foundation — Verification Suite")
    print(SEP)

    total_failures  = 0
    total_failures += test_distance_matrix()
    total_failures += test_state_representation()
    total_failures += test_baseline_solver()
    total_failures += test_edge_cases()
    total_failures += test_integration()

    print(f"\n{SEP}")
    if total_failures == 0:
        print("  ALL TESTS PASSED — Phase 1 foundation is verified.")
        print("  Ready to proceed to Phase 2: A* core search.")
    else:
        print(f"  {total_failures} TEST(S) FAILED — resolve before proceeding.")
    print(SEP)

    sys.exit(0 if total_failures == 0 else 1)