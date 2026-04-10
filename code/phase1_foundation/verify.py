from __future__ import annotations
import sys
import os
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))

from tsp import City, build_distance_matrix, euclidean, manhattan, generate_random_cities
from state import TSPState, make_start_state
from baseline import brute_force_tsp, tour_cost, all_optimal_tours, solve_and_time

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "  PASS"
FAIL = "  FAIL"
SEP  = "=" * 55


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    line   = f"{status} | {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return condition


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── Test suite ────────────────────────────────────────────────────────────────

def test_distance_matrix() -> int:
    """Verify distance matrix properties: symmetry, zero diagonal, positivity."""
    section("1. Distance matrix")
    failures = 0

    cities = [
        City("A", 0, 0),
        City("B", 3, 0),
        City("C", 3, 4),
    ]
    dist = build_distance_matrix(cities)
    n = len(cities)

    # Zero diagonal
    for i in range(n):
        ok = check(f"dist[{i}][{i}] == 0", dist[i][i] == 0.0)
        if not ok:
            failures += 1

    # Symmetry
    for i in range(n):
        for j in range(n):
            ok = check(
                f"dist[{i}][{j}] == dist[{j}][{i}]",
                abs(dist[i][j] - dist[j][i]) < 1e-9,
            )
            if not ok:
                failures += 1

    # Known values — 3-4-5 right triangle
    ok = check("dist A→B == 3.0", abs(dist[0][1] - 3.0) < 1e-9, f"got {dist[0][1]:.4f}")
    failures += 0 if ok else 1
    ok = check("dist B→C == 4.0", abs(dist[1][2] - 4.0) < 1e-9, f"got {dist[1][2]:.4f}")
    failures += 0 if ok else 1
    ok = check("dist A→C == 5.0", abs(dist[0][2] - 5.0) < 1e-9, f"got {dist[0][2]:.4f}")
    failures += 0 if ok else 1

    # Triangle inequality
    ok = check(
        "triangle inequality: dist[A][C] <= dist[A][B] + dist[B][C]",
        dist[0][2] <= dist[0][1] + dist[1][2] + 1e-9,
    )
    failures += 0 if ok else 1

    # Manhattan metric
    dist_m = build_distance_matrix(cities, metric=manhattan)
    ok = check(
        "manhattan dist A→C == 7.0",
        abs(dist_m[0][2] - 7.0) < 1e-9,
        f"got {dist_m[0][2]:.4f}",
    )
    failures += 0 if ok else 1

    return failures


def test_state_representation() -> int:
    """Verify TSPState hashing, equality, expansion, and goal detection."""
    section("2. State representation")
    failures = 0

    dist = [
        [0.0, 1.0, 4.0, 3.0],
        [1.0, 0.0, 2.0, 5.0],
        [4.0, 2.0, 0.0, 1.0],
        [3.0, 5.0, 1.0, 0.0],
    ]

    start = make_start_state(start_city=0)

    # Start state properties
    ok = check("start city == 0",       start.current_city == 0)
    failures += 0 if ok else 1
    ok = check("start visited == {0}",  start.visited == frozenset({0}))
    failures += 0 if ok else 1
    ok = check("start g_cost == 0.0",   start.g_cost == 0.0)
    failures += 0 if ok else 1
    ok = check("start path == [0]",     start.path == [0])
    failures += 0 if ok else 1
    ok = check("not goal (4 cities)",   not start.is_goal(n=4))
    failures += 0 if ok else 1

    # Expansion
    successors = start.expand(dist)
    ok = check("3 successors from start", len(successors) == 3, f"got {len(successors)}")
    failures += 0 if ok else 1

    cities_in_successors = {s.current_city for s in successors}
    ok = check("successors cover cities 1,2,3", cities_in_successors == {1, 2, 3})
    failures += 0 if ok else 1

    s_to_1 = next(s for s in successors if s.current_city == 1)
    ok = check("g_cost 0→1 == 1.0",    abs(s_to_1.g_cost - 1.0) < 1e-9, f"got {s_to_1.g_cost}")
    failures += 0 if ok else 1
    ok = check("visited after 0→1 == {0,1}", s_to_1.visited == frozenset({0, 1}))
    failures += 0 if ok else 1
    ok = check("path after 0→1 == [0,1]",    s_to_1.path == [0, 1])
    failures += 0 if ok else 1

    # No self-loops — visited city must not appear in successors
    s2 = s_to_1.expand(dist)
    ok = check("no revisit of city 0 or 1", all(s.current_city not in {0, 1} for s in s2))
    failures += 0 if ok else 1

    # Goal detection
    goal = TSPState(
        current_city=3,
        visited=frozenset({0, 1, 2, 3}),
        g_cost=4.0,
        path=[0, 1, 2, 3],
    )
    ok = check("goal state detected (all 4 visited)", goal.is_goal(n=4))
    failures += 0 if ok else 1

    final = goal.final_cost(dist)
    ok = check("final_cost == g + return edge", abs(final - (4.0 + dist[3][0])) < 1e-9,
               f"got {final:.4f}")
    failures += 0 if ok else 1

    tour = goal.full_tour()
    ok = check("full_tour starts and ends at 0",
               tour[0] == 0 and tour[-1] == 0)
    failures += 0 if ok else 1

    # Hashing and equality
    duplicate = TSPState(
        current_city=3,
        visited=frozenset({0, 1, 2, 3}),
        g_cost=99.0,           # different cost — same identity
        path=[0, 3, 2, 1, 3],
    )
    ok = check("equal states with different g_cost", goal == duplicate)
    failures += 0 if ok else 1
    ok = check("same hash for equal states", hash(goal) == hash(duplicate))
    failures += 0 if ok else 1

    different = TSPState(
        current_city=2,
        visited=frozenset({0, 1, 2, 3}),
        g_cost=4.0,
        path=[0, 1, 3, 2],
    )
    ok = check("different city → not equal", goal != different)
    failures += 0 if ok else 1

    # States usable as dict keys (critical for A* visited set)
    seen: dict[TSPState, float] = {}
    seen[goal] = goal.g_cost
    ok = check("state usable as dict key", goal in seen)
    failures += 0 if ok else 1

    return failures


def test_baseline_solver() -> int:
    """Verify brute-force returns correct optimal costs on known instances."""
    section("3. Brute-force baseline")
    failures = 0

    # ── Unit square: known optimal = 4.0 ─────────────────────────────────────
    cities_sq = [City("A", 0,0), City("B", 1,0), City("C", 1,1), City("D", 0,1)]
    dist_sq   = build_distance_matrix(cities_sq)
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
    ok = check("tour_cost matches returned cost",
               abs(tour_cost(path, dist_sq) - cost) < 1e-9)
    failures += 0 if ok else 1

    # ── Collinear 3 cities: A(0,0) B(1,0) C(2,0) → optimal = 4.0 ────────────
    cities_line = [City("A",0,0), City("B",1,0), City("C",2,0)]
    dist_line   = build_distance_matrix(cities_line)
    _, cost_line = brute_force_tsp(dist_line)

    ok = check("collinear 3-city optimal == 4.0",
               abs(cost_line - 4.0) < 1e-6, f"got {cost_line:.6f}")
    failures += 0 if ok else 1

    # ── 5-city standard instance ──────────────────────────────────────────────
    cities_5 = [
        City("A", 0, 0), City("B", 2, 4), City("C", 5, 2),
        City("D", 6, 6), City("E", 1, 7),
    ]
    dist_5 = build_distance_matrix(cities_5)
    path5, cost5 = brute_force_tsp(dist_5)

    ok = check("5-city tour visits all cities",   len(set(path5)) == 5)
    failures += 0 if ok else 1
    ok = check("5-city tour cost > 0",            cost5 > 0)
    failures += 0 if ok else 1
    ok = check("5-city tour_cost consistent",
               abs(tour_cost(path5, dist_5) - cost5) < 1e-9)
    failures += 0 if ok else 1

    # all_optimal_tours must agree on cost
    opt_tours, opt_cost = all_optimal_tours(dist_5)
    ok = check("all_optimal_tours agrees on cost",
               abs(opt_cost - cost5) < 1e-9,
               f"brute={cost5:.4f} all_opt={opt_cost:.4f}")
    failures += 0 if ok else 1
    ok = check("at least one optimal tour found", len(opt_tours) >= 1)
    failures += 0 if ok else 1

    # ── solve_and_time returns expected keys ──────────────────────────────────
    result = solve_and_time(dist_5)
    for key in ("path", "cost", "runtime_ms", "nodes_explored", "n_cities"):
        ok = check(f"solve_and_time has key '{key}'", key in result)
        failures += 0 if ok else 1

    ok = check("nodes_explored == (n-1)! == 24",
               result["nodes_explored"] == math.factorial(4),
               f"got {result['nodes_explored']}")
    failures += 0 if ok else 1

    return failures


def test_edge_cases() -> int:
    """Verify graceful handling of 1-city and 2-city degenerate instances."""
    section("4. Edge cases")
    failures = 0

    # 1 city
    cities_1 = [City("A", 0, 0)]
    dist_1   = build_distance_matrix(cities_1)
    path1, cost1 = brute_force_tsp(dist_1)

    ok = check("1-city cost == 0.0", abs(cost1 - 0.0) < 1e-9, f"got {cost1}")
    failures += 0 if ok else 1

    start_1 = make_start_state(0)
    ok = check("1-city is immediately goal", start_1.is_goal(n=1))
    failures += 0 if ok else 1

    # 2 cities
    cities_2 = [City("A", 0, 0), City("B", 3, 4)]
    dist_2   = build_distance_matrix(cities_2)
    path2, cost2 = brute_force_tsp(dist_2)

    ok = check("2-city cost == 10.0 (5+5 return)",
               abs(cost2 - 10.0) < 1e-6, f"got {cost2:.4f}")
    failures += 0 if ok else 1

    # Random seed reproducibility
    c1 = generate_random_cities(5, seed=99)
    c2 = generate_random_cities(5, seed=99)
    ok = check("random cities reproducible with same seed",
               all(a.x == b.x and a.y == b.y for a, b in zip(c1, c2)))
    failures += 0 if ok else 1

    # Different seeds differ
    c3 = generate_random_cities(5, seed=1)
    ok = check("different seeds produce different cities",
               any(a.x != b.x or a.y != b.y for a, b in zip(c1, c3)))
    failures += 0 if ok else 1

    return failures


def test_integration() -> int:
    """
    Full pipeline check: load cities → build matrix → run baseline.
    This is the test A* must eventually pass too.
    """
    section("5. Integration — full pipeline")
    failures = 0

    cities = [
        City("A", 0, 0), City("B", 2, 4), City("C", 5, 2),
        City("D", 6, 6), City("E", 1, 7),
    ]
    dist = build_distance_matrix(cities)

    # Build state space manually and confirm expand works end-to-end
    start = make_start_state(0)
    frontier = [start]
    all_states_seen = 0

    # BFS expansion (not A*, just checking state graph is correct)
    visited_ids: set[tuple] = set()
    while frontier:
        state = frontier.pop()
        key = (state.current_city, state.visited)
        if key in visited_ids:
            continue
        visited_ids.add(key)
        all_states_seen += 1

        if not state.is_goal(n=5):
            frontier.extend(state.expand(dist))

    # For n=5, state space = sum_{k=1}^{5} C(4, k-1)*(k-1)! = 1+4+12+24+24 = 65
    ok = check("state space size == 65 for n=5",
               all_states_seen == 65,
               f"got {all_states_seen}")
    failures += 0 if ok else 1

    # Run baseline and verify the tour is actually valid
    path, cost = brute_force_tsp(dist)
    recomputed = tour_cost(path, dist)

    ok = check("recomputed cost matches returned cost",
               abs(recomputed - cost) < 1e-9)
    failures += 0 if ok else 1
    ok = check("tour length == n+1 (includes return)",
               len(path) == len(cities) + 1)
    failures += 0 if ok else 1
    ok = check("tour visits every city exactly once",
               sorted(set(path)) == list(range(len(cities))))
    failures += 0 if ok else 1

    print(f"\n  Optimal tour : {' → '.join(cities[i].name for i in path)}")
    print(f"  Total cost   : {cost:.4f}")

    return failures


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nTSP Foundation — Verification Suite")
    print(SEP)

    total_failures = 0
    total_failures += test_distance_matrix()
    total_failures += test_state_representation()
    total_failures += test_baseline_solver()
    total_failures += test_edge_cases()
    total_failures += test_integration()

    print(f"\n{SEP}")
    if total_failures == 0:
        print("  ALL TESTS PASSED — foundation is solid.")
        print("  Ready to move to Phase 2: A* core.")
    else:
        print(f"  {total_failures} TEST(S) FAILED — fix before proceeding.")
    print(SEP)
    sys.exit(0 if total_failures == 0 else 1)