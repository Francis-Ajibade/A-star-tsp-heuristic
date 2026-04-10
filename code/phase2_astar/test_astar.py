from __future__ import annotations
import sys
import os
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase2_astar'))

from tsp import City, build_distance_matrix, generate_random_cities
from state import TSPState, make_start_state
from baseline import brute_force_tsp, tour_cost, all_optimal_tours
from astar import astar_tsp, null_heuristic, min_edge_heuristic, AStarResult
from frontier import Frontier, VisitedSet, astar_with_frontier


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


def make_dist(*cities: City) -> list[list[float]]:
    return build_distance_matrix(list(cities))


# ── Test 1: null heuristic == brute force ─────────────────────────────────────

def test_null_heuristic_correctness() -> int:
    """
    A* with h(n)=0 is Uniform Cost Search and MUST find the exact
    optimal cost on every instance. If this fails, the bug is in
    state expansion or the search loop — not the heuristic.
    """
    section("1. Null heuristic correctness (UCS baseline)")
    failures = 0

    instances = [
        # (label, cities)
        ("unit square 4",  [City("A",0,0), City("B",1,0), City("C",1,1), City("D",0,1)]),
        ("collinear 3",    [City("A",0,0), City("B",1,0), City("C",2,0)]),
        ("standard 5",     [City("A",0,0), City("B",2,4), City("C",5,2),
                            City("D",6,6), City("E",1,7)]),
        ("random 6",       generate_random_cities(6, seed=42)),
        ("random 7",       generate_random_cities(7, seed=7)),
        ("random 8",       generate_random_cities(8, seed=13)),
    ]

    for label, cities in instances:
        dist          = build_distance_matrix(cities)
        bf_path, bf_cost = brute_force_tsp(dist)
        result        = astar_tsp(dist, null_heuristic)

        ok = check(
            f"{label}: A* cost == brute-force",
            result is not None and abs(result.cost - bf_cost) < 1e-6,
            f"A*={result.cost:.4f} BF={bf_cost:.4f}" if result else "no result",
        )
        failures += 0 if ok else 1

        if result:
            ok = check(
                f"{label}: tour visits all cities",
                len(set(result.path)) == len(cities),
            )
            failures += 0 if ok else 1

            ok = check(
                f"{label}: tour starts and ends at city 0",
                result.path[0] == 0 and result.path[-1] == 0,
            )
            failures += 0 if ok else 1

            ok = check(
                f"{label}: tour cost consistent with path",
                abs(tour_cost(result.path, dist) - result.cost) < 1e-6,
            )
            failures += 0 if ok else 1

    return failures


# ── Test 2: min-edge heuristic correctness ────────────────────────────────────

def test_min_edge_correctness() -> int:
    """
    Min-edge heuristic must still find the optimal cost on all instances.
    A heuristic is only useful if it doesn't sacrifice optimality.
    """
    section("2. Min-edge heuristic correctness")
    failures = 0

    instances = [
        ("unit square 4", [City("A",0,0), City("B",1,0), City("C",1,1), City("D",0,1)]),
        ("standard 5",    [City("A",0,0), City("B",2,4), City("C",5,2),
                           City("D",6,6), City("E",1,7)]),
        ("random 6",      generate_random_cities(6, seed=42)),
        ("random 7",      generate_random_cities(7, seed=7)),
        ("random 8",      generate_random_cities(8, seed=13)),
    ]

    for label, cities in instances:
        dist             = build_distance_matrix(cities)
        _, bf_cost       = brute_force_tsp(dist)
        result           = astar_tsp(dist, min_edge_heuristic)

        ok = check(
            f"{label}: min-edge cost == brute-force",
            result is not None and abs(result.cost - bf_cost) < 1e-6,
            f"A*={result.cost:.4f} BF={bf_cost:.4f}" if result else "no result",
        )
        failures += 0 if ok else 1

    return failures


# ── Test 3: heuristic admissibility ──────────────────────────────────────────

def test_admissibility() -> int:
    """
    For every state reachable in a 5-city instance, verify:
        h(state) <= true remaining cost to complete the tour optimally

    This is the mathematical proof-by-exhaustion that both heuristics
    are admissible. Include this output in your report.
    """
    section("3. Admissibility check — h(n) <= true remaining cost")
    failures = 0

    cities = [
        City("A",0,0), City("B",2,4), City("C",5,2),
        City("D",6,6), City("E",1,7),
    ]
    dist = build_distance_matrix(cities)
    n    = len(cities)

    # Collect every reachable state via BFS
    all_states : list[TSPState] = []
    queue       = [make_start_state(0)]
    seen        : set[tuple]    = set()

    while queue:
        state = queue.pop(0)
        key   = (state.current_city, state.visited)
        if key in seen:
            continue
        seen.add(key)
        all_states.append(state)
        if not state.is_goal(n):
            queue.extend(state.expand(dist))

    print(f"  States checked: {len(all_states)}")

    heuristics = [
        ("null",     null_heuristic),
        ("min-edge", min_edge_heuristic),
    ]

    for h_name, h_fn in heuristics:
        violations = 0
        for state in all_states:
            h_val = h_fn(state, dist)

            # True remaining cost: cheapest completion from this state
            true_remaining = _true_remaining_cost(state, dist, n)

            if h_val > true_remaining + 1e-6:
                violations += 1
                print(f"  VIOLATION: {h_name} h={h_val:.4f} > true={true_remaining:.4f} "
                      f"at city={state.current_city} visited={sorted(state.visited)}")

        ok = check(
            f"{h_name}: admissible across all {len(all_states)} states",
            violations == 0,
            f"{violations} violations" if violations else "no violations",
        )
        failures += 0 if ok else 1

    return failures


def _true_remaining_cost(
    state : TSPState,
    dist  : list[list[float]],
    n     : int,
) -> float:
    """
    Compute the true minimum remaining cost to complete the tour from state.
    Uses brute-force over remaining cities — only feasible on small n.
    """
    unvisited = [c for c in range(n) if c not in state.visited]

    if not unvisited:
        # Only return edge remains
        return dist[state.current_city][state.path[0]]

    best = float('inf')
    from itertools import permutations
    for perm in permutations(unvisited):
        cost  = dist[state.current_city][perm[0]]
        cost += sum(dist[perm[i]][perm[i+1]] for i in range(len(perm)-1))
        cost += dist[perm[-1]][state.path[0]]   # return to start
        best  = min(best, cost)

    return best


# ── Test 4: heuristic pruning power ──────────────────────────────────────────

def test_pruning_power() -> int:
    """
    A better heuristic must expand fewer nodes than a weaker one.
    null >= min_edge in nodes expanded (null is worst, min-edge is better).
    This sets up the comparison table you'll extend in Phase 3 with MST.
    """
    section("4. Pruning power — nodes expanded comparison")
    failures = 0

    instances = [
        ("standard 5", [City("A",0,0), City("B",2,4), City("C",5,2),
                        City("D",6,6), City("E",1,7)]),
        ("random 6",   generate_random_cities(6, seed=42)),
        ("random 7",   generate_random_cities(7, seed=7)),
        ("random 8",   generate_random_cities(8, seed=13)),
    ]

    print(f"\n  {'Instance':<14} {'Null':>8} {'Min-edge':>10} {'Saved':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*8}")

    for label, cities in instances:
        dist   = build_distance_matrix(cities)
        r_null = astar_tsp(dist, null_heuristic)
        r_me   = astar_tsp(dist, min_edge_heuristic)

        saved  = r_null.nodes_expanded - r_me.nodes_expanded
        print(f"  {label:<14} {r_null.nodes_expanded:>8} "
              f"{r_me.nodes_expanded:>10} {saved:>8}")

        ok = check(
            f"{label}: min-edge expands <= null",
            r_me.nodes_expanded <= r_null.nodes_expanded,
        )
        failures += 0 if ok else 1

    return failures


# ── Test 5: Frontier + VisitedSet unit tests ──────────────────────────────────

def test_frontier_and_visited() -> int:
    """
    Unit tests for the Frontier and VisitedSet classes in frontier.py.
    Verifies ordering, staleness detection, and deduplication independently
    of the full A* loop.
    """
    section("5. Frontier and VisitedSet unit tests")
    failures = 0

    # ── Frontier ordering ─────────────────────────────────────────────────────
    fr = Frontier()
    vs = VisitedSet()

    s_high = make_start_state(0)
    s_low  = make_start_state(0)

    fr.push(s_high, f=9.0)
    fr.push(s_low,  f=2.0)
    fr.push(s_high, f=5.0)

    vs.should_visit(s_low)   # register so nothing is stale

    popped = fr.pop(vs)
    ok = check("frontier pops lowest f first",
               popped is not None and popped.g_cost == s_low.g_cost)
    failures += 0 if ok else 1

    ok = check("frontier peak size tracked", fr.peak_size >= 2)
    failures += 0 if ok else 1

    ok = check("frontier not empty after one pop", not fr.is_empty())
    failures += 0 if ok else 1

    # ── VisitedSet deduplication ──────────────────────────────────────────────
    vs2 = VisitedSet()

    state_cheap = TSPState(
        current_city=1,
        visited=frozenset({0, 1}),
        g_cost=1.0,
        path=[0, 1],
    )
    state_expensive = TSPState(
        current_city=1,
        visited=frozenset({0, 1}),
        g_cost=5.0,
        path=[0, 1],
    )

    ok = check("first visit to state accepted",
               vs2.should_visit(state_cheap))
    failures += 0 if ok else 1

    ok = check("more expensive path to same state rejected",
               not vs2.should_visit(state_expensive))
    failures += 0 if ok else 1

    ok = check("expensive state is stale",
               vs2.is_stale(state_expensive))
    failures += 0 if ok else 1

    ok = check("cheap state is not stale",
               not vs2.is_stale(state_cheap))
    failures += 0 if ok else 1

    ok = check("updates counter stays 0 (no cheaper path found yet)",
               vs2.updates == 0)
    failures += 0 if ok else 1

    # Push a cheaper path to same state — should trigger an update
    state_cheaper = TSPState(
        current_city=1,
        visited=frozenset({0, 1}),
        g_cost=0.5,
        path=[0, 1],
    )
    ok = check("even cheaper path accepted",
               vs2.should_visit(state_cheaper))
    failures += 0 if ok else 1

    ok = check("updates counter incremented",
               vs2.updates == 1, f"got {vs2.updates}")
    failures += 0 if ok else 1

    ok = check("states_seen == 1 (same key, different costs)",
               vs2.states_seen() == 1, f"got {vs2.states_seen()}")
    failures += 0 if ok else 1

    # ── astar_with_frontier agrees with astar_tsp ─────────────────────────────
    cities = [City("A",0,0), City("B",2,4), City("C",5,2),
              City("D",6,6), City("E",1,7)]
    dist   = build_distance_matrix(cities)
    _, bf  = brute_force_tsp(dist)

    r_frontier = astar_with_frontier(dist, null_heuristic)
    ok = check("astar_with_frontier cost == brute-force",
               r_frontier is not None and abs(r_frontier["cost"] - bf) < 1e-6,
               f"got {r_frontier['cost']:.4f}" if r_frontier else "None")
    failures += 0 if ok else 1

    ok = check("frontier_peak reported in result", "frontier_peak" in r_frontier)
    failures += 0 if ok else 1

    ok = check("memory_bytes reported in result", "memory_bytes" in r_frontier)
    failures += 0 if ok else 1

    return failures


# ── Test 6: edge cases ────────────────────────────────────────────────────────

def test_edge_cases() -> int:
    """
    Degenerate inputs: 1 city, 2 cities, already-at-goal start.
    A* must not crash or return nonsense on these.
    """
    section("6. Edge cases")
    failures = 0

    # 1 city
    dist_1 = build_distance_matrix([City("A",0,0)])
    r1     = astar_tsp(dist_1, null_heuristic)
    ok = check("1-city: result not None",   r1 is not None)
    failures += 0 if ok else 1
    if r1:
        ok = check("1-city: cost == 0.0",  abs(r1.cost - 0.0) < 1e-9,
                   f"got {r1.cost}")
        failures += 0 if ok else 1

    # 2 cities
    dist_2 = build_distance_matrix([City("A",0,0), City("B",3,4)])
    r2     = astar_tsp(dist_2, null_heuristic)
    ok = check("2-city: cost == 10.0",
               r2 is not None and abs(r2.cost - 10.0) < 1e-6,
               f"got {r2.cost:.4f}" if r2 else "None")
    failures += 0 if ok else 1

    # AStarResult has all expected fields
    if r2:
        for field in ("path", "cost", "nodes_expanded",
                      "nodes_generated", "runtime_ms", "n_cities"):
            ok = check(f"AStarResult has field '{field}'", hasattr(r2, field))
            failures += 0 if ok else 1

    return failures


# ── Test 7: result integrity ──────────────────────────────────────────────────

def test_result_integrity() -> int:
    """
    Structural checks on AStarResult — the object Phase 4 will consume
    for logging and charting. Every field must be present and sensible.
    """
    section("7. AStarResult integrity")
    failures = 0

    cities = [City("A",0,0), City("B",2,4), City("C",5,2),
              City("D",6,6), City("E",1,7)]
    dist   = build_distance_matrix(cities)
    result = astar_tsp(dist, null_heuristic)

    ok = check("result is not None", result is not None)
    failures += 0 if ok else 1
    if not result:
        return failures

    ok = check("cost > 0",                   result.cost > 0)
    failures += 0 if ok else 1
    ok = check("nodes_expanded > 0",         result.nodes_expanded > 0)
    failures += 0 if ok else 1
    ok = check("nodes_generated >= expanded", result.nodes_generated >= result.nodes_expanded)
    failures += 0 if ok else 1
    ok = check("runtime_ms > 0",             result.runtime_ms > 0)
    failures += 0 if ok else 1
    ok = check("n_cities == 5",              result.n_cities == 5)
    failures += 0 if ok else 1
    ok = check("path length == n+1",         len(result.path) == 6)
    failures += 0 if ok else 1
    ok = check("path starts at 0",           result.path[0] == 0)
    failures += 0 if ok else 1
    ok = check("path ends at 0",             result.path[-1] == 0)
    failures += 0 if ok else 1
    ok = check("path visits all 5 cities",   len(set(result.path)) == 5)
    failures += 0 if ok else 1
    ok = check("path cost consistent",
               abs(tour_cost(result.path, dist) - result.cost) < 1e-6)
    failures += 0 if ok else 1

    return failures


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nPhase 2 — A* Test Suite")
    print(SEP)

    total = 0
    total += test_null_heuristic_correctness()
    total += test_min_edge_correctness()
    total += test_admissibility()
    total += test_pruning_power()
    total += test_frontier_and_visited()
    total += test_edge_cases()
    total += test_result_integrity()

    print(f"\n{SEP}")
    if total == 0:
        print("  ALL TESTS PASSED — Phase 2 complete.")
        print("  Ready to move to Phase 3: heuristic design.")
    else:
        print(f"  {total} TEST(S) FAILED — fix before proceeding.")
    print(SEP)
    sys.exit(0 if total == 0 else 1)