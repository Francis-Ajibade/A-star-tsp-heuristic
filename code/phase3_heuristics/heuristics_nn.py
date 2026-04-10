from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase2_astar'))

from tsp import City, build_distance_matrix, generate_random_cities
from state import TSPState, make_start_state
from astar import astar_tsp, null_heuristic
from baseline import brute_force_tsp


# ── Nearest-neighbour heuristic ───────────────────────────────────────────────

def nn_heuristic(state: TSPState, dist: list[list[float]]) -> float:
    """
    Nearest-neighbour greedy estimate of remaining tour cost.

    From the current city, repeatedly visit the closest unvisited
    city until all are visited, then return to start.

    Admissibility: NOT guaranteed.
      The greedy path can overestimate the true remaining cost because
      it commits to locally cheap edges that force expensive ones later.
      Use only for comparison — never as the primary A* heuristic.

    Informativeness: often very tight in practice on random instances,
      which is why it expands fewer nodes than MST despite being
      inadmissible. This is your 'striking failure' case — fewer nodes
      but wrong (non-optimal) answers.

    Time complexity: O(k²) where k = number of unvisited cities.
    """
    n         = len(dist)
    unvisited = set(c for c in range(n) if c not in state.visited)
    start     = state.path[0]

    if not unvisited:
        return dist[state.current_city][start]

    current   = state.current_city
    cost      = 0.0
    remaining = set(unvisited)

    while remaining:
        nearest  = min(remaining, key=lambda v: dist[current][v])
        cost    += dist[current][nearest]
        current  = nearest
        remaining.remove(nearest)

    cost += dist[current][start]
    return cost


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60

    cities_5 = [
        City("A", 0, 0), City("B", 2, 4), City("C", 5, 2),
        City("D", 6, 6), City("E", 1, 7),
    ]
    dist_5   = build_distance_matrix(cities_5)
    n        = len(cities_5)
    _, bf_cost = brute_force_tsp(dist_5)

    # ── Test 1: basic output check ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Test 1 — Basic output")
    print(SEP)

    start = make_start_state(0)
    h_val = nn_heuristic(start, dist_5)
    print(f"  h(start) = {h_val:.4f}")
    print(f"  BF cost  = {bf_cost:.4f}")
    print(f"  Admissible at start? {'YES' if h_val <= bf_cost + 1e-6 else 'NO  ← expected'}")

    # ── Test 2: admissibility violations (expected for NN) ────────────────────
    print(f"\n{SEP}")
    print("  Test 2 — Admissibility violations (NN is NOT admissible)")
    print(SEP)

    from collections import deque
    from itertools import permutations

    queue  = deque([make_start_state(0)])
    seen   = set()
    states = []

    while queue:
        s   = queue.popleft()
        key = (s.current_city, s.visited)
        if key in seen:
            continue
        seen.add(key)
        states.append(s)
        if not s.is_goal(n):
            queue.extend(s.expand(dist_5))

    violations = []
    for s in states:
        h_val = nn_heuristic(s, dist_5)

        unvisited = [c for c in range(n) if c not in s.visited]
        if not unvisited:
            true_rem = dist_5[s.current_city][s.path[0]]
        else:
            best = float('inf')
            for perm in permutations(unvisited):
                cost  = dist_5[s.current_city][perm[0]]
                cost += sum(dist_5[perm[i]][perm[i+1]] for i in range(len(perm)-1))
                cost += dist_5[perm[-1]][s.path[0]]
                best  = min(best, cost)
            true_rem = best

        if h_val > true_rem + 1e-6:
            violations.append((s, h_val, true_rem))

    print(f"  States checked : {len(states)}")
    print(f"  Violations     : {len(violations)}  (non-zero = NOT admissible, as expected)")
    for s, h, true in violations[:3]:
        print(f"    city={s.current_city} visited={sorted(s.visited)} "
              f"h={h:.4f} true={true:.4f} overestimate={h-true:.4f}")
    if len(violations) > 3:
        print(f"    ... and {len(violations)-3} more")

    # ── Test 3: A* with NN — nodes vs optimality ──────────────────────────────
    print(f"\n{SEP}")
    print("  Test 3 — A* with NN: fewer nodes but may sacrifice optimality")
    print(SEP)

    instances = [
        ("standard 5",  cities_5),
        ("random 6",    generate_random_cities(6, seed=42)),
        ("random 7",    generate_random_cities(7, seed=7)),
        ("random 8",    generate_random_cities(8, seed=13)),
    ]

    print(f"  {'Instance':<14} {'BF cost':>10} {'NN cost':>10} "
          f"{'Optimal?':>9} {'Null nodes':>11} {'NN nodes':>9}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*9} {'-'*11} {'-'*9}")

    for label, cities in instances:
        dist        = build_distance_matrix(cities)
        _, bf_cost  = brute_force_tsp(dist)
        r_null      = astar_tsp(dist, null_heuristic)
        r_nn        = astar_tsp(dist, nn_heuristic)

        optimal = abs(r_nn.cost - bf_cost) < 1e-6 if r_nn else False
        print(f"  {label:<14} {bf_cost:>10.4f} "
              f"{r_nn.cost if r_nn else 'N/A':>10.4f} "
              f"{'YES' if optimal else 'NO':>9} "
              f"{r_null.nodes_expanded:>11} "
              f"{r_nn.nodes_expanded if r_nn else 'N/A':>9}")

    print(f"\n  Key insight: NN expands fewer nodes than Null but")
    print(f"  may not find the optimal tour — inadmissibility in action.")
    print(f"\n{SEP}")
    print("  heuristic_nn.py smoke test complete.")
    print(SEP)