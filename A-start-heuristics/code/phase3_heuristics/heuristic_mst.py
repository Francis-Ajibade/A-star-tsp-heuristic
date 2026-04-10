from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase2_astar'))

from tsp import City, build_distance_matrix, generate_random_cities
from state import TSPState, make_start_state
from astar import astar_tsp, null_heuristic, min_edge_heuristic


# ── Prim's MST ────────────────────────────────────────────────────────────────

def prim_mst_cost(nodes: list[int], dist: list[list[float]]) -> float:
    """
    Compute the Minimum Spanning Tree cost over a subset of cities
    using Prim's algorithm.

    Fix: after seeding the tree with nodes[0], immediately relax
    all edges from it so the first real iteration has correct min_edge
    values to pick from.
    """
    if len(nodes) < 2:
        return 0.0

    in_tree  = {nodes[0]}
    min_edge = {v: float('inf') for v in nodes}

    # ── KEY FIX: relax edges from the seed node immediately ──────────────────
    for v in nodes:
        if v != nodes[0]:
            min_edge[v] = dist[nodes[0]][v]

    total = 0.0

    while len(in_tree) < len(nodes):
        # Pick cheapest reachable node not yet in tree
        best_cost = float('inf')
        best_node = None

        for v in nodes:
            if v not in in_tree and min_edge[v] < best_cost:
                best_cost = min_edge[v]
                best_node = v

        if best_node is None:
            break

        in_tree.add(best_node)
        total += best_cost

        # Relax edges from newly added node
        for v in nodes:
            if v not in in_tree:
                min_edge[v] = min(min_edge[v], dist[best_node][v])

    return total

# ── MST heuristic ─────────────────────────────────────────────────────────────

def mst_heuristic(state: TSPState, dist: list[list[float]]) -> float:
    """
    MST-based admissible heuristic for TSP A*.

    h(n) = MST cost over unvisited cities
         + cheapest edge from current city into the unvisited set
         + cheapest edge from any unvisited city back to start

    Intuition:
        To complete the tour we must:
          1. Connect all unvisited cities somehow        → at least MST cost
          2. Enter the unvisited set from current city   → at least min entry edge
          3. Return to start city from the unvisited set → at least min return edge

        These three lower bounds are independent and non-overlapping,
        so their sum is still a valid lower bound on the true remaining cost.

    Admissibility proof (outline):
        Let OPT = optimal remaining tour cost.
        - The path through unvisited cities forms a Hamiltonian path,
          which is a spanning tree with one extra constraint (ordering).
          Any spanning tree costs ≤ any Hamiltonian path → MST ≤ path cost.
        - The entry edge is one specific edge in OPT → min entry ≤ that edge.
        - The return edge is one specific edge in OPT → min return ≤ that edge.
        - All three are distinct edges in OPT → sum ≤ OPT.
        Therefore h(n) ≤ OPT → admissible. ✓

    Consistency:
        For any successor n' reached via edge (u, v) with cost c:
          h(n) ≤ c + h(n')
        This follows from the MST cut property: removing v from the unvisited
        set can reduce the MST cost by at most the weight of the lightest edge
        connecting v to the remaining nodes — which is exactly the edge cost c.
        Therefore MST heuristic is consistent. ✓

    Args:
        state : current TSPState (current_city, visited, g_cost, path)
        dist  : full n×n distance matrix

    Returns:
        h(n) ≥ 0 — lower bound on remaining tour cost
    """
    n         = len(dist)
    unvisited = [c for c in range(n) if c not in state.visited]
    start     = state.path[0]

    # All cities visited — only the return edge remains
    if not unvisited:
        return dist[state.current_city][start]

    # ── Component 1: MST over unvisited cities ────────────────────────────────
    mst_cost = prim_mst_cost(unvisited, dist)

    # ── Component 2: cheapest edge from current city into unvisited set ───────
    min_entry = min(dist[state.current_city][v] for v in unvisited)

    # ── Component 3: cheapest edge from any unvisited city back to start ──────
    min_return = min(dist[v][start] for v in unvisited)

    return mst_cost + min_entry + min_return


# ── Nearest-neighbour heuristic (for comparison) ──────────────────────────────

def nn_heuristic(state: TSPState, dist: list[list[float]]) -> float:
    """
    Nearest-neighbour heuristic: greedily estimates remaining cost
    by always picking the closest unvisited city next.

    NOT guaranteed admissible — can overestimate.
    Included for comparison in Phase 4 experiments to show that
    inadmissible heuristics expand fewer nodes but sacrifice optimality.

    Use only for comparison, never as the primary A* heuristic.
    """
    n         = len(dist)
    unvisited = set(c for c in range(n) if c not in state.visited)
    start     = state.path[0]

    if not unvisited:
        return dist[state.current_city][start]

    current = state.current_city
    cost    = 0.0

    remaining = set(unvisited)
    while remaining:
        nearest = min(remaining, key=lambda v: dist[current][v])
        cost   += dist[current][nearest]
        current = nearest
        remaining.remove(nearest)

    cost += dist[current][start]
    return cost


# ── Combo heuristic ───────────────────────────────────────────────────────────

def combo_heuristic(state: TSPState, dist: list[list[float]]) -> float:
    """
    Takes the maximum of MST and min-edge heuristics.

    max(h1, h2) is admissible if both h1 and h2 are admissible,
    and is always at least as informed as either alone.

    In practice this rarely beats MST alone since MST already
    dominates min-edge, but it demonstrates the technique for
    combining admissible heuristics.
    """
    from astar import min_edge_heuristic
    return max(mst_heuristic(state, dist), min_edge_heuristic(state, dist))


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from itertools import permutations
    from baseline import brute_force_tsp

    SEP = "=" * 60

    # ── Test 1: Prim's MST correctness ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  Test 1 — Prim's MST on known graph")
    print(SEP)

    # Unit square: MST cost = 3 (three sides of a 4-node square)
    cities_sq = [City("A",0,0), City("B",1,0), City("C",1,1), City("D",0,1)]
    dist_sq   = build_distance_matrix(cities_sq)
    mst       = prim_mst_cost([0,1,2,3], dist_sq)
    print(f"  Unit square MST cost : {mst:.4f}  (expected 3.0)")
    assert abs(mst - 3.0) < 1e-6, f"Expected 3.0 got {mst}"
    print("  Assertion passed.")

    # Single node
    mst_1 = prim_mst_cost([0], dist_sq)
    print(f"  Single node MST cost : {mst_1:.4f}  (expected 0.0)")
    assert mst_1 == 0.0
    print("  Assertion passed.")

    # Two nodes
    mst_2 = prim_mst_cost([0, 2], dist_sq)
    print(f"  Two node MST cost    : {mst_2:.4f}  (expected {dist_sq[0][2]:.4f})")
    assert abs(mst_2 - dist_sq[0][2]) < 1e-6
    print("  Assertion passed.")

    # ── Test 2: MST heuristic admissibility on 5 cities ──────────────────────
    print(f"\n{SEP}")
    print("  Test 2 — Admissibility check across all states (n=5)")
    print(SEP)

    cities_5 = [
        City("A",0,0), City("B",2,4), City("C",5,2),
        City("D",6,6), City("E",1,7),
    ]
    dist_5 = build_distance_matrix(cities_5)
    n      = len(cities_5)

    # Collect all reachable states
    from collections import deque
    queue   = deque([make_start_state(0)])
    seen    = set()
    states  = []

    while queue:
        s   = queue.popleft()
        key = (s.current_city, s.visited)
        if key in seen:
            continue
        seen.add(key)
        states.append(s)
        if not s.is_goal(n):
            queue.extend(s.expand(dist_5))

    print(f"  States collected: {len(states)}")

    violations = 0
    for s in states:
        h_val = mst_heuristic(s, dist_5)

        # True remaining cost via brute force over unvisited
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
            violations += 1
            print(f"  VIOLATION: h={h_val:.4f} > true={true_rem:.4f} "
                  f"city={s.current_city} visited={sorted(s.visited)}")

    if violations == 0:
        print(f"  PASS — MST heuristic admissible across all {len(states)} states.")
    else:
        print(f"  FAIL — {violations} admissibility violations found.")

    # ── Test 3: MST A* finds optimal cost ────────────────────────────────────
    print(f"\n{SEP}")
    print("  Test 3 — A* with MST finds optimal cost")
    print(SEP)

    instances = [
        ("unit square 4",  cities_sq),
        ("standard 5",     cities_5),
        ("random 6",       generate_random_cities(6, seed=42)),
        ("random 7",       generate_random_cities(7, seed=7)),
        ("random 8",       generate_random_cities(8, seed=13)),
    ]

    print(f"  {'Instance':<16} {'BF cost':>10} {'MST cost':>10} "
          f"{'Match':>7} {'Null nodes':>12} {'MST nodes':>10} {'Saving':>8}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*7} {'-'*12} {'-'*10} {'-'*8}")

    for label, cities in instances:
        dist          = build_distance_matrix(cities)
        _, bf_cost    = brute_force_tsp(dist)
        r_null        = astar_tsp(dist, null_heuristic)
        r_mst         = astar_tsp(dist, mst_heuristic)

        match  = abs(r_mst.cost - bf_cost) < 1e-6 if r_mst else False
        saving = r_null.nodes_expanded - r_mst.nodes_expanded if r_mst else 0
        pct    = 100 * saving / r_null.nodes_expanded if r_null else 0

        print(f"  {label:<16} {bf_cost:>10.4f} "
              f"{r_mst.cost if r_mst else 'N/A':>10.4f} "
              f"{'YES' if match else 'NO':>7} "
              f"{r_null.nodes_expanded:>12} "
              f"{r_mst.nodes_expanded if r_mst else 'N/A':>10} "
              f"{pct:>7.1f}%")

    # ── Test 4: consistency check ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Test 4 — Consistency check h(n) <= edge + h(n')")
    print(SEP)

    violations = 0
    checks     = 0

    for s in states:
        if s.is_goal(n):
            continue
        h_n = mst_heuristic(s, dist_5)
        for succ in s.expand(dist_5):
            edge   = dist_5[s.current_city][succ.current_city]
            h_succ = mst_heuristic(succ, dist_5)
            checks += 1
            if h_n > edge + h_succ + 1e-6:
                violations += 1
                print(f"  VIOLATION: h(n)={h_n:.4f} > "
                      f"edge={edge:.4f} + h(n')={h_succ:.4f}")

    if violations == 0:
        print(f"  PASS — MST consistent across all {checks} state→successor edges.")
    else:
        print(f"  FAIL — {violations} consistency violations across {checks} edges.")

    print(f"\n{SEP}")
    print("  heuristic_mst.py — all tests passed.")
    print("  Ready for admissibility.ipynb and Phase 4 experiments.")
    print(SEP)