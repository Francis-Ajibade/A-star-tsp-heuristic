from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase2_astar'))

from tsp import City, build_distance_matrix, generate_random_cities
from state import TSPState, make_start_state
from astar import astar_tsp, null_heuristic, min_edge_heuristic
from baseline import brute_force_tsp
from heuristic_mst import mst_heuristic


# ── Combo heuristic ───────────────────────────────────────────────────────────

def combo_heuristic(state: TSPState, dist: list[list[float]]) -> float:
    """
    Combination heuristic: max(MST, min-edge).

    Theory:
        If h1 and h2 are both admissible, then max(h1, h2) is also
        admissible — and is at least as informed as either alone.

        Proof: for any state n,
          h1(n) <= true_remaining  AND  h2(n) <= true_remaining
          => max(h1, h2) <= true_remaining  ✓

        Consistency: max of two consistent heuristics is consistent.
          h1(n) <= c + h1(n')  AND  h2(n) <= c + h2(n')
          => max(h1,h2)(n) <= c + max(h1,h2)(n')  ✓

    In practice:
        MST already dominates min-edge almost everywhere — the combo
        rarely beats pure MST. Its value is pedagogical: it demonstrates
        how admissible heuristics can be safely combined, which is the
        foundation for more advanced techniques like pattern databases.

    Returns:
        max(mst_heuristic, min_edge_heuristic) at this state
    """
    h_mst      = mst_heuristic(state, dist)
    h_min_edge = min_edge_heuristic(state, dist)
    return max(h_mst, h_min_edge)


def weighted_combo_heuristic(
    state  : TSPState,
    dist   : list[list[float]],
    w_mst  : float = 0.7,
    w_me   : float = 0.3,
) -> float:
    """
    Weighted blend of MST and min-edge heuristics.

    h(n) = w_mst * MST(n) + w_me * min_edge(n)

    Admissibility:
        A weighted sum is NOT guaranteed admissible unless you verify
        w_mst + w_me <= 1 AND each component is already a lower bound.
        For safety this defaults to max() — use weighted only if you
        have verified admissibility empirically via admissibility.ipynb.

    Included here as an experiment: vary w_mst and w_me in Phase 4
    to see how the blend affects nodes expanded vs optimality.
    """
    h_mst      = mst_heuristic(state, dist)
    h_min_edge = min_edge_heuristic(state, dist)
    return w_mst * h_mst + w_me * h_min_edge


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import deque
    from itertools import permutations

    SEP = "=" * 60

    cities_5 = [
        City("A", 0, 0), City("B", 2, 4), City("C", 5, 2),
        City("D", 6, 6), City("E", 1, 7),
    ]
    dist_5   = build_distance_matrix(cities_5)
    n        = len(cities_5)
    _, bf_cost = brute_force_tsp(dist_5)

    # ── Test 1: combo >= both components at every state ───────────────────────
    print(f"\n{SEP}")
    print("  Test 1 — Combo always >= both components")
    print(SEP)

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

    failures = 0
    for s in states:
        h_mst   = mst_heuristic(s, dist_5)
        h_me    = min_edge_heuristic(s, dist_5)
        h_combo = combo_heuristic(s, dist_5)

        if h_combo < h_mst - 1e-9 or h_combo < h_me - 1e-9:
            failures += 1
            print(f"  FAIL: combo={h_combo:.4f} < mst={h_mst:.4f} or me={h_me:.4f}")

    if failures == 0:
        print(f"  PASS — combo >= max(mst, min_edge) across all {len(states)} states.")

    # ── Test 2: admissibility of combo ────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Test 2 — Admissibility of combo heuristic")
    print(SEP)

    violations = 0
    for s in states:
        h_val = combo_heuristic(s, dist_5)

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
            print(f"  VIOLATION: h={h_val:.4f} true={true_rem:.4f} "
                  f"city={s.current_city} visited={sorted(s.visited)}")

    status = "PASS — ADMISSIBLE" if violations == 0 else f"FAIL — {violations} violations"
    print(f"  {status} across all {len(states)} states.")

    # ── Test 3: full comparison table ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Test 3 — All heuristics comparison")
    print(SEP)

    from heuristic_nn import nn_heuristic

    HEURISTICS = {
        "Null"    : null_heuristic,
        "Min-edge": min_edge_heuristic,
        "MST"     : mst_heuristic,
        "Combo"   : combo_heuristic,
        "NN*"     : nn_heuristic,
    }

    instances = [
        ("standard 5", cities_5),
        ("random 6",   generate_random_cities(6, seed=42)),
        ("random 7",   generate_random_cities(7, seed=7)),
        ("random 8",   generate_random_cities(8, seed=13)),
    ]

    for label, cities in instances:
        dist       = build_distance_matrix(cities)
        _, bf_cost = brute_force_tsp(dist)
        print(f"\n  {label}  (BF optimal = {bf_cost:.4f})")
        print(f"  {'Heuristic':<12} {'Nodes':>8} {'Cost':>10} {'Optimal?':>9} {'Time ms':>9}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*9} {'-'*9}")

        for h_name, h_fn in HEURISTICS.items():
            r       = astar_tsp(dist, h_fn)
            optimal = abs(r.cost - bf_cost) < 1e-6 if r else False
            print(f"  {h_name:<12} "
                  f"{r.nodes_expanded if r else 'N/A':>8} "
                  f"{r.cost if r else 'N/A':>10.4f} "
                  f"{'YES' if optimal else 'NO*':>9} "
                  f"{r.runtime_ms if r else 'N/A':>9.3f}")

    print(f"\n  * NN is inadmissible — 'NO' means a sub-optimal tour was returned.")
    print(f"\n{SEP}")
    print("  heuristic_combo.py smoke test complete.")
    print("  Phase 3 fully done — ready for run_experiments.py.")
    print(SEP)