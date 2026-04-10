from __future__ import annotations
import heapq
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))

from tsp import City, build_distance_matrix
from state import TSPState, make_start_state


# ── Result dataclass ──────────────────────────────────────────────────────────

class AStarResult:
    """
    Holds everything A* produces — path, cost, and search statistics.
    Passed directly to the experiment logger in Phase 4.
    """

    def __init__(
        self,
        path       : list[int],
        cost       : float,
        nodes_expanded  : int,
        nodes_generated : int,
        runtime_ms : float,
        n_cities   : int,
    ):
        self.path            = path
        self.cost            = cost
        self.nodes_expanded  = nodes_expanded
        self.nodes_generated = nodes_generated
        self.runtime_ms      = runtime_ms
        self.n_cities        = n_cities

    def __repr__(self) -> str:
        tour = " → ".join(str(c) for c in self.path)
        return (
            f"AStarResult(\n"
            f"  tour     = {tour}\n"
            f"  cost     = {self.cost:.4f}\n"
            f"  expanded = {self.nodes_expanded}\n"
            f"  generated= {self.nodes_generated}\n"
            f"  time     = {self.runtime_ms:.3f} ms\n"
            f")"
        )

    def print_summary(self, cities: list[City] | None = None) -> None:
        if cities:
            tour_str = " → ".join(cities[i].name for i in self.path)
        else:
            tour_str = " → ".join(str(i) for i in self.path)

        print(f"  Tour          : {tour_str}")
        print(f"  Cost          : {self.cost:.4f}")
        print(f"  Nodes expanded: {self.nodes_expanded}")
        print(f"  Nodes generated:{self.nodes_generated}")
        print(f"  Runtime       : {self.runtime_ms:.3f} ms")


# ── Priority queue entry ──────────────────────────────────────────────────────

class _Entry:
    """
    Wraps a TSPState for insertion into heapq.

    heapq is a min-heap ordered by f(n) = g(n) + h(n).
    Ties broken by insertion order (counter) to avoid comparing TSPStates
    directly on f-value alone, which is fragile with floats.
    """

    __slots__ = ("f", "counter", "state")

    def __init__(self, f: float, counter: int, state: TSPState):
        self.f       = f
        self.counter = counter
        self.state   = state

    def __lt__(self, other: _Entry) -> bool:
        if self.f != other.f:
            return self.f < other.f
        return self.counter < other.counter   # FIFO on ties = breadth-first tiebreak


# ── Core A* algorithm ─────────────────────────────────────────────────────────

def astar_tsp(
    dist      : list[list[float]],
    heuristic : callable,
    start     : int = 0,
) -> AStarResult | None:
    """
    Solve TSP optimally using A* search.

    f(n) = g(n) + h(n)
      g(n) : exact cost from start to current state
      h(n) : heuristic estimate of remaining cost to complete the tour

    For A* to guarantee an optimal solution, h(n) must be:
      - Admissible  : h(n) <= true remaining cost  (never overestimates)
      - Consistent  : h(n) <= cost(n, n') + h(n')  (monotone along paths)

    Args:
        dist      : n×n distance matrix from phase1_foundation/tsp.py
        heuristic : callable(state, dist) -> float
        start     : starting city index (default 0)

    Returns:
        AStarResult on success, None if no solution exists
    """
    n              = len(dist)
    nodes_expanded = 0
    nodes_generated= 1         # start state counts
    counter        = 0         # tie-breaking insertion order

    # ── Initialise frontier with start state ──────────────────────────────────
    start_state = make_start_state(start)
    h0          = heuristic(start_state, dist)
    frontier    : list[_Entry] = []
    heapq.heappush(frontier, _Entry(h0, counter, start_state))

    # ── Visited set: (current_city, visited) → best g_cost seen ──────────────
    # If we reach the same (city, visited) again with a higher g, prune it.
    best_g: dict[tuple, float] = {
        (start_state.current_city, start_state.visited): 0.0
    }

    t0 = time.perf_counter()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while frontier:
        entry = heapq.heappop(frontier)
        state = entry.state

        # Stale entry check — a better path to this state was found later
        key    = (state.current_city, state.visited)
        if state.g_cost > best_g.get(key, float('inf')) + 1e-9:
            continue

        nodes_expanded += 1

        # ── Goal test ─────────────────────────────────────────────────────────
        if state.is_goal(n):
            runtime_ms = (time.perf_counter() - t0) * 1000
            final_cost = state.final_cost(dist)
            full_tour  = state.full_tour()
            return AStarResult(
                path            = full_tour,
                cost            = final_cost,
                nodes_expanded  = nodes_expanded,
                nodes_generated = nodes_generated,
                runtime_ms      = runtime_ms,
                n_cities        = n,
            )

        # ── Expand successors ─────────────────────────────────────────────────
        for successor in state.expand(dist):
            s_key = (successor.current_city, successor.visited)

            # Prune if we already found a cheaper path to this state
            if successor.g_cost >= best_g.get(s_key, float('inf')) - 1e-9:
                continue

            best_g[s_key] = successor.g_cost
            nodes_generated += 1

            h  = heuristic(successor, dist)
            f  = successor.g_cost + h
            counter += 1
            heapq.heappush(frontier, _Entry(f, counter, successor))

    # Frontier exhausted with no solution
    return None


# ── Null heuristic (acts as Uniform Cost Search) ──────────────────────────────

def null_heuristic(state: TSPState, dist: list[list[float]]) -> float:
    """
    h(n) = 0 always.
    Reduces A* to Uniform Cost Search — guaranteed optimal but blind.
    Use this in test_astar.py to verify A* finds the correct cost
    before plugging in real heuristics.
    """
    return 0.0


def min_edge_heuristic(state: TSPState, dist: list[list[float]]) -> float:
    """
    h(n) = (number of unvisited cities + 1 return edge) * global minimum edge.

    Admissible: the cheapest possible remaining edges can never exceed
    the true remaining cost.
    Fast to compute: O(1) after precomputing the global minimum edge.
    Weak: often a large underestimate, so doesn't prune much.
    Use as a stepping stone before MST heuristic in Phase 3.
    """
    n         = len(dist)
    unvisited = [c for c in range(n) if c not in state.visited]

    if not unvisited:
        # Only the return edge remains
        return dist[state.current_city][state.path[0]]

    # Global minimum non-zero edge
    min_edge = min(
        dist[i][j]
        for i in range(n)
        for j in range(n)
        if i != j
    )

    # Remaining edges = unvisited cities + 1 return edge
    remaining_edges = len(unvisited) + 1
    return remaining_edges * min_edge


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from baseline import brute_force_tsp

    SEP = "=" * 55

    cities = [
        City("A", 0, 0),
        City("B", 2, 4),
        City("C", 5, 2),
        City("D", 6, 6),
        City("E", 1, 7),
    ]
    dist = build_distance_matrix(cities)

    # ── Run brute-force reference ─────────────────────────────────────────────
    bf_path, bf_cost = brute_force_tsp(dist)
    print(f"\n{SEP}")
    print("  Brute-force reference")
    print(SEP)
    print(f"  Tour : {' → '.join(cities[i].name for i in bf_path)}")
    print(f"  Cost : {bf_cost:.4f}")

    # ── Run A* with null heuristic (UCS) ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  A* with null heuristic (UCS)")
    print(SEP)
    result_null = astar_tsp(dist, heuristic=null_heuristic)
    result_null.print_summary(cities)

    assert abs(result_null.cost - bf_cost) < 1e-6, (
        f"UCS cost {result_null.cost:.4f} != brute-force {bf_cost:.4f}"
    )
    print("  Assertion passed: cost matches brute-force")

    # ── Run A* with min-edge heuristic ────────────────────────────────────────
    print(f"\n{SEP}")
    print("  A* with min-edge heuristic")
    print(SEP)
    result_me = astar_tsp(dist, heuristic=min_edge_heuristic)
    result_me.print_summary(cities)

    assert abs(result_me.cost - bf_cost) < 1e-6, (
        f"Min-edge cost {result_me.cost:.4f} != brute-force {bf_cost:.4f}"
    )
    print("  Assertion passed: cost matches brute-force")

    # ── Compare node counts ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Heuristic comparison (nodes expanded)")
    print(SEP)
    print(f"  Null (UCS)  : {result_null.nodes_expanded} nodes")
    print(f"  Min-edge    : {result_me.nodes_expanded} nodes")
    improvement = result_null.nodes_expanded - result_me.nodes_expanded
    print(f"  Improvement : {improvement} fewer nodes with min-edge")
    print(f"\n{SEP}")
    print("  astar.py smoke test passed — ready for Phase 3 heuristics.")
    print(SEP)