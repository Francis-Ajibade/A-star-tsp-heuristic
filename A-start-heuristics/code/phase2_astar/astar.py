"""
astar.py — Core A* search engine for TSP with pluggable heuristics.

Phase 2 A* Core — A* Search for TSP
Repository: https://github.com/Francis-Ajibade/A-star-tsp-heuristic

Provides:
    AStarResult        — structured result object returned by the search
    astar_tsp()        — main A* search function with lazy-deletion heap
    null_heuristic()   — h(n) = 0, reduces A* to Uniform Cost Search (baseline)
    min_edge_heuristic() — h(n) = remaining_edges * global_min_edge (weak lower bound)

Design overview:
    A* maintains a min-heap ordered by f(n) = g(n) + h(n). A lazy-deletion
    strategy is used: when a better path to a state is found, the new entry
    is pushed without removing the old one. Stale entries are detected and
    discarded when popped. This avoids expensive decrease-key operations.

    For the search to return an optimal tour, h(n) must be admissible
    (never overestimates) and ideally consistent (monotone). Both heuristics
    in this module satisfy admissibility; consistency is verified in
    test_astar.py and the Phase 3 admissibility notebook.
"""

from __future__ import annotations
import heapq
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))

from tsp   import City, build_distance_matrix
from state import TSPState, make_start_state


# ── Result container ──────────────────────────────────────────────────────────

class AStarResult:
    """
    Structured container for the output of a single A* run.

    Returned by astar_tsp() and consumed by the Phase 4 experiment logger.
    All fields are populated before returning — there are no lazy attributes.

    Attributes:
        path            (list[int]): Ordered city indices; starts and ends at
                                     the start city (length = n + 1).
        cost            (float):     Total optimal tour distance.
        nodes_expanded  (int):       States popped and fully processed.
        nodes_generated (int):       States pushed onto the heap (includes
                                     duplicates and states later pruned).
        runtime_ms      (float):     Wall-clock time in milliseconds.
        n_cities        (int):       Number of cities in the instance.
    """

    def __init__(
        self,
        path           : list[int],
        cost           : float,
        nodes_expanded : int,
        nodes_generated: int,
        runtime_ms     : float,
        n_cities       : int,
    ) -> None:
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
            f"  tour      = {tour}\n"
            f"  cost      = {self.cost:.4f}\n"
            f"  expanded  = {self.nodes_expanded}\n"
            f"  generated = {self.nodes_generated}\n"
            f"  time      = {self.runtime_ms:.3f} ms\n"
            f")"
        )

    def print_summary(self, cities: list[City] | None = None) -> None:
        """
        Print a formatted one-block summary of the result.

        Args:
            cities: optional list of City objects; when provided, city names
                    are shown in the tour string instead of numeric indices.
        """
        if cities:
            tour_str = " → ".join(cities[i].name for i in self.path)
        else:
            tour_str = " → ".join(str(i) for i in self.path)

        print(f"  Tour            : {tour_str}")
        print(f"  Cost            : {self.cost:.4f}")
        print(f"  Nodes expanded  : {self.nodes_expanded}")
        print(f"  Nodes generated : {self.nodes_generated}")
        print(f"  Runtime         : {self.runtime_ms:.3f} ms")


# ── Priority queue entry ──────────────────────────────────────────────────────

class _Entry:
    """
    Wraps a TSPState for storage in the heapq min-heap.

    The heap is ordered by f(n) = g(n) + h(n). When two entries have
    equal f-values, ties are broken by insertion counter (earlier push
    wins) to keep ordering deterministic and avoid direct TSPState
    comparison, which is undefined for equal f-values.

    Attributes:
        f       (float): Priority value — g(n) + h(n).
        counter (int):   Monotonically increasing insertion index.
        state   (TSPState): The search state this entry wraps.
    """

    __slots__ = ("f", "counter", "state")

    def __init__(self, f: float, counter: int, state: TSPState) -> None:
        self.f       = f
        self.counter = counter
        self.state   = state

    def __lt__(self, other: _Entry) -> bool:
        if self.f != other.f:
            return self.f < other.f
        return self.counter < other.counter   # earlier insertion wins on tie


# ── Core A* search ────────────────────────────────────────────────────────────

def astar_tsp(
    dist      : list[list[float]],
    heuristic : callable,
    start     : int = 0,
) -> AStarResult | None:
    """
    Solve TSP optimally using A* search with a pluggable heuristic.

    Uses a lazy-deletion min-heap. When a cheaper path to an already-queued
    state is found, a new entry is pushed and the old one is left in the heap.
    On pop, stale entries (where g_cost > best known for that state) are
    discarded immediately without expanding.

    Optimality guarantee:
        The returned tour is optimal if and only if heuristic is admissible,
        i.e. heuristic(state, dist) <= true_remaining_cost(state) for all states.
        With a consistent heuristic, no state is ever re-expanded.

    Args:
        dist:      n×n distance matrix from tsp.build_distance_matrix().
        heuristic: callable(state: TSPState, dist: list) -> float.
                   Must return a non-negative admissible estimate.
        start:     Index of the start city, defaults to 0.

    Returns:
        AStarResult on success, None if the frontier is exhausted with no goal.
    """
    n               = len(dist)
    nodes_expanded  = 0
    nodes_generated = 1          # the start state is generated at initialisation
    counter         = 0

    # Initialise with the start state
    start_state = make_start_state(start)
    h0          = heuristic(start_state, dist)
    frontier    : list[_Entry] = []
    heapq.heappush(frontier, _Entry(h0, counter, start_state))

    # best_g maps (current_city, visited_frozenset) -> cheapest g_cost seen.
    # A state is pruned on pop if its g_cost exceeds the stored best.
    best_g: dict[tuple, float] = {
        (start_state.current_city, start_state.visited): 0.0
    }

    t0 = time.perf_counter()

    while frontier:
        entry = heapq.heappop(frontier)
        state = entry.state

        # Staleness check — skip if a cheaper path to this state was found later
        key = (state.current_city, state.visited)
        if state.g_cost > best_g.get(key, float("inf")) + 1e-9:
            continue

        nodes_expanded += 1

        # Goal: all cities visited — treat the return edge as one more expansion.
        #
        # Returning immediately on the first goal state popped is incorrect for
        # TSP because the total tour cost is g_cost + return_edge, not g_cost
        # alone. Two all-visited states may have different g_costs but the one
        # with the lower g_cost can have a longer return edge, producing a worse
        # total. We must let the heap order complete tours by their full cost.
        #
        # Implementation: when all cities are visited, push a synthetic terminal
        # state whose g_cost is the full tour cost (g + return edge). Terminal
        # states are identified by visited containing a sentinel value n (which
        # is never a valid city index). When such a state is popped, return.
        if len(state.visited) > n:
            # This is a terminal (closed-tour) state — return the result
            runtime_ms = (time.perf_counter() - t0) * 1000
            return AStarResult(
                path            = state.path,
                cost            = state.g_cost,
                nodes_expanded  = nodes_expanded,
                nodes_generated = nodes_generated,
                runtime_ms      = runtime_ms,
                n_cities        = n,
            )

        if state.is_goal(n):
            # All cities visited — push a terminal entry with the closed-tour cost
            return_cost   = dist[state.current_city][state.path[0]]
            terminal_g    = state.g_cost + return_cost
            terminal_path = state.full_tour()

            # Build a synthetic terminal TSPState using a sentinel visited set.
            # visited size > n flags it as terminal on the next pop.
            from state import TSPState as _TSPState
            terminal = _TSPState(
                current_city = state.path[0],         # back at start city
                visited      = state.visited | {n},   # sentinel: size = n+1
                g_cost       = terminal_g,
                path         = terminal_path,
            )
            t_key = (terminal.current_city, terminal.visited)
            if terminal_g < best_g.get(t_key, float("inf")) - 1e-9:
                best_g[t_key]    = terminal_g
                nodes_generated += 1
                counter         += 1
                heapq.heappush(frontier, _Entry(terminal_g, counter, terminal))
            continue   # do not expand this goal state further

        # Expand: generate successors and push those with improved g_cost
        for successor in state.expand(dist):
            s_key = (successor.current_city, successor.visited)

            # Prune if a cheaper or equal path to this state is already known
            if successor.g_cost >= best_g.get(s_key, float("inf")) - 1e-9:
                continue

            best_g[s_key]    = successor.g_cost
            nodes_generated += 1

            h       = heuristic(successor, dist)
            f       = successor.g_cost + h
            counter += 1
            heapq.heappush(frontier, _Entry(f, counter, successor))

    return None   # frontier exhausted, no tour found


# ── Built-in heuristics ───────────────────────────────────────────────────────

def null_heuristic(state: TSPState, dist: list[list[float]]) -> float:
    """
    h(n) = 0 for every state.

    Reduces A* to Uniform Cost Search (UCS) — the algorithm expands states
    purely by g(n) with no future-cost guidance. Always admissible since 0
    never overestimates. Used as the uninformed baseline: it finds the optimal
    tour but explores far more states than any informative heuristic.

    Args:
        state: current search state (unused).
        dist:  distance matrix (unused).

    Returns:
        0.0
    """
    return 0.0


def min_edge_heuristic(state: TSPState, dist: list[list[float]]) -> float:
    """
    h(n) = (unvisited cities + 1 return edge) * global minimum edge weight.

    Admissibility proof:
        Every remaining edge in the optimal completion must cost at least
        the minimum possible edge weight. The remaining tour requires exactly
        len(unvisited) + 1 edges. Multiplying these gives a valid lower bound
        on the true remaining cost.

    Limitations:
        - Recomputes the global minimum by scanning all n² pairs on every call:
          O(n²) per heuristic evaluation. For large n this becomes a bottleneck;
          precomputing the global minimum once before the search loop is an
          easy optimisation for production use.
        - Produces a weak lower bound — the global minimum edge is rarely
          achievable for every remaining step, so the estimate underestimates
          substantially and prunes less aggressively than the MST heuristic.

    Args:
        state: current search state.
        dist:  n×n distance matrix.

    Returns:
        Non-negative admissible lower bound on remaining tour cost.
    """
    n         = len(dist)
    unvisited = [c for c in range(n) if c not in state.visited]

    if not unvisited:
        # All cities visited — only the return edge remains
        return dist[state.current_city][state.path[0]]

    # Global minimum non-zero edge across the entire graph
    min_edge = min(
        dist[i][j]
        for i in range(n)
        for j in range(n)
        if i != j
    )

    remaining_edges = len(unvisited) + 1   # forward edges + mandatory return
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

    # Brute-force reference
    bf_path, bf_cost = brute_force_tsp(dist)
    print(f"\n{SEP}")
    print("  Brute-force reference")
    print(SEP)
    print(f"  Tour : {' → '.join(cities[i].name for i in bf_path)}")
    print(f"  Cost : {bf_cost:.4f}")

    # A* with null heuristic (UCS)
    print(f"\n{SEP}")
    print("  A* — Null heuristic (UCS)")
    print(SEP)
    result_null = astar_tsp(dist, heuristic=null_heuristic)
    result_null.print_summary(cities)
    assert abs(result_null.cost - bf_cost) < 1e-6, (
        f"Null heuristic cost {result_null.cost:.4f} != brute-force {bf_cost:.4f}"
    )
    print("  Assertion passed: cost matches brute-force")

    # A* with min-edge heuristic
    print(f"\n{SEP}")
    print("  A* — Min-edge heuristic")
    print(SEP)
    result_me = astar_tsp(dist, heuristic=min_edge_heuristic)
    result_me.print_summary(cities)
    assert abs(result_me.cost - bf_cost) < 1e-6, (
        f"Min-edge cost {result_me.cost:.4f} != brute-force {bf_cost:.4f}"
    )
    print("  Assertion passed: cost matches brute-force")

    # Node count comparison
    print(f"\n{SEP}")
    print("  Node expansion comparison")
    print(SEP)
    print(f"  Null (UCS) : {result_null.nodes_expanded} nodes expanded")
    print(f"  Min-edge   : {result_me.nodes_expanded} nodes expanded")
    saved = result_null.nodes_expanded - result_me.nodes_expanded
    print(f"  Saving     : {saved} fewer nodes with min-edge")
    print(f"\n{SEP}")
    print("  astar.py smoke test passed — ready for Phase 3 heuristics.")
    print(SEP)