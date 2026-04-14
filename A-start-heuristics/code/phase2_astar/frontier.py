"""
frontier.py — Explicit Frontier and VisitedSet classes for A* memory analysis.

Phase 2 A* Core — A* Search for TSP
Repository: https://github.com/Francis-Ajibade/A-star-tsp-heuristic

Provides:
    Frontier         — min-heap priority queue with lazy deletion and peak tracking
    VisitedSet       — best-g-cost table with staleness detection and update counting
    astar_with_frontier() — A* using these classes, returning extended stats for
                            Phase 4 memory and efficiency experiments

Relationship to astar.py:
    astar_tsp() in astar.py implements the same algorithm with an inline heap
    and dict for maximum clarity. astar_with_frontier() wraps identical logic
    in Frontier and VisitedSet objects so Phase 4 can inspect frontier peak
    size, visited set memory, and update counts without instrumenting astar.py.
    Both functions must return the same optimal cost for any given instance.
"""

from __future__ import annotations
import heapq
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))

from state import TSPState, make_start_state


# ── Frontier ──────────────────────────────────────────────────────────────────

class Frontier:
    """
    Min-heap priority queue for A* search ordered by f(n) = g(n) + h(n).

    Design decisions:
        Lazy deletion: when a better path to a state is found, the improved
        entry is pushed without removing the stale one. Stale entries are
        detected and discarded in pop() using the VisitedSet. This avoids
        the O(n) cost of heap search-and-remove and keeps push() at O(log n).

        Tie-breaking: equal f-values are broken by insertion counter so that
        earlier pushes are popped first (FIFO within a priority level). This
        avoids direct TSPState comparison, which is unreliable for float
        f-values, and keeps search behaviour deterministic.

        Peak tracking: self.peak_size records the maximum number of live
        (non-stale) entries seen, reported in Phase 4 memory experiments.

    Attributes:
        peak_size (int): Maximum live entry count observed during the search.
    """

    def __init__(self) -> None:
        self._heap    : list[tuple] = []   # (f, counter, state)
        self._counter : int         = 0    # monotonic insertion index
        self._size    : int         = 0    # estimated live (non-stale) entries
        self.peak_size: int         = 0    # maximum live count seen

    def push(self, state: TSPState, f: float) -> None:
        """
        Insert a state with priority f into the frontier.

        Duplicate states are permitted — the VisitedSet identifies stale
        entries on pop. The live size estimate and peak are updated here.

        Args:
            state: the TSPState to enqueue.
            f:     f(n) = g(n) + h(n) priority value.
        """
        heapq.heappush(self._heap, (f, self._counter, state))
        self._counter += 1
        self._size    += 1
        if self._size > self.peak_size:
            self.peak_size = self._size

    def pop(self, visited: VisitedSet) -> TSPState | None:
        """
        Remove and return the lowest-f state that is not stale.

        An entry is stale when the VisitedSet holds a strictly cheaper g_cost
        for the same (current_city, visited_frozenset) key — meaning a better
        path was discovered and pushed after this entry was enqueued.

        Args:
            visited: the VisitedSet used by the current search run.

        Returns:
            The lowest-f non-stale TSPState, or None if the heap is empty.
        """
        while self._heap:
            f, _, state = heapq.heappop(self._heap)
            self._size -= 1

            if not visited.is_stale(state):
                return state
            # Stale entry — discard and continue to next

        return None   # heap exhausted

    def peek_f(self) -> float | None:
        """
        Return the lowest f(n) currently in the heap without removing it.

        Returns None if the heap is empty. Note: the peeked entry may be stale.
        """
        return self._heap[0][0] if self._heap else None

    def is_empty(self) -> bool:
        """Return True when the heap contains no entries (including stale)."""
        return len(self._heap) == 0

    def live_size(self) -> int:
        """
        Approximate number of non-stale entries in the heap.

        Decremented on each push that replaces a previous entry and on each
        pop. Because stale entries are only identified lazily, this is an
        estimate rather than an exact count.
        """
        return self._size

    def __len__(self) -> int:
        """Total heap entries including stale ones — use for raw memory sizing."""
        return len(self._heap)

    def __repr__(self) -> str:
        return (
            f"Frontier(heap_size={len(self._heap)}, "
            f"live~={self._size}, peak={self.peak_size})"
        )


# ── Visited set ───────────────────────────────────────────────────────────────

class VisitedSet:
    """
    Tracks the cheapest g_cost seen for every (current_city, visited) pair.

    Two responsibilities:
        Record keeping:   store and update the best g_cost per state identity.
        Staleness check:  tell Frontier.pop() whether a popped entry is outdated.

    Why not a simple closed list?
        A closed list works correctly when the heuristic is consistent — a
        consistent heuristic guarantees that the first expansion of a state
        always uses the optimal g_cost, so it never needs to be revisited.
        When using heuristics that are admissible but not proven consistent,
        the best-g approach is safer: it re-expands a state only if a strictly
        cheaper path is found, and skips all duplicates otherwise.

    Attributes:
        updates (int): Number of times a state was reached via a cheaper path
                       than previously recorded. Useful for diagnosing how
                       often the heuristic fails to be consistent.
    """

    def __init__(self) -> None:
        self._best_g: dict[tuple, float] = {}
        self.updates: int = 0

    def should_visit(self, state: TSPState) -> bool:
        """
        Return True if this state should be pushed onto the frontier.

        Returns True and updates the stored g_cost when:
            - This (current_city, visited) pair has never been seen before, OR
            - The new g_cost is strictly cheaper than the previously recorded best.

        Returns False (and does not update) when the new g_cost is equal to
        or worse than the known best, meaning this path cannot improve on what
        is already queued.

        Args:
            state: the candidate state to evaluate.

        Returns:
            True if the state should be expanded; False if it should be pruned.
        """
        key  = self._key(state)
        best = self._best_g.get(key, float("inf"))

        if state.g_cost < best - 1e-9:
            if key in self._best_g:
                self.updates += 1   # a cheaper path superseded a previous one
            self._best_g[key] = state.g_cost
            return True

        return False

    def is_stale(self, state: TSPState) -> bool:
        """
        Return True if the recorded best g_cost for this state is cheaper
        than state.g_cost, meaning this heap entry has been superseded.

        Used by Frontier.pop() to filter out outdated entries without
        searching or modifying the heap.

        Args:
            state: the state popped from the heap.

        Returns:
            True if the entry should be discarded; False if it should be expanded.
        """
        key  = self._key(state)
        best = self._best_g.get(key, float("inf"))
        return state.g_cost > best + 1e-9

    def best_cost_for(self, state: TSPState) -> float | None:
        """
        Return the best g_cost recorded for this state's identity, or None
        if this (current_city, visited) pair has not been seen.
        """
        return self._best_g.get(self._key(state))

    def states_seen(self) -> int:
        """Total unique (current_city, visited) pairs recorded."""
        return len(self._best_g)

    def memory_bytes(self) -> int:
        """
        Rough estimate of the VisitedSet memory footprint in bytes.

        Approximates each entry as 200 bytes (frozenset key + int city index +
        float value + dict overhead). This is a heuristic estimate for Phase 4
        reporting, not a precise measurement.
        """
        return len(self._best_g) * 200

    @staticmethod
    def _key(state: TSPState) -> tuple:
        return (state.current_city, state.visited)

    def __repr__(self) -> str:
        return (
            f"VisitedSet(unique_states={len(self._best_g)}, "
            f"updates={self.updates})"
        )


# ── A* using Frontier + VisitedSet ────────────────────────────────────────────

def astar_with_frontier(
    dist      : list[list[float]],
    heuristic : callable,
    start     : int = 0,
) -> dict | None:
    """
    A* search using the explicit Frontier and VisitedSet classes.

    Produces the same optimal tour as astar_tsp() in astar.py but returns
    a richer result dict that includes memory and frontier statistics. Used
    by Phase 4 experiments to compare search effort across heuristics.

    Note on return type:
        Returns a dict (not AStarResult) to accommodate the extra fields
        (frontier_peak, states_in_visited, visited_updates, memory_bytes)
        that have no place in the lean AStarResult dataclass.

    Args:
        dist:      n×n distance matrix.
        heuristic: callable(state: TSPState, dist: list) -> float.
        start:     index of the start city, defaults to 0.

    Returns:
        Dict with keys: path, cost, nodes_expanded, nodes_generated,
        runtime_ms, frontier_peak, states_in_visited, visited_updates,
        memory_bytes, n_cities.
        Returns None if no tour exists (frontier exhausted).
    """
    n        = len(dist)
    frontier = Frontier()
    visited  = VisitedSet()

    start_state = make_start_state(start)
    h0          = heuristic(start_state, dist)

    visited.should_visit(start_state)     # register start before pushing
    frontier.push(start_state, f=h0)

    nodes_expanded  = 0
    nodes_generated = 1                   # start state generated at init

    t0 = time.perf_counter()

    while True:
        state = frontier.pop(visited)

        if state is None:
            return None   # frontier exhausted — no tour found

        nodes_expanded += 1

        if len(state.visited) > n:
            # Terminal state (closed tour) — visited sentinel size > n flags completion
            runtime_ms = (time.perf_counter() - t0) * 1000
            return {
                "path"              : state.path,
                "cost"              : state.g_cost,
                "nodes_expanded"    : nodes_expanded,
                "nodes_generated"   : nodes_generated,
                "runtime_ms"        : runtime_ms,
                "frontier_peak"     : frontier.peak_size,
                "states_in_visited" : visited.states_seen(),
                "visited_updates"   : visited.updates,
                "memory_bytes"      : visited.memory_bytes(),
                "n_cities"          : n,
            }

        if state.is_goal(n):
            # Push a terminal entry with the full closed-tour cost
            from state import TSPState as _TSPState
            return_cost = dist[state.current_city][state.path[0]]
            terminal    = _TSPState(
                current_city = state.path[0],
                visited      = state.visited | {n},
                g_cost       = state.g_cost + return_cost,
                path         = state.full_tour(),
            )
            if visited.should_visit(terminal):
                h = heuristic(terminal, dist)
                frontier.push(terminal, f=terminal.g_cost + h)
                nodes_generated += 1
            continue

        for successor in state.expand(dist):
            if visited.should_visit(successor):
                h = heuristic(successor, dist)
                f = successor.g_cost + h
                frontier.push(successor, f)
                nodes_generated += 1


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tsp      import City, build_distance_matrix
    from baseline import brute_force_tsp

    SEP = "=" * 55

    def null_heuristic(state: TSPState, dist: list) -> float:
        return 0.0

    cities = [
        City("A", 0, 0),
        City("B", 2, 4),
        City("C", 5, 2),
        City("D", 6, 6),
        City("E", 1, 7),
    ]
    dist             = build_distance_matrix(cities)
    bf_path, bf_cost = brute_force_tsp(dist)

    print(f"\n{SEP}")
    print("  Frontier + VisitedSet — smoke test")
    print(SEP)

    # Frontier unit check
    print("\n-- Frontier ordering check --")
    fr = Frontier()
    vs = VisitedSet()

    s_low  = make_start_state(0)
    s_high = make_start_state(0)

    fr.push(s_high, f=9.0)
    fr.push(s_low,  f=2.0)
    fr.push(s_high, f=5.0)

    vs.should_visit(s_low)   # register so neither entry appears stale

    popped = fr.pop(vs)
    print(f"  Heap size after 3 pushes : {len(fr) + 1}")  # +1 for the popped entry
    print(f"  First pop (expected f=2) : city={popped.current_city}, g={popped.g_cost}")
    print(f"  Frontier repr            : {fr}")
    print(f"  VisitedSet repr          : {vs}")

    # Full A* run via astar_with_frontier
    print("\n-- Full A* run (null heuristic) --")
    result = astar_with_frontier(dist, null_heuristic)

    tour_str = " → ".join(cities[i].name for i in result["path"])
    print(f"  Tour               : {tour_str}")
    print(f"  Cost               : {result['cost']:.4f}")
    print(f"  Nodes expanded     : {result['nodes_expanded']}")
    print(f"  Nodes generated    : {result['nodes_generated']}")
    print(f"  Frontier peak      : {result['frontier_peak']}")
    print(f"  Unique states      : {result['states_in_visited']}")
    print(f"  Visited updates    : {result['visited_updates']}")
    print(f"  Memory (estimate)  : {result['memory_bytes']} bytes")
    print(f"  Runtime            : {result['runtime_ms']:.3f} ms")

    assert abs(result["cost"] - bf_cost) < 1e-6, (
        f"Cost mismatch: A*={result['cost']:.4f}  BF={bf_cost:.4f}"
    )
    print(f"\n  Assertion passed: A* cost == brute-force ({bf_cost:.4f})")
    print(SEP)