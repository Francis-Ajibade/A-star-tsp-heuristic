from __future__ import annotations
import heapq
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))

from state import TSPState, make_start_state


# ── Frontier ──────────────────────────────────────────────────────────────────

class Frontier:
    """
    Min-heap priority queue for A* search, ordered by f(n) = g(n) + h(n).

    Design decisions:
      - Uses lazy deletion instead of decrease-key. When a better path
        to a state is found, the new entry is pushed and the old one is
        left in the heap. Stale entries are detected and skipped on pop.
      - Ties in f(n) broken by insertion counter (FIFO) — avoids comparing
        TSPState objects and keeps behaviour deterministic.
      - Tracks peak size for memory reporting in Phase 4 experiments.
    """

    def __init__(self):
        self._heap    : list[tuple]  = []   # (f, counter, state)
        self._counter : int          = 0    # insertion order tie-breaker
        self._size    : int          = 0    # live entries (not counting stale)
        self.peak_size: int          = 0    # max live size seen — for reporting

    # ── Core operations ───────────────────────────────────────────────────────

    def push(self, state: TSPState, f: float) -> None:
        """
        Insert a state with priority f into the frontier.
        Duplicate states are allowed — stale ones are filtered on pop().
        """
        heapq.heappush(self._heap, (f, self._counter, state))
        self._counter += 1
        self._size    += 1
        if self._size > self.peak_size:
            self.peak_size = self._size

    def pop(self, visited: "VisitedSet") -> TSPState | None:
        """
        Remove and return the lowest-f state that is not stale.

        A state is stale if the VisitedSet already holds a cheaper or equal
        g_cost for the same (city, visited) key — meaning a better path was
        found and pushed after this entry.

        Returns None when the frontier is empty.
        """
        while self._heap:
            f, _, state = heapq.heappop(self._heap)
            self._size -= 1

            if not visited.is_stale(state):
                return state
            # else: stale — discard silently and keep popping

        return None   # frontier exhausted

    def peek_f(self) -> float | None:
        """Return the lowest f(n) on the frontier without removing it."""
        return self._heap[0][0] if self._heap else None

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        """Number of entries in the heap (includes stale — use for memory only)."""
        return len(self._heap)

    def live_size(self) -> int:
        """Approximate number of non-stale entries."""
        return self._size

    # ── Debug ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Frontier(heap_size={len(self._heap)}, "
            f"live={self._size}, peak={self.peak_size})"
        )


# ── Visited set ───────────────────────────────────────────────────────────────

class VisitedSet:
    """
    Tracks the best g_cost seen for every (current_city, visited) state pair.

    Two roles:
      1. Record keeping  — store the cheapest g(n) reaching each unique state.
      2. Staleness check — tell the Frontier whether a popped entry is outdated.

    Why not a simple 'closed list'?
    Classic A* with a consistent heuristic can use a closed list and never
    re-expand a state. TSP heuristics are admissible but not always consistent
    across all implementations, so we use the safer best_g approach: re-expand
    only if a strictly cheaper path is found.
    """

    def __init__(self):
        self._best_g : dict[tuple, float] = {}
        self.updates : int = 0    # how many times a cheaper path was found

    # ── Core operations ───────────────────────────────────────────────────────

    def should_visit(self, state: TSPState) -> bool:
        """
        Return True if this state should be expanded.

        True when:
          - We have never seen this (city, visited) pair before, OR
          - The new g_cost is strictly cheaper than the best known.

        Side effect: updates the stored best_g if returning True.
        """
        key    = self._key(state)
        best   = self._best_g.get(key, float('inf'))

        if state.g_cost < best - 1e-9:
            if key in self._best_g:
                self.updates += 1
            self._best_g[key] = state.g_cost
            return True

        return False

    def is_stale(self, state: TSPState) -> bool:
        """
        Return True if a cheaper path to this state is already known.
        Used by Frontier.pop() to discard outdated heap entries.
        """
        key  = self._key(state)
        best = self._best_g.get(key, float('inf'))
        return state.g_cost > best + 1e-9

    def best_cost_for(self, state: TSPState) -> float | None:
        """Return the best g_cost seen for this state, or None if unseen."""
        return self._best_g.get(self._key(state))

    # ── Stats ─────────────────────────────────────────────────────────────────

    def states_seen(self) -> int:
        """Total unique (city, visited) pairs encountered."""
        return len(self._best_g)

    def memory_bytes(self) -> int:
        """Rough memory footprint of the best_g dictionary."""
        # Each entry: frozenset + int key + float value ≈ 200 bytes average
        return len(self._best_g) * 200

    # ── Internal ──────────────────────────────────────────────────────────────

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
    A* implementation that uses the explicit Frontier and VisitedSet classes.

    Identical in behaviour to astar_tsp() in astar.py but exposes the
    internal data structures so Phase 4 can inspect memory usage in detail.

    Returns a results dict including frontier and visited stats, or None
    if no solution is found.
    """
    n        = len(dist)
    frontier = Frontier()
    visited  = VisitedSet()

    start_state = make_start_state(start)
    h0          = heuristic(start_state, dist)

    visited.should_visit(start_state)          # register start
    frontier.push(start_state, f=h0)

    nodes_expanded  = 0
    nodes_generated = 1

    import time
    t0 = time.perf_counter()

    while True:
        state = frontier.pop(visited)

        if state is None:
            return None   # no solution

        nodes_expanded += 1

        # ── Goal ─────────────────────────────────────────────────────────────
        if state.is_goal(n):
            runtime_ms = (time.perf_counter() - t0) * 1000
            return {
                "path"            : state.full_tour(),
                "cost"            : state.final_cost(dist),
                "nodes_expanded"  : nodes_expanded,
                "nodes_generated" : nodes_generated,
                "runtime_ms"      : runtime_ms,
                "frontier_peak"   : frontier.peak_size,
                "states_in_visited": visited.states_seen(),
                "visited_updates" : visited.updates,
                "memory_bytes"    : visited.memory_bytes(),
                "n_cities"        : n,
            }

        # ── Expand ───────────────────────────────────────────────────────────
        for successor in state.expand(dist):
            if visited.should_visit(successor):
                h = heuristic(successor, dist)
                f = successor.g_cost + h
                frontier.push(successor, f)
                nodes_generated += 1


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_foundation'))
    from tsp import City, build_distance_matrix
    from baseline import brute_force_tsp

    SEP = "=" * 55

    def null_heuristic(state, dist):
        return 0.0

    cities = [
        City("A", 0, 0),
        City("B", 2, 4),
        City("C", 5, 2),
        City("D", 6, 6),
        City("E", 1, 7),
    ]
    dist = build_distance_matrix(cities)
    bf_path, bf_cost = brute_force_tsp(dist)

    print(f"\n{SEP}")
    print("  Frontier + VisitedSet — smoke test")
    print(SEP)

    # ── Test Frontier in isolation ────────────────────────────────────────────
    print("\n-- Frontier unit check --")
    f  = Frontier()
    vs = VisitedSet()

    s0 = make_start_state(0)
    s1 = make_start_state(0)   # same state, different object

    f.push(s0, 5.0)
    f.push(s1, 3.0)
    vs.should_visit(s0)

    print(f"  Heap size after 2 pushes : {len(f)}")
    popped = f.pop(vs)
    print(f"  First pop f=3.0, city    : {popped.current_city}")
    print(f"  Frontier repr            : {f}")
    print(f"  VisitedSet repr          : {vs}")

    # ── Full A* via astar_with_frontier ───────────────────────────────────────
    print("\n-- Full A* run --")
    result = astar_with_frontier(dist, null_heuristic)

    tour_str = " → ".join(cities[i].name for i in result["path"])
    print(f"  Tour             : {tour_str}")
    print(f"  Cost             : {result['cost']:.4f}")
    print(f"  Nodes expanded   : {result['nodes_expanded']}")
    print(f"  Nodes generated  : {result['nodes_generated']}")
    print(f"  Frontier peak    : {result['frontier_peak']}")
    print(f"  Unique states    : {result['states_in_visited']}")
    print(f"  Visited updates  : {result['visited_updates']}")
    print(f"  Memory (approx)  : {result['memory_bytes']} bytes")
    print(f"  Runtime          : {result['runtime_ms']:.3f} ms")

    assert abs(result["cost"] - bf_cost) < 1e-6, (
        f"Cost mismatch: A*={result['cost']:.4f} BF={bf_cost:.4f}"
    )
    print(f"\n  Assertion passed: A* cost == brute-force ({bf_cost:.4f})")
    print(f"{SEP}")