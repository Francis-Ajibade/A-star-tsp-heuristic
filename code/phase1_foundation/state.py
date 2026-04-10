from __future__ import annotations


# ── TSP State ─────────────────────────────────────────────────────────────────

class TSPState:
    """
    Represents a single node in the A* search tree for TSP.

    A state captures:
      - which city the agent is currently at
      - which cities have already been visited
      - the exact cost g(n) to reach this state from the start
      - the full path taken so far (for reconstructing the solution)

    Two states are considered identical (for duplicate detection) if they
    share the same current_city AND the same visited set — the path and
    cost are irrelevant for equality purposes.
    """

    def __init__(
        self,
        current_city: int,
        visited: frozenset[int],
        g_cost: float,
        path: list[int],
    ):
        self.current_city = current_city
        self.visited = visited        # frozenset makes this hashable
        self.g_cost = g_cost          # exact cost from start → here
        self.path = path              # ordered list of city indices visited

    # ── Goal test ─────────────────────────────────────────────────────────────

    def is_goal(self, n: int) -> bool:
        """
        Goal = all n cities visited AND we can return to start (city 0).
        A* will add the return edge cost when this returns True.
        """
        return len(self.visited) == n

    # ── Successor generation ──────────────────────────────────────────────────

    def expand(self, dist: list[list[float]]) -> list[TSPState]:
        """
        Generate all valid next states from this one.

        A successor exists for every city not yet in self.visited.
        The new g_cost includes the edge from current_city → next_city.

        Args:
            dist: precomputed n×n distance matrix from tsp.py

        Returns:
            list of TSPState, one per unvisited city
        """
        successors = []
        n = len(dist)

        for next_city in range(n):
            if next_city not in self.visited:
                edge_cost = dist[self.current_city][next_city]
                successors.append(
                    TSPState(
                        current_city=next_city,
                        visited=self.visited | {next_city},   # frozenset union
                        g_cost=self.g_cost + edge_cost,
                        path=self.path + [next_city],
                    )
                )

        return successors

    def final_cost(self, dist: list[list[float]]) -> float:
        """
        Total tour cost once the goal is reached:
        g(n) + the return edge back to the start city (index 0).
        """
        return self.g_cost + dist[self.current_city][self.path[0]]

    def full_tour(self) -> list[int]:
        """Return the complete tour including the return to start."""
        return self.path + [self.path[0]]

    # ── Comparison (required by heapq) ────────────────────────────────────────

    def __lt__(self, other: TSPState) -> bool:
        """
        heapq needs to compare states when f(n) values tie.
        Breaking ties by g_cost prefers deeper (more complete) paths.
        """
        return self.g_cost > other.g_cost  # prefer higher g on tie = deeper path

    # ── Hashing + equality (required for visited-set deduplication) ───────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TSPState):
            return NotImplemented
        return (
            self.current_city == other.current_city
            and self.visited == other.visited
        )

    def __hash__(self) -> int:
        """
        Only current_city + visited matter for identity.
        Two paths reaching the same (city, visited) are duplicates —
        A* keeps whichever has the lower f(n).
        """
        return hash((self.current_city, self.visited))

    # ── Debug display ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        visited_str = "{" + ", ".join(str(c) for c in sorted(self.visited)) + "}"
        return (
            f"TSPState(city={self.current_city}, "
            f"visited={visited_str}, "
            f"g={self.g_cost:.3f})"
        )


# ── Initial state factory ─────────────────────────────────────────────────────

def make_start_state(start_city: int = 0) -> TSPState:
    """
    Create the initial A* state.
    The agent starts at start_city with only that city marked visited.
    """
    return TSPState(
        current_city=start_city,
        visited=frozenset({start_city}),
        g_cost=0.0,
        path=[start_city],
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal 4-city distance matrix (hand-crafted, easy to verify)
    dist = [
        [0.0, 1.0, 4.0, 3.0],
        [1.0, 0.0, 2.0, 5.0],
        [4.0, 2.0, 0.0, 1.0],
        [3.0, 5.0, 1.0, 0.0],
    ]

    start = make_start_state(start_city=0)
    print("Start state:", start)
    print("Is goal?   ", start.is_goal(n=4))

    successors = start.expand(dist)
    print(f"\nSuccessors from city 0 ({len(successors)} states):")
    for s in successors:
        print(" ", s)

    # Expand one level deeper
    s1 = successors[0]
    print(f"\nExpanding {s1}:")
    for s in s1.expand(dist):
        print(" ", s)

    # Simulate reaching the goal
    goal = TSPState(
        current_city=3,
        visited=frozenset({0, 1, 2, 3}),
        g_cost=4.0,
        path=[0, 1, 2, 3],
    )
    print(f"\nGoal state:   {goal}")
    print(f"Is goal?      {goal.is_goal(n=4)}")
    print(f"Return cost:  {dist[3][0]:.1f}")
    print(f"Total tour:   {goal.full_tour()}")
    print(f"Final cost:   {goal.final_cost(dist):.3f}")

    # Verify hashing works correctly
    duplicate = TSPState(
        current_city=3,
        visited=frozenset({0, 1, 2, 3}),
        g_cost=99.0,          # different cost, same identity
        path=[0, 3, 2, 1, 3], # different path
    )
    print(f"\nDuplicate detection:")
    print(f"  goal == duplicate: {goal == duplicate}")   # True
    print(f"  same hash:         {hash(goal) == hash(duplicate)}")  # True