import math
import json
import csv
import os


# ── Data structures ───────────────────────────────────────────────────────────

class City:
    """Represents a single city with a name and (x, y) coordinates."""

    def __init__(self, name: str, x: float, y: float):
        self.name = name
        self.x = x
        self.y = y

    def __repr__(self):
        return f"City({self.name!r}, x={self.x}, y={self.y})"


# ── Distance functions ────────────────────────────────────────────────────────

def euclidean(c1: City, c2: City) -> float:
    """Straight-line distance between two cities."""
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


def manhattan(c1: City, c2: City) -> float:
    """Manhattan (grid) distance — useful for city-block layouts."""
    return abs(c1.x - c2.x) + abs(c1.y - c2.y)


# ── Distance matrix ───────────────────────────────────────────────────────────

def build_distance_matrix(cities: list[City], metric=euclidean) -> list[list[float]]:
    """
    Build an n×n distance matrix for the given list of cities.

    dist[i][j] is the cost of travelling directly from city i to city j.
    The diagonal (dist[i][i]) is always 0.

    Args:
        cities: ordered list of City objects
        metric:  distance function, defaults to euclidean

    Returns:
        n×n list of floats
    """
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = metric(cities[i], cities[j])

    return dist


def print_distance_matrix(cities: list[City], dist: list[list[float]]) -> None:
    """Pretty-print the distance matrix with city name headers."""
    names = [c.name for c in cities]
    col_w = 10

    header = " " * col_w + "".join(n.rjust(col_w) for n in names)
    print(header)
    print("-" * len(header))

    for i, row_name in enumerate(names):
        row = row_name.rjust(col_w)
        for j in range(len(cities)):
            row += f"{dist[i][j]:>{col_w}.2f}"
        print(row)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_from_json(filepath: str) -> list[City]:
    """
    Load cities from a JSON file.

    Supports two formats:

      Array of [x, y] pairs (cities get auto-named A, B, C...):
        [[0, 0], [2, 4], [5, 2]]

      Array of objects with name + coordinates:
        [{"name": "London", "x": 0.0, "y": 51.5}, ...]
    """
    with open(filepath) as f:
        data = json.load(f)

    cities = []
    for i, item in enumerate(data):
        if isinstance(item, (list, tuple)):
            name = chr(ord("A") + i) if i < 26 else f"C{i}"
            cities.append(City(name, float(item[0]), float(item[1])))
        elif isinstance(item, dict):
            cities.append(City(item["name"], float(item["x"]), float(item["y"])))
        else:
            raise ValueError(f"Unrecognised city format at index {i}: {item!r}")

    return cities


def load_from_csv(filepath: str) -> list[City]:
    """
    Load cities from a CSV file with columns: name, x, y

    Example:
        name,x,y
        London,0.0,51.5
        Paris,2.35,48.85
    """
    cities = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append(City(row["name"], float(row["x"]), float(row["y"])))
    return cities


def load_cities(filepath: str, metric=euclidean):
    """
    Convenience loader: detects file type by extension, returns
    (cities, distance_matrix) ready for use in A* or baseline solver.

    Args:
        filepath: path to .json or .csv file
        metric:   distance function to use when building the matrix

    Returns:
        tuple (list[City], list[list[float]])
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".json":
        cities = load_from_json(filepath)
    elif ext == ".csv":
        cities = load_from_csv(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext!r}. Use .json or .csv")

    dist = build_distance_matrix(cities, metric=metric)
    return cities, dist


# ── Programmatic generators (no file needed) ──────────────────────────────────

def generate_random_cities(n: int, seed: int = 42, bound: float = 100.0) -> list[City]:
    """
    Generate n cities at random (x, y) positions in [0, bound].
    Useful for experiments without needing a data file.
    """
    import random
    rng = random.Random(seed)
    return [
        City(chr(ord("A") + i) if i < 26 else f"C{i}",
             round(rng.uniform(0, bound), 2),
             round(rng.uniform(0, bound), 2))
        for i in range(n)
    ]


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Inline 5-city example — no file needed
    cities = [
        City("A", 0, 0),
        City("B", 2, 4),
        City("C", 5, 2),
        City("D", 6, 6),
        City("E", 1, 7),
    ]

    dist = build_distance_matrix(cities)
    print("Distance matrix (euclidean):\n")
    print_distance_matrix(cities, dist)

    print(f"\nDist A→C : {dist[0][2]:.4f}")
    print(f"Dist C→A : {dist[2][0]:.4f}  (symmetric)")

    # Random generation example
    print("\nRandom 6-city instance:")
    rand_cities = generate_random_cities(6, seed=7)
    for c in rand_cities:
        print(f"  {c}")