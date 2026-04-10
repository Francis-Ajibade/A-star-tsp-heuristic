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

# ── TSPLIB loader ─────────────────────────────────────────────────────────────

def load_tsplib(filepath: str) -> list[City]:
    """
    Load a dataset from TSPLIB95 .tsp format.

    Supports NODE_COORD_SECTION with 2D coordinates.
    Used for benchmark datasets like wi29 (Western Sahara, 29 cities).

    File format example:
        NAME: wi29
        TYPE: TSP
        DIMENSION: 29
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
        1  20833.3333  17100.0000
        2  20900.0000  17066.6667
        ...
        EOF

    Args:
        filepath: path to the .tsp file

    Returns:
        list of City objects with coordinates scaled to a 0–100 range
        so they work naturally with the existing distance matrix code
    """
    cities  = []
    reading = False

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading = True
                continue
            if line in ("EOF", ""):
                if reading:
                    break
                continue
            if reading:
                parts = line.split()
                if len(parts) >= 3:
                    idx = parts[0]
                    x   = float(parts[1])
                    y   = float(parts[2])
                    cities.append(City(f"C{idx}", x, y))

    if not cities:
        raise ValueError(f"No cities found in {filepath}. "
                         f"Check the file has a NODE_COORD_SECTION.")

    # ── Normalise to 0–100 range ──────────────────────────────────────────────
    # Raw TSPLIB coordinates are often in the thousands (e.g. 20833.3333).
    # Normalising keeps distance values comparable to the random instances
    # used in earlier experiments without changing tour structure.
    xs    = [c.x for c in cities]
    ys    = [c.y for c in cities]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span  = max(max_x - min_x, max_y - min_y)

    if span == 0:
        span = 1.0

    normed = []
    for c in cities:
        nx = 100.0 * (c.x - min_x) / span
        ny = 100.0 * (c.y - min_y) / span
        normed.append(City(c.name, round(nx, 4), round(ny, 4)))

    return normed


def download_wi29(save_dir: str = "data") -> str:
    """
    Download the wi29 TSPLIB benchmark dataset (Western Sahara, 29 cities).
    Saves to save_dir/wi29.tsp and returns the filepath.

    If the file already exists it is not re-downloaded.
    """
    import urllib.request
    import gzip
    import shutil

    os.makedirs(save_dir, exist_ok=True)
    tsp_path = os.path.join(save_dir, "wi29.tsp")

    if os.path.exists(tsp_path):
        print(f"  wi29.tsp already exists at {tsp_path}")
        return tsp_path

    gz_path = tsp_path + ".gz"
    url     = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/wi29.tsp.gz"

    print(f"  Downloading wi29 from TSPLIB...")
    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, 'rb') as f_in:
            with open(tsp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"  Saved to {tsp_path}")
    except Exception as e:
        # Fallback: write the wi29 coordinates directly
        print(f"  Download failed ({e}), using embedded wi29 coordinates.")
        _write_wi29_embedded(tsp_path)

    return tsp_path


def _write_wi29_embedded(filepath: str) -> None:
    """
    Write wi29 coordinates directly to a file as a fallback
    if the TSPLIB server is unreachable.
    These are the official wi29 coordinates from TSPLIB95.
    """
    content = """NAME: wi29
TYPE: TSP
COMMENT: 29 Cities in Western Sahara (Gr, 1977)
DIMENSION: 29
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 20833.3333 17100.0000
2 20900.0000 17066.6667
3 21300.0000 13016.6667
4 21600.0000 14150.0000
5 21600.0000 14966.6667
6 21600.0000 16500.0000
7 22183.3333 13133.3333
8 22583.3333 14300.0000
9 22683.3333 12716.6667
10 23616.6667 15866.6667
11 23700.0000 15933.3333
12 23883.3333 14533.3333
13 24166.6667 13250.0000
14 25149.1667 12365.8333
15 26133.3333 14500.0000
16 26150.0000 10550.0000
17 26283.3333 12766.6667
18 26433.3333 13433.3333
19 26550.0000 13850.0000
20 26733.3333 11683.3333
21 27026.6667 13051.6667
22 27096.6667 13415.0000
23 27153.3333 13203.3333
24 27166.6667 9833.3333
25 27233.3333 10450.0000
26 27233.3333 11783.3333
27 27250.0000 11899.9667
28 27266.6667 10383.3333
29 27433.3333 12316.6667
EOF
"""
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  Embedded wi29 written to {filepath}")


# ── Smoke test for TSPLIB loader ──────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("\nTesting TSPLIB loader with embedded wi29...\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tsp_path = download_wi29(save_dir=os.path.join(ROOT_DIR, 'data'))
        cities   = load_tsplib(tsp_path)
        dist     = build_distance_matrix(cities)

        print(f"  Cities loaded : {len(cities)}")
        print(f"  First city    : {cities[0]}")
        print(f"  Last city     : {cities[-1]}")
        print(f"  X range       : {min(c.x for c in cities):.2f} – {max(c.x for c in cities):.2f}")
        print(f"  Y range       : {min(c.y for c in cities):.2f} – {max(c.y for c in cities):.2f}")
        print(f"  Matrix size   : {len(dist)}×{len(dist)}")
        print(f"  dist[0][1]    : {dist[0][1]:.4f}")

        assert len(cities) == 29, f"Expected 29 cities, got {len(cities)}"
        assert all(0.0 <= c.x <= 100.0 for c in cities), "X out of range"
        assert all(0.0 <= c.y <= 100.0 for c in cities), "Y out of range"
        print("\n  PASS — tsp.py TSPLIB loader working correctly.")