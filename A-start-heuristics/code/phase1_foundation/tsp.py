"""
tsp.py — City representation, distance metrics, matrix builder, and data loaders.

Phase 1 Foundation — A* Search for TSP
Repository: https://github.com/Francis-Ajibade/A-star-tsp-heuristic

Provides:
    City                  — data class holding a city's name and (x, y) coordinates
    euclidean()           — straight-line distance between two cities
    manhattan()           — grid distance between two cities
    build_distance_matrix() — builds the n×n cost matrix used by A* and the baseline
    load_from_json()      — load cities from a JSON file
    load_from_csv()       — load cities from a CSV file
    load_cities()         — auto-detecting convenience loader
    generate_random_cities() — reproducible random instance generator
    load_tsplib()         — parse TSPLIB95 .tsp benchmark files
    download_wi29()       — fetch the wi29 Western Sahara benchmark dataset
"""

import math
import json
import csv
import os


# ── Data structures ───────────────────────────────────────────────────────────

class City:
    """
    A single city with a name and (x, y) Euclidean coordinates.

    Attributes:
        name (str):  Human-readable label, e.g. "London" or "C1".
        x    (float): Horizontal coordinate.
        y    (float): Vertical coordinate.
    """

    def __init__(self, name: str, x: float, y: float) -> None:
        self.name = name
        self.x    = x
        self.y    = y

    def __repr__(self) -> str:
        return f"City({self.name!r}, x={self.x}, y={self.y})"


# ── Distance metrics ──────────────────────────────────────────────────────────

def euclidean(c1: City, c2: City) -> float:
    """
    Straight-line (L2) distance between two cities.

    Used as the default metric throughout all experiments.
    Consistent with the EUC_2D weight type used in TSPLIB95 benchmarks.
    """
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


def manhattan(c1: City, c2: City) -> float:
    """
    Manhattan (L1) distance between two cities.

    Provided as an alternative metric for grid-based problem variants.
    Not used in the primary experiments but available via the metric parameter.
    """
    return abs(c1.x - c2.x) + abs(c1.y - c2.y)


# ── Distance matrix ───────────────────────────────────────────────────────────

def build_distance_matrix(
    cities: list[City],
    metric: callable = euclidean,
) -> list[list[float]]:
    """
    Build an n×n distance matrix for a list of cities.

    dist[i][j] holds the direct travel cost from city i to city j.
    The diagonal is always 0. The matrix is symmetric when using
    Euclidean or Manhattan distance.

    Args:
        cities: ordered list of City objects.
        metric: distance function to apply, defaults to euclidean.

    Returns:
        n×n list of floats.
    """
    n    = len(cities)
    dist = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = metric(cities[i], cities[j])

    return dist


def print_distance_matrix(
    cities: list[City],
    dist: list[list[float]],
) -> None:
    """
    Pretty-print the distance matrix with city name headers.

    Primarily a debugging aid — not used in production paths.
    """
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


# ── File loaders ──────────────────────────────────────────────────────────────

def load_from_json(filepath: str) -> list[City]:
    """
    Load cities from a JSON file.

    Supports two input formats:

        Array of [x, y] pairs — cities are auto-named A, B, C, ...:
            [[0, 0], [2, 4], [5, 2]]

        Array of objects with explicit name and coordinates:
            [{"name": "London", "x": 0.0, "y": 51.5}, ...]

    Args:
        filepath: path to the .json file.

    Returns:
        List of City objects in file order.

    Raises:
        ValueError: if an entry does not match either supported format.
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
    Load cities from a CSV file with columns: name, x, y.

    Expected file format:
        name,x,y
        London,0.0,51.5
        Paris,2.35,48.85

    Args:
        filepath: path to the .csv file.

    Returns:
        List of City objects in row order.
    """
    cities = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append(City(row["name"], float(row["x"]), float(row["y"])))
    return cities


def load_cities(
    filepath: str,
    metric: callable = euclidean,
) -> tuple[list[City], list[list[float]]]:
    """
    Convenience loader: detect file type by extension and return
    both the city list and the precomputed distance matrix.

    Supported extensions: .json, .csv

    Args:
        filepath: path to the city data file.
        metric:   distance function for the matrix, defaults to euclidean.

    Returns:
        Tuple (list[City], list[list[float]]).

    Raises:
        ValueError: if the file extension is not .json or .csv.
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


# ── Programmatic instance generator ──────────────────────────────────────────

def generate_random_cities(
    n: int,
    seed: int = 42,
    bound: float = 100.0,
) -> list[City]:
    """
    Generate n cities at uniformly random (x, y) positions in [0, bound].

    The seed parameter makes instances fully reproducible — the same seed
    always produces the same set of cities. Used in controlled experiments
    to benchmark heuristic performance across consistent problem instances.

    Args:
        n:     number of cities to generate.
        seed:  random seed for reproducibility.
        bound: coordinate range, cities placed in [0, bound] × [0, bound].

    Returns:
        List of n City objects with auto-assigned names (A, B, C, ...).
    """
    import random
    rng = random.Random(seed)
    return [
        City(
            chr(ord("A") + i) if i < 26 else f"C{i}",
            round(rng.uniform(0, bound), 2),
            round(rng.uniform(0, bound), 2),
        )
        for i in range(n)
    ]


# ── TSPLIB95 loader ───────────────────────────────────────────────────────────

def load_tsplib(filepath: str) -> list[City]:
    """
    Parse a TSPLIB95 .tsp file and return normalised City objects.

    Supports the NODE_COORD_SECTION format with EUC_2D coordinates.
    After parsing, coordinates are scaled to [0, 100] so distances are
    comparable to the random instances used in controlled experiments.
    Scaling preserves all pairwise distance ratios and tour structure.

    Example TSPLIB file structure:
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
        filepath: path to the .tsp file.

    Returns:
        List of City objects with coordinates in [0, 100].

    Raises:
        ValueError: if no cities are found (missing NODE_COORD_SECTION).
    """
    cities  = []
    reading = False

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading = True
                continue
            if line in ("EOF", "") and reading:
                break
            if reading:
                parts = line.split()
                if len(parts) >= 3:
                    idx = parts[0]
                    x   = float(parts[1])
                    y   = float(parts[2])
                    cities.append(City(f"C{idx}", x, y))

    if not cities:
        raise ValueError(
            f"No cities found in {filepath!r}. "
            "Verify the file contains a NODE_COORD_SECTION."
        )

    # Normalise raw TSPLIB coordinates (often in the thousands) to [0, 100].
    # The largest axis span becomes 100; the shorter axis scales proportionally.
    xs           = [c.x for c in cities]
    ys           = [c.y for c in cities]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span         = max(max_x - min_x, max_y - min_y) or 1.0

    return [
        City(c.name, round(100.0 * (c.x - min_x) / span, 4),
                      round(100.0 * (c.y - min_y) / span, 4))
        for c in cities
    ]


def download_wi29(save_dir: str = "data") -> str:
    """
    Download the wi29 Western Sahara TSPLIB95 benchmark (29 cities).

    Saves the decompressed file to save_dir/wi29.tsp and returns its path.
    If the file already exists, the download is skipped. If the TSPLIB
    server is unreachable, the coordinates are written directly from an
    embedded copy of the official dataset.

    Source: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

    Args:
        save_dir: directory in which to store wi29.tsp.

    Returns:
        Absolute path to the saved wi29.tsp file.
    """
    import urllib.request
    import gzip
    import shutil

    os.makedirs(save_dir, exist_ok=True)
    tsp_path = os.path.join(save_dir, "wi29.tsp")

    if os.path.exists(tsp_path):
        print(f"  wi29.tsp already present at {tsp_path}")
        return tsp_path

    gz_path = tsp_path + ".gz"
    url     = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/wi29.tsp.gz"

    print("  Downloading wi29 from TSPLIB95 ...")
    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(tsp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"  Saved to {tsp_path}")
    except Exception as exc:
        print(f"  Download failed ({exc}). Writing embedded wi29 coordinates.")
        _write_wi29_embedded(tsp_path)

    return tsp_path


def _write_wi29_embedded(filepath: str) -> None:
    """
    Write the official wi29 TSPLIB95 coordinates directly to filepath.

    Called automatically by download_wi29() when the TSPLIB server is
    unreachable. Coordinates are the original values from:
        Gr, 1977 — 29 cities in Western Sahara.
    """
    content = """\
NAME: wi29
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
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  Embedded wi29 written to {filepath}")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # ── Part 1: Euclidean distance matrix ─────────────────────────────────────
    print("=" * 55)
    print("Distance matrix — 5 inline cities")
    print("=" * 55)

    cities = [
        City("A", 0, 0),
        City("B", 2, 4),
        City("C", 5, 2),
        City("D", 6, 6),
        City("E", 1, 7),
    ]

    dist = build_distance_matrix(cities)
    print_distance_matrix(cities, dist)
    print(f"\nDist A→C : {dist[0][2]:.4f}")
    print(f"Dist C→A : {dist[2][0]:.4f}  (symmetric)")

    # ── Part 2: Random instance generator ─────────────────────────────────────
    print("\nRandom 6-city instance (seed=7):")
    rand_cities = generate_random_cities(6, seed=7)
    for c in rand_cities:
        print(f"  {c}")

    # ── Part 3: TSPLIB95 loader and downloader ─────────────────────────────────
    print("\n" + "=" * 55)
    print("TSPLIB95 loader — wi29 Western Sahara")
    print("=" * 55)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    tsp_path = download_wi29(save_dir=data_dir)
    wi29     = load_tsplib(tsp_path)
    wi29_dist = build_distance_matrix(wi29)

    print(f"  Cities loaded : {len(wi29)}")
    print(f"  First city    : {wi29[0]}")
    print(f"  Last city     : {wi29[-1]}")
    print(f"  X range       : {min(c.x for c in wi29):.2f} – {max(c.x for c in wi29):.2f}")
    print(f"  Y range       : {min(c.y for c in wi29):.2f} – {max(c.y for c in wi29):.2f}")
    print(f"  Matrix size   : {len(wi29_dist)}×{len(wi29_dist)}")
    print(f"  dist[0][1]    : {wi29_dist[0][1]:.4f}")

    assert len(wi29) == 29,                            "Expected 29 cities"
    assert all(0.0 <= c.x <= 100.0 for c in wi29),    "X coordinate out of [0, 100]"
    assert all(0.0 <= c.y <= 100.0 for c in wi29),    "Y coordinate out of [0, 100]"
    print("\n  PASS — tsp.py smoke test complete.")