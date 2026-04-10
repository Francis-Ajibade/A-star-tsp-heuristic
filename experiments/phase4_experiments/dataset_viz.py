from __future__ import annotations
import sys
import os
import time

# ── Path fix: works from any working directory ────────────────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(THIS_DIR)

sys.path.insert(0, os.path.join(ROOT_DIR, 'phase1_foundation'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'phase2_astar'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'phase3_heuristics'))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

from tsp import City, build_distance_matrix, load_tsplib, download_wi29
from state import TSPState, make_start_state
from baseline import brute_force_tsp, tour_cost
from astar import astar_tsp, null_heuristic, min_edge_heuristic
from heuristic_mst import mst_heuristic


# ── Palette ───────────────────────────────────────────────────────────────────

COLORS = {
    'Null (UCS)' : '#888780',
    'Min-edge'   : '#378ADD',
    'MST'        : '#1D9E75',
    'cities'     : '#0D1B2A',
    'start'      : '#F4A324',
    'edge'       : '#DDDDDD',
}

HEURISTICS = {
    'Null (UCS)' : null_heuristic,
    'Min-edge'   : min_edge_heuristic,
    'MST'        : mst_heuristic,
}


# ── Tour drawing helper ───────────────────────────────────────────────────────

def draw_tour(
    ax,
    cities  : list[City],
    tour    : list[int],
    color   : str,
    title   : str,
    cost    : float,
    nodes   : int,
    time_ms : float,
    optimal : bool,
):
    """
    Draw a single TSP tour on a matplotlib axes.
    Cities as dots, tour as lines, start city highlighted in amber.
    """
    xs = [c.x for c in cities]
    ys = [c.y for c in cities]

    # Tour edges
    for i in range(len(tour)):
        a = cities[tour[i]]
        b = cities[tour[(i + 1) % len(tour)]]
        ax.plot([a.x, b.x], [a.y, b.y],
                color=color, linewidth=1.4, alpha=0.75, zorder=1)

    # All city dots
    ax.scatter(xs, ys, s=40, color=color, zorder=3,
               edgecolors='white', linewidths=0.8)

    # Start city highlighted
    ax.scatter([cities[0].x], [cities[0].y],
               s=100, color=COLORS['start'], zorder=4,
               edgecolors='white', linewidths=1.2)

    # City index labels (small, only for ≤ 15 cities)
    if len(cities) <= 15:
        for i, c in enumerate(cities):
            ax.annotate(str(i + 1), (c.x, c.y),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=6.5, color='#444444')

    opt_str  = "Optimal" if optimal else "Sub-optimal"
    opt_col  = '#1D9E75' if optimal else '#E85D4A'
    ax.set_title(
        f"{title}",
        fontsize=11, fontweight='bold', pad=4
    )
    ax.set_xlabel(
        f"Cost: {cost:.2f}  |  Nodes: {nodes:,}  |  "
        f"Time: {time_ms:.1f}ms  |  "
        f"{opt_str}",
        fontsize=8.5,
        color=opt_col
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_edgecolor('#DDDDDD')
        spine.set_linewidth(0.8)


# ── Breakdown table helper ────────────────────────────────────────────────────

def draw_table(ax, rows: list[list], col_labels: list[str], title: str):
    """
    Draw a styled comparison table on a matplotlib axes.
    """
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)

    colors_header = ['#0D1B2A'] * len(col_labels)
    cell_colors   = []

    for i, row in enumerate(rows):
        row_colors = ['#F7F9FB' if i % 2 == 0 else '#FFFFFF'] * len(row)
        # Highlight optimal column green, sub-optimal red
        if len(row) >= 5:
            opt_val = row[4]
            if opt_val == 'Yes':
                row_colors[4] = '#EAF6F2'
            elif opt_val == 'No':
                row_colors[4] = '#FCEAEA'
        cell_colors.append(row_colors)

    table = ax.table(
        cellText    = rows,
        colLabels   = col_labels,
        cellLoc     = 'center',
        loc         = 'center',
        cellColours = cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#0D1B2A')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#DDDDDD')

    # Style body
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_edgecolor('#DDDDDD')
            # Colour the Optimal? column
            if j == 4:
                val = rows[i][4]
                cell.set_text_props(
                    color='#0F6E56' if val == 'Yes' else '#A32D2D',
                    fontweight='bold'
                )


# ── Main figure ───────────────────────────────────────────────────────────────

def generate_dataset_figure(
    cities   : list[City],
    dist     : list[list[float]],
    out_path : str,
    dataset_name: str = "wi29 — Western Sahara (TSPLIB95)",
):
    """
    Generate the full dataset visualisation figure:

    Row 1 : Three tour maps side by side (one per heuristic)
    Row 2 : Per-heuristic breakdown table
    Row 3 : Cross-size comparison table (5, 8, 10, wi29)

    Saves to out_path as a PNG.
    """
    n = len(cities)
    print(f"\nRunning all heuristics on {dataset_name} ({n} cities)...\n")

    # ── Run all heuristics ────────────────────────────────────────────────────
    results = {}
    for h_name, h_fn in HEURISTICS.items():
        t0 = time.perf_counter()
        r  = astar_tsp(dist, h_fn)
        elapsed = (time.perf_counter() - t0) * 1000

        if r:
            results[h_name] = {
                'path'    : r.path,
                'cost'    : r.cost,
                'nodes'   : r.nodes_expanded,
                'time_ms' : r.runtime_ms,
            }
        print(f"  {h_name:<14} cost={r.cost:.4f}  "
              f"nodes={r.nodes_expanded:>6,}  time={r.runtime_ms:.1f}ms")

    # Brute-force reference (only feasible for small n)
    if n <= 12:
        _, bf_cost = brute_force_tsp(dist)
        print(f"  Brute-force    cost={bf_cost:.4f}  (reference)")
        optimal_cost = bf_cost
    else:
        # Use MST result as proxy for optimal (MST is admissible → optimal)
        optimal_cost = results['MST']['cost']
        print(f"  Optimal cost (MST):  {optimal_cost:.4f}")

    print()

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        3, 3,
        figure    = fig,
        hspace    = 0.45,
        wspace    = 0.25,
        height_ratios = [2.2, 1.0, 1.0],
    )

    # ── Row 1: Tour maps ──────────────────────────────────────────────────────
    h_names = list(HEURISTICS.keys())
    for col, h_name in enumerate(h_names):
        ax  = fig.add_subplot(gs[0, col])
        res = results[h_name]
        is_optimal = abs(res['cost'] - optimal_cost) < 1e-6

        draw_tour(
            ax      = ax,
            cities  = cities,
            tour    = res['path'],
            color   = COLORS[h_name],
            title   = h_name,
            cost    = res['cost'],
            nodes   = res['nodes'],
            time_ms = res['time_ms'],
            optimal = is_optimal,
        )

    # ── Row 2: Per-heuristic breakdown table ──────────────────────────────────
    ax_table = fig.add_subplot(gs[1, :])

    col_labels = ['Heuristic', 'Tour cost', 'Nodes expanded',
                  'Runtime (ms)', 'Optimal?', 'Admissible?', 'Consistent?']

    admissible  = {'Null (UCS)': 'Yes', 'Min-edge': 'Yes', 'MST': 'Yes'}
    consistent  = {'Null (UCS)': 'Yes', 'Min-edge': 'No',  'MST': 'Yes'}

    table_rows = []
    for h_name in h_names:
        res        = results[h_name]
        is_optimal = abs(res['cost'] - optimal_cost) < 1e-6
        table_rows.append([
            h_name,
            f"{res['cost']:.4f}",
            f"{res['nodes']:,}",
            f"{res['time_ms']:.2f}",
            'Yes' if is_optimal else 'No',
            admissible[h_name],
            consistent[h_name],
        ])

    draw_table(
        ax         = ax_table,
        rows       = table_rows,
        col_labels = col_labels,
        title      = f"Heuristic breakdown — {dataset_name}",
    )

    # ── Row 3: Cross-size comparison table ────────────────────────────────────
    ax_cross = fig.add_subplot(gs[2, :])

    from tsp import generate_random_cities

    sizes      = [5, 8, 10, n]
    size_labels= ['5 cities\n(random)', '8 cities\n(random)',
                  '10 cities\n(random)', f'{n} cities\n(wi29 benchmark)']

    cross_cols = ['Dataset', 'Null nodes', 'Min-edge nodes',
                  'MST nodes', 'MST saving vs Null', 'MST cost']
    cross_rows = []

    print("  Computing cross-size comparison...\n")
    for sz, lbl in zip(sizes, size_labels):
        if sz == n:
            c_list = cities
            d_mat  = dist
        else:
            c_list = generate_random_cities(sz, seed=42)
            d_mat  = build_distance_matrix(c_list)

        r_null = astar_tsp(d_mat, null_heuristic)
        r_me   = astar_tsp(d_mat, min_edge_heuristic)
        r_mst  = astar_tsp(d_mat, mst_heuristic)

        saving = 0
        if r_null and r_mst:
            saving = 100 * (r_null.nodes_expanded - r_mst.nodes_expanded) \
                     / r_null.nodes_expanded

        cross_rows.append([
            lbl.replace('\n', ' '),
            f"{r_null.nodes_expanded:,}" if r_null else 'N/A',
            f"{r_me.nodes_expanded:,}"   if r_me   else 'N/A',
            f"{r_mst.nodes_expanded:,}"  if r_mst  else 'N/A',
            f"{saving:.1f}%",
            f"{r_mst.cost:.4f}"          if r_mst  else 'N/A',
        ])
        print(f"    {lbl.replace(chr(10),' '):<25} "
              f"Null={r_null.nodes_expanded:>6,}  "
              f"MST={r_mst.nodes_expanded:>5,}  "
              f"saving={saving:.1f}%")

    draw_table(
        ax         = ax_cross,
        rows       = cross_rows,
        col_labels = cross_cols,
        title      = "Cross-dataset comparison — random instances vs wi29 benchmark",
    )

    # ── Super title ───────────────────────────────────────────────────────────
    fig.suptitle(
        f"A* Heuristic Comparison — {dataset_name}",
        fontsize = 15,
        fontweight = 'bold',
        y = 0.98,
        color = '#0D1B2A',
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COLORS[h], label=h) for h in h_names
    ]
    legend_patches.append(
        mpatches.Patch(color=COLORS['start'], label='Start city')
    )
    fig.legend(
        handles   = legend_patches,
        loc       = 'lower center',
        ncol      = 4,
        fontsize  = 9,
        frameon   = False,
        bbox_to_anchor = (0.5, 0.01),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"\n  Saved → {out_path}")
    return results, optimal_cost


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tsp import load_tsplib, download_wi29, build_distance_matrix

    print("=" * 60)
    print("  Dataset Visualisation — wi29 Western Sahara")
    print("=" * 60)

    # Download / load wi29
    tsp_path = download_wi29(save_dir="../data")
    cities   = load_tsplib(tsp_path)
    dist     = build_distance_matrix(cities)

    print(f"  Loaded {len(cities)} cities from wi29")

    out_path = "../results/charts/dataset_wi29.png"
    results, opt_cost = generate_dataset_figure(
        cities       = cities,
        dist         = dist,
        out_path     = out_path,
        dataset_name = "wi29 — Western Sahara (TSPLIB95)",
    )

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  {'Heuristic':<14} {'Cost':>10} {'Nodes':>10} "
          f"{'Time ms':>10} {'Optimal':>8}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for h_name, res in results.items():
        is_opt = abs(res['cost'] - opt_cost) < 1e-6
        print(f"  {h_name:<14} {res['cost']:>10.4f} "
              f"{res['nodes']:>10,} "
              f"{res['time_ms']:>10.2f} "
              f"{'YES' if is_opt else 'NO':>8}")

    print(f"\n  Figure saved → {out_path}")
    print("  Ready to add to PowerPoint slide.")
    print("=" * 60)