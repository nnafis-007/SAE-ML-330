#!/usr/bin/env python3
"""
plot_synonym_results.py
=======================

Generates interpretable plots from a synonym_test_report.json produced by
run_synonym_test.py.

Plots produced
--------------
1. cluster_overview.png   – Mean Jaccard & cosine per cluster (bar chart)
2. pairwise_heatmaps.png  – Per-cluster pairwise Jaccard + cosine heatmaps
3. feature_breakdown.png  – Stacked bar: unique / pair-shared / universal features
4. jaccard_vs_cosine.png  – Scatter of Jaccard vs cosine for every word pair
5. universal_features.png – Count of universal-shared features per cluster

Usage
-----
    python plot_synonym_results.py
    python plot_synonym_results.py --report path/to/synonym_test_report.json
    python plot_synonym_results.py --output-dir my_plots/
"""

import argparse
import json
from pathlib import Path
from itertools import combinations
from typing import Any

import matplotlib
matplotlib.use("Agg")             # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --------------------------------------------------------------------------- #
# Colour / style constants                                                      #
# --------------------------------------------------------------------------- #

SIGNAL_COLOURS = {
    "STRONG synonym signal":   "#2ecc71",
    "MODERATE synonym signal": "#f39c12",
    "WEAK synonym signal":     "#e74c3c",
}

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
           "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]

plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# =========================================================================== #
# Data helpers                                                                 #
# =========================================================================== #

def load_report(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def pairwise_matrix(cluster: dict, metric: str) -> tuple[list[str], np.ndarray]:
    """Build a symmetric matrix of *metric* values for one cluster."""
    words = cluster["words"]
    n = len(words)
    idx = {w: i for i, w in enumerate(words)}
    mat = np.ones((n, n))          # diagonal = 1.0 (self-similarity)

    for pair in cluster["pairwise"]:
        i, j = idx[pair["word_a"]], idx[pair["word_b"]]
        mat[i, j] = mat[j, i] = pair[metric]

    return words, mat


# =========================================================================== #
# Plot 1 – Cluster overview                                                    #
# =========================================================================== #

def plot_cluster_overview(clusters: list[dict], out_path: Path) -> None:
    """
    Grouped bar chart: mean Jaccard (left axis) + mean cosine (right axis)
    for every cluster, colour-coded by signal strength.
    """
    names     = [c["cluster"] for c in clusters]
    jaccards  = [c["mean_jaccard"] for c in clusters]
    cosines   = [c["mean_cosine_sim"] for c in clusters]
    signals   = [c["interpretation"] for c in clusters]
    bar_cols  = [SIGNAL_COLOURS[s] for s in signals]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, jaccards, width, color=bar_cols,
                    label="Mean Jaccard", alpha=0.85, edgecolor="white")
    bars2 = ax2.bar(x + width / 2, cosines,  width, color=bar_cols,
                    label="Mean cosine",  alpha=0.50, edgecolor="white",
                    hatch="///")

    # Value labels
    for bar, val in zip(bars1, jaccards):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, cosines):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("Mean Jaccard similarity")
    ax2.set_ylabel("Mean cosine similarity")
    ax1.set_ylim(0, 1.15)
    ax2.set_ylim(0, 1.15)
    ax1.axhline(0.40, color="grey", lw=1, ls="--", alpha=0.6,
                label="Jaccard strong threshold (0.40)")
    ax1.axhline(0.20, color="grey", lw=1, ls=":",  alpha=0.6,
                label="Jaccard moderate threshold (0.20)")

    # Legend: signal colours
    legend_patches = [
        mpatches.Patch(color=col, label=sig)
        for sig, col in SIGNAL_COLOURS.items()
    ]
    ax1.legend(handles=legend_patches, loc="upper left", fontsize=8,
               title="Signal strength", title_fontsize=8)

    fig.suptitle("Synonym Cluster Overview – Feature Overlap Metrics", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Plot 2 – Per-cluster pairwise heatmaps                                       #
# =========================================================================== #

def plot_pairwise_heatmaps(clusters: list[dict], out_path: Path) -> None:
    """
    Two sub-figure groups (Jaccard | cosine), each with one panel per cluster.
    """
    n_clusters = len(clusters)
    fig, axes = plt.subplots(
        2, n_clusters,
        figsize=(3.5 * n_clusters, 7),
        squeeze=False,
    )

    for col, cluster in enumerate(clusters):
        for row, (metric, cmap, vmin, title_prefix) in enumerate([
            ("jaccard",    "YlOrRd", 0.0, "Jaccard"),
            ("cosine_sim", "Blues",  0.7, "Cosine"),
        ]):
            words, mat = pairwise_matrix(cluster, metric)
            ax = axes[row][col]
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=1.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(len(words)))
            ax.set_yticks(range(len(words)))
            ax.set_xticklabels(words, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels(words, fontsize=8)

            # Annotate cells
            for i in range(len(words)):
                for j in range(len(words)):
                    ax.text(j, i, f"{mat[i, j]:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > 0.75 else "black")

            if col == 0:
                ax.set_ylabel(f"{title_prefix} similarity")
            if row == 0:
                ax.set_title(f"'{cluster['cluster']}'", fontsize=11)

    fig.suptitle("Pairwise Similarity Between Synonyms", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Plot 3 – Feature overlap breakdown per word                                  #
# =========================================================================== #

def _count_feature_categories(cluster: dict) -> dict[str, dict[str, int]]:
    """
    For each word in a cluster, count how many of its top-K features are:
      - 'universal'  : shared by ALL words
      - 'pair-shared': shared with at least one other word but not all
      - 'unique'     : not in any other word's top-K
    Returns {word: {category: count}}.
    """
    words        = cluster["words"]
    top_sets     = {w: set(cluster["top_features_per_word"][w]) for w in words}
    universal    = set(cluster["universal_shared_features"])
    unique_to    = {w: set(cluster["unique_features_per_word"][w]) for w in words}

    counts = {}
    for w in words:
        n_universal    = len(top_sets[w] & universal)
        n_unique       = len(unique_to[w])
        n_pair_shared  = len(top_sets[w]) - n_universal - n_unique
        counts[w] = {
            "universal":    n_universal,
            "pair-shared":  n_pair_shared,
            "unique":       n_unique,
        }
    return counts


def plot_feature_breakdown(clusters: list[dict], out_path: Path) -> None:
    """
    Stacked horizontal bar chart per cluster showing how each word's top-K
    features split into universal / pair-shared / unique categories.
    """
    cat_colours = {
        "universal":   "#2ecc71",
        "pair-shared": "#f39c12",
        "unique":      "#e74c3c",
    }
    categories = ["universal", "pair-shared", "unique"]

    n_clusters = len(clusters)
    fig, axes = plt.subplots(1, n_clusters,
                             figsize=(4.5 * n_clusters, 3.5),
                             squeeze=False)

    for col, cluster in enumerate(clusters):
        ax     = axes[0][col]
        counts = _count_feature_categories(cluster)
        words  = cluster["words"]
        y_pos  = np.arange(len(words))

        lefts = np.zeros(len(words))
        for cat in categories:
            vals = np.array([counts[w][cat] for w in words], dtype=float)
            ax.barh(y_pos, vals, left=lefts, label=cat,
                    color=cat_colours[cat], edgecolor="white", height=0.6)
            for i, (v, l) in enumerate(zip(vals, lefts)):
                if v > 0:
                    ax.text(l + v / 2, i, str(int(v)),
                            ha="center", va="center", fontsize=8, color="white")
            lefts += vals

        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.set_xlabel("# features")
        ax.set_title(f"'{cluster['cluster']}'")
        ax.set_xlim(0, cluster["top_k"] + 2)

        if col == n_clusters - 1:
            ax.legend(title="Feature type", loc="lower right",
                      fontsize=8, title_fontsize=8)

    fig.suptitle("Top-K Feature Breakdown per Word", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Plot 4 – Jaccard vs cosine scatter                                           #
# =========================================================================== #

def plot_jaccard_vs_cosine(clusters: list[dict], out_path: Path) -> None:
    """
    Scatter plot of Jaccard vs cosine similarity for every synonym pair,
    colour-coded by cluster.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, cluster in enumerate(clusters):
        colour  = PALETTE[i % len(PALETTE)]
        pairs   = cluster["pairwise"]
        jac     = [p["jaccard"]    for p in pairs]
        cos     = [p["cosine_sim"] for p in pairs]
        labels  = [f"{p['word_a']}↔{p['word_b']}" for p in pairs]

        sc = ax.scatter(jac, cos, color=colour, s=70, alpha=0.85, zorder=3,
                        label=cluster["cluster"])

        # Annotate each point
        for x, y, lbl in zip(jac, cos, labels):
            ax.annotate(lbl, (x, y), xytext=(4, 3),
                        textcoords="offset points", fontsize=6.5, alpha=0.85)

    # Reference lines
    for thresh, label in [(0.40, "strong"), (0.20, "moderate")]:
        ax.axvline(thresh, color="grey", lw=1, ls="--", alpha=0.5)
        ax.text(thresh + 0.005, ax.get_ylim()[0] + 0.002,
                label, fontsize=7, color="grey", va="bottom")

    ax.set_xlabel("Pairwise Jaccard similarity")
    ax.set_ylabel("Pairwise cosine similarity")
    ax.set_title("Jaccard vs Cosine Similarity – All Synonym Pairs")
    ax.legend(title="Cluster", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Plot 5 – Universal shared features per cluster                               #
# =========================================================================== #

def plot_universal_features(clusters: list[dict], out_path: Path) -> None:
    """
    Bar chart showing how many features are shared by every word in each
    cluster, as an absolute count and as a fraction of top-K.
    """
    names      = [c["cluster"] for c in clusters]
    n_univ     = [len(c["universal_shared_features"]) for c in clusters]
    top_k      = clusters[0]["top_k"]
    fractions  = [n / top_k for n in n_univ]
    bar_cols   = [SIGNAL_COLOURS[c["interpretation"]] for c in clusters]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    bars = ax1.bar(names, n_univ, color=bar_cols, alpha=0.85, edgecolor="white",
                   width=0.5)
    ax2.plot(names, fractions, "o--", color="#2c3e50", lw=1.5, ms=6,
             label="Fraction of top-K")

    for bar, n in zip(bars, n_univ):
        ax1.text(bar.get_x() + bar.get_width() / 2, n + 0.15,
                 str(n), ha="center", va="bottom", fontsize=9)

    ax1.set_ylabel(f"# features shared by ALL words")
    ax2.set_ylabel(f"Fraction of top-{top_k}")
    ax1.set_ylim(0, top_k * 1.1)
    ax2.set_ylim(0, 1.1)
    ax1.set_title("Universal Shared Features per Cluster\n"
                  "(features in every word's top-K)")

    # Signal-strength legend
    legend_patches = [
        mpatches.Patch(color=col, label=sig)
        for sig, col in SIGNAL_COLOURS.items()
    ]
    ax1.legend(handles=legend_patches, loc="upper right",
               fontsize=8, title="Signal strength", title_fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Entry point                                                                  #
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot synonym test results from a synonym_test_report.json."
    )
    parser.add_argument(
        "--report", default="synonym_test_report.json",
        help="Path to the JSON report. Default: synonym_test_report.json"
    )
    parser.add_argument(
        "--output-dir", default="plots",
        help="Directory for output PNG files. Default: plots/"
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading report from {report_path} …")
    report   = load_report(report_path)
    clusters = report["clusters"]

    print(f"Generating plots → {out_dir}/")

    plot_cluster_overview(
        clusters,
        out_dir / "cluster_overview.png",
    )
    plot_pairwise_heatmaps(
        clusters,
        out_dir / "pairwise_heatmaps.png",
    )
    plot_feature_breakdown(
        clusters,
        out_dir / "feature_breakdown.png",
    )
    plot_jaccard_vs_cosine(
        clusters,
        out_dir / "jaccard_vs_cosine.png",
    )
    plot_universal_features(
        clusters,
        out_dir / "universal_features.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
