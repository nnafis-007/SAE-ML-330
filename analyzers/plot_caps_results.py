#!/usr/bin/env python3
"""
plot_caps_results.py
====================

Generates interpretable plots from a caps_test_report.json produced by
run_caps_test.py.

Plots produced
--------------
1. word_overview.png       – Mean Jaccard & cosine per word (bar chart)
2. pairwise_heatmaps.png   – Per-word pairwise Jaccard + cosine heatmaps
3. lower_vs_upper.png      – Direct lower↔UPPER comparison across all words
4. feature_breakdown.png   – Stacked bar: unique / shared / universal features
5. jaccard_vs_cosine.png   – Scatter: every variant pair, colour-coded by word

Usage
-----
    python plot_caps_results.py
    python plot_caps_results.py --report caps_test_report.json
    python plot_caps_results.py --output-dir my_plots/
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --------------------------------------------------------------------------- #
# Style constants                                                               #
# --------------------------------------------------------------------------- #

SIGNAL_COLOURS = {
    "CASE-INVARIANT (strong)":        "#2ecc71",
    "PARTIALLY case-sensitive":        "#f39c12",
    "CASE-SENSITIVE (features differ)": "#e74c3c",
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


def pairwise_matrix(word_result: dict, metric: str) -> tuple[list[str], np.ndarray]:
    """Return (variant_labels, symmetric similarity matrix) for one word."""
    forms  = [v["form"] for v in word_result["variants"]]
    n      = len(forms)
    idx    = {f: i for i, f in enumerate(forms)}
    mat    = np.ones((n, n))

    for pair in word_result["pairwise"]:
        i = idx[pair["variant_a"]]
        j = idx[pair["variant_b"]]
        mat[i, j] = mat[j, i] = pair[metric]

    return forms, mat


# =========================================================================== #
# Plot 1 – Word-level overview                                                 #
# =========================================================================== #

def plot_word_overview(words: list[dict], out_path: Path) -> None:
    """
    Grouped bars: mean Jaccard (solid) and mean cosine (hatched) per word,
    colour-coded by case-sensitivity signal.
    """
    names    = [w["word"] for w in words]
    jaccards = [w["mean_jaccard"]   for w in words]
    cosines  = [w["mean_cosine_sim"] for w in words]
    colours  = [SIGNAL_COLOURS[w["interpretation"]] for w in words]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(7, 1.8 * len(names)), 4.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, jaccards, width, color=colours,
                    alpha=0.85, edgecolor="white", label="Mean Jaccard")
    bars2 = ax2.bar(x + width / 2, cosines, width, color=colours,
                    alpha=0.45, edgecolor="white", hatch="///",
                    label="Mean cosine")

    for bar, val in zip(bars1, jaccards):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, cosines):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Mean Jaccard similarity")
    ax2.set_ylabel("Mean cosine similarity")
    ax1.set_ylim(0, 1.15)
    ax2.set_ylim(0, 1.15)
    ax1.axhline(0.40, color="grey", lw=1, ls="--", alpha=0.5)
    ax1.axhline(0.20, color="grey", lw=1, ls=":",  alpha=0.5)
    ax1.text(len(names) - 0.5, 0.41, "strong threshold", fontsize=7,
             color="grey", va="bottom")
    ax1.text(len(names) - 0.5, 0.21, "moderate threshold", fontsize=7,
             color="grey", va="bottom")

    patches = [mpatches.Patch(color=c, label=s)
               for s, c in SIGNAL_COLOURS.items()]
    ax1.legend(handles=patches, fontsize=8, title="Signal",
               title_fontsize=8, loc="upper left")

    fig.suptitle("Capitalisation Invariance – Feature Overlap per Word", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Plot 2 – Per-word pairwise heatmaps                                          #
# =========================================================================== #

def plot_pairwise_heatmaps(words: list[dict], out_path: Path) -> None:
    """
    Two rows (Jaccard | cosine), one column per word.
    """
    n = len(words)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7), squeeze=False)

    for col, word in enumerate(words):
        for row, (metric, cmap, vmin, row_label) in enumerate([
            ("jaccard",    "YlOrRd", 0.0, "Jaccard"),
            ("cosine_sim", "Blues",  0.7, "Cosine"),
        ]):
            forms, mat = pairwise_matrix(word, metric)
            ax = axes[row][col]
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=1.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(len(forms)))
            ax.set_yticks(range(len(forms)))
            ax.set_xticklabels(forms, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels(forms, fontsize=8)

            for i in range(len(forms)):
                for j in range(len(forms)):
                    ax.text(j, i, f"{mat[i, j]:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > 0.75 else "black")

            if col == 0:
                ax.set_ylabel(f"{row_label} similarity")
            if row == 0:
                ax.set_title(f"'{word['word']}'", fontsize=11)

    fig.suptitle("Pairwise Similarity Across Capitalisation Variants",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Plot 3 – lower vs UPPER direct comparison                                    #
# =========================================================================== #

def plot_lower_vs_upper(words: list[dict], out_path: Path) -> None:
    """
    Side-by-side bars showing Jaccard and cosine for the lower↔UPPER pair
    specifically — the starkest capitalisation contrast.
    """
    valid = [w for w in words if w["lower_vs_upper"] is not None]
    if not valid:
        print("  skipped lower_vs_upper plot (no lower↔UPPER pairs found)")
        return

    names    = [w["word"] for w in valid]
    jaccards = [w["lower_vs_upper"]["jaccard"]    for w in valid]
    cosines  = [w["lower_vs_upper"]["cosine_sim"] for w in valid]
    colours  = [SIGNAL_COLOURS[w["interpretation"]] for w in valid]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(names)), 4.5))
    bars1 = ax.bar(x - width / 2, jaccards, width, color=colours,
                   alpha=0.85, edgecolor="white", label="Jaccard")
    bars2 = ax.bar(x + width / 2, cosines,  width, color=colours,
                   alpha=0.45, edgecolor="white", hatch="///", label="Cosine")

    for bar, val in zip(bars1, jaccards):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, cosines):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Similarity")
    ax.set_ylim(0, 1.15)
    ax.set_title("Lower ↔ UPPER Feature Overlap\n"
                 "(starkest capitalisation contrast)")
    ax.axhline(0.40, color="grey", lw=1, ls="--", alpha=0.5,
               label="Strong threshold (0.40)")
    ax.axhline(0.20, color="grey", lw=1, ls=":",  alpha=0.5,
               label="Moderate threshold (0.20)")
    ax.legend(fontsize=8)

    patches = [mpatches.Patch(color=c, label=s)
               for s, c in SIGNAL_COLOURS.items()]
    ax.legend(handles=patches + [
        mpatches.Patch(facecolor="grey", alpha=0.85, label="Jaccard"),
        mpatches.Patch(facecolor="grey", alpha=0.45, hatch="///",
                       label="Cosine"),
    ], fontsize=7, title="Legend", title_fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Plot 4 – Feature category breakdown per variant                              #
# =========================================================================== #

def _feature_categories(word: dict) -> dict[str, dict[str, int]]:
    """
    For each variant, count features as:
      universal   – in every variant's top-K
      pair-shared – in at least one other variant but not all
      unique      – in no other variant's top-K
    """
    variants  = [v["form"] for v in word["variants"]]
    top_sets  = {v: set(word["top_features_per_variant"][v]) for v in variants}
    universal = set(word["universal_shared_features"])

    counts = {}
    for v in variants:
        others       = set.union(*(top_sets[o] for o in variants if o != v))
        n_universal  = len(top_sets[v] & universal)
        n_unique     = len(top_sets[v] - others)
        n_pair       = len(top_sets[v]) - n_universal - n_unique
        counts[v]    = {"universal": n_universal,
                        "pair-shared": n_pair,
                        "unique": n_unique}
    return counts


def plot_feature_breakdown(words: list[dict], out_path: Path) -> None:
    """
    Stacked horizontal bars — one panel per word — showing the feature
    category breakdown for each capitalisation variant.
    """
    cat_colours = {
        "universal":   "#2ecc71",
        "pair-shared": "#f39c12",
        "unique":      "#e74c3c",
    }
    categories = ["universal", "pair-shared", "unique"]
    n = len(words)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5), squeeze=False)

    for col, word in enumerate(words):
        ax       = axes[0][col]
        counts   = _feature_categories(word)
        variants = [v["form"] for v in word["variants"]]
        y_pos    = np.arange(len(variants))
        lefts    = np.zeros(len(variants))

        for cat in categories:
            vals = np.array([counts[v][cat] for v in variants], dtype=float)
            ax.barh(y_pos, vals, left=lefts, label=cat,
                    color=cat_colours[cat], edgecolor="white", height=0.6)
            for i, (v, l) in enumerate(zip(vals, lefts)):
                if v > 0:
                    ax.text(l + v / 2, i, str(int(v)),
                            ha="center", va="center", fontsize=8, color="white")
            lefts += vals

        ax.set_yticks(y_pos)
        ax.set_yticklabels(variants, fontsize=9)
        ax.set_xlabel("# features")
        ax.set_title(f"'{word['word']}'")
        ax.set_xlim(0, word["top_k"] + 2)

        if col == n - 1:
            ax.legend(title="Feature type", loc="lower right",
                      fontsize=8, title_fontsize=8)

    fig.suptitle("Top-K Feature Breakdown per Capitalisation Variant",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Plot 5 – Jaccard vs cosine scatter                                           #
# =========================================================================== #

def plot_jaccard_vs_cosine(words: list[dict], out_path: Path) -> None:
    """
    Scatter of pairwise Jaccard vs cosine for all variant pairs, grouped by
    word.  Lower↔UPPER pairs are annotated with a star marker.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, word in enumerate(words):
        colour = PALETTE[i % len(PALETTE)]
        for pair in word["pairwise"]:
            is_lu = (pair["label_a"] == "lower" and pair["label_b"] == "upper")
            marker = "*" if is_lu else "o"
            ms     = 130  if is_lu else 55
            ax.scatter(pair["jaccard"], pair["cosine_sim"],
                       color=colour, s=ms, marker=marker,
                       alpha=0.85, zorder=3,
                       label=word["word"] if pair is word["pairwise"][0] else "")
            label = f"{pair['variant_a']}↔{pair['variant_b']}"
            ax.annotate(label, (pair["jaccard"], pair["cosine_sim"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=6, alpha=0.8)

    for thresh, txt in [(0.40, "strong"), (0.20, "moderate")]:
        ax.axvline(thresh, color="grey", lw=1, ls="--", alpha=0.4)
        ax.text(thresh + 0.005, ax.get_ylim()[0] + 0.002,
                txt, fontsize=7, color="grey", va="bottom")

    ax.set_xlabel("Pairwise Jaccard similarity")
    ax.set_ylabel("Pairwise cosine similarity")
    ax.set_title("Jaccard vs Cosine – All Capitalisation Pairs\n"
                 "(★ = lower↔UPPER pair)")
    ax.grid(True, alpha=0.25)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), title="Word",
              fontsize=8, title_fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# =========================================================================== #
# Entry point                                                                  #
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot capitalisation invariance results from caps_test_report.json."
    )
    parser.add_argument(
        "--report", default="caps_test_report.json",
        help="Path to the JSON report. Default: caps_test_report.json"
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
    report = load_report(report_path)
    words  = report["words"]

    print(f"Generating plots → {out_dir}/")

    plot_word_overview(words,     out_dir / "caps_word_overview.png")
    plot_pairwise_heatmaps(words, out_dir / "caps_pairwise_heatmaps.png")
    plot_lower_vs_upper(words,    out_dir / "caps_lower_vs_upper.png")
    plot_feature_breakdown(words, out_dir / "caps_feature_breakdown.png")
    plot_jaccard_vs_cosine(words, out_dir / "caps_jaccard_vs_cosine.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
