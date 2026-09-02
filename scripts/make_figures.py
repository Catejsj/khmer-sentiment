#!/usr/bin/env python3
"""
Figures for the Person 4 slides.

  1. Words that signal each sentiment -> DIVERGING (polarity around a neutral
     zero), so two hues with a gray midpoint.
  2. Feature count per representation -> MAGNITUDE, single hue.

Khmer needs a font that covers the script; matplotlib's default does not, and
silently renders empty boxes. Noto Sans Khmer is set explicitly and the script
fails loudly if it is missing rather than producing unreadable figures.

Usage:
    python scripts/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.feature_extraction.text import CountVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
FIGURES = os.path.join(ROOT, "reports", "figures")

BLUE = "#2a78d6"      # positive pole
RED = "#e34948"       # negative pole
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"

KHMER_FONT = "Noto Sans Khmer"
TOKEN_PATTERN = r"\S+"


def use_khmer_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    if KHMER_FONT not in available:
        raise SystemExit(
            f"'{KHMER_FONT}' not installed. Khmer would render as empty boxes.\n"
            "Install with: sudo pacman -S noto-fonts"
        )
    plt.rcParams["font.family"] = [KHMER_FONT, "DejaVu Sans"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=11)


def fig_sentiment_words():
    """Positive vs negative words, log-odds, diverging."""
    train = pd.read_csv(os.path.join(SPLITS, "train.csv"))
    train["cleaned_text"] = train["cleaned_text"].fillna("")
    vec = CountVectorizer(min_df=2, token_pattern=TOKEN_PATTERN)
    X = vec.fit_transform(train["cleaned_text"])
    vocab = vec.get_feature_names_out()

    is_pos = (train["label"] == "positive").to_numpy()
    is_neg = (train["label"] == "negative").to_numpy()
    pos = np.asarray(X[is_pos].sum(axis=0)).ravel() + 1
    neg = np.asarray(X[is_neg].sum(axis=0)).ravel() + 1
    total = pos + neg - 2
    ratio = np.log((pos / pos.sum()) / (neg / neg.sum()))
    ok = total >= 4

    top = np.where(ok, ratio, -np.inf).argsort()[::-1][:10]
    bot = np.where(ok, ratio, np.inf).argsort()[:10]
    idx = list(bot[::-1]) + list(top[::-1])
    words = [vocab[i] for i in idx]
    vals = [ratio[i] for i in idx]

    fig, ax = plt.subplots(figsize=(11, 8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    bars = ax.barh(range(len(vals)), vals,
                   color=[BLUE if v > 0 else RED for v in vals], height=0.68)
    for b in bars:
        b.set_joinstyle("round")

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=13, color=INK)
    ax.axvline(0, color="#9a9992", linewidth=1.2)
    ax.set_xlabel("← more NEGATIVE        log-odds        more POSITIVE →",
                  fontsize=11, color=INK_2, labelpad=12)
    ax.set_title("Which Khmer words carry sentiment",
                 fontsize=17, color=INK, pad=38, loc="left", fontweight="bold")
    ax.text(0, 1.012, "computed on the 272 training sentences",
            transform=ax.transAxes, fontsize=11, color=INK_2)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    style(ax)

    for i, v in enumerate(vals):
        ax.text(v + (0.08 if v > 0 else -0.08), i, f"{v:+.1f}", va="center",
                ha="left" if v > 0 else "right", fontsize=10, color=INK_2)
    ax.set_xlim(min(vals) - 0.8, max(vals) + 0.8)

    fig.tight_layout()
    out = os.path.join(FIGURES, "fig1_sentiment_words.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_feature_counts():
    names = ["Bag-of-Words", "N-grams\n(1–2)", "TF-IDF\n(1–2)", "Word2Vec", "fastText"]
    counts = [858, 1086, 1086, 100, 100]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    bars = ax.bar(range(len(counts)), counts, color=BLUE, width=0.6)
    for b in bars:
        b.set_joinstyle("round")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=12, color=INK)
    ax.set_ylabel("number of features", fontsize=11, color=INK_2, labelpad=10)
    ax.set_title("Five representations, very different sizes",
                 fontsize=17, color=INK, pad=38, loc="left", fontweight="bold")
    ax.text(0, 1.015, "sparse word columns, or 100 dense dimensions",
            transform=ax.transAxes, fontsize=11, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    style(ax)

    for i, c in enumerate(counts):
        ax.text(i, c + 25, f"{c:,}", ha="center", fontsize=12, color=INK)
    ax.set_ylim(0, max(counts) * 1.14)

    fig.tight_layout()
    out = os.path.join(FIGURES, "fig2_feature_counts.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_model_results():
    """Validation vs test macro-F1 for the top models - the two disagree, and
    that disagreement is the point of the figure."""
    path = os.path.join(ROOT, "reports", "model_results.csv")
    if not os.path.exists(path):
        print("skipping model figure - run scripts/train_models.py first")
        return None
    d = pd.read_csv(path).dropna(subset=["test_f1"]).nlargest(10, "test_f1")
    d = d.iloc[::-1]
    names = [f"{r.model}\n{r.representation[:22]}" for r in d.itertuples()]

    fig, ax = plt.subplots(figsize=(11, 8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(d))
    ax.barh(y + 0.19, d["val_f1"], height=0.36, color="#9ec5f4", label="validation")
    ax.barh(y - 0.19, d["test_f1"], height=0.36, color=BLUE, label="test")
    ax.axvline(1 / 3, color="#9a9992", linewidth=1.4, linestyle="--")
    ax.text(1 / 3 + 0.006, len(d) - 0.4, "random baseline 0.333",
            fontsize=10, color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.set_xlabel("macro-F1", fontsize=11, color=INK_2, labelpad=10)
    ax.set_title("Top 10 models: validation vs test",
                 fontsize=17, color=INK, pad=38, loc="left", fontweight="bold")
    ax.text(0, 1.012, "ranked by test score — validation ranks them differently",
            transform=ax.transAxes, fontsize=11, color=INK_2)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(d["test_f1"].max(), d["val_f1"].max()) * 1.15)
    style(ax)

    fig.tight_layout()
    out = os.path.join(FIGURES, "fig3_model_results.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGURES, exist_ok=True)
    use_khmer_font()
    for path in (fig_sentiment_words(), fig_feature_counts(), fig_model_results()):
        if path:
            print("wrote", path)


if __name__ == "__main__":
    main()
