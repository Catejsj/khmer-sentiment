#!/usr/bin/env python3
"""
Figures for the slides.

  1. Words that signal each sentiment -> DIVERGING (polarity around a neutral
     zero), so two hues with a gray midpoint.
  2. Feature count per representation -> MAGNITUDE, single hue.
  3. Validation against test per model -> the two disagree (Person 5).
  4. Confusion matrix of the reported model -> where the errors go (Person 6).
  5. Human annotators scored like a model -> the ceiling (Person 6).

Khmer needs a font that covers the script; matplotlib's default does not, and
silently renders empty boxes. Noto Sans Khmer is set explicitly and the script
fails loudly if it is missing rather than producing unreadable figures.

Usage:
    python scripts/make_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

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
    ax.text(0, 1.012, f"computed on the {len(train)} training sentences",
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
    """Counts are read from the saved feature files so the figure cannot drift
    out of step with the data the way a hardcoded list does."""
    import numpy as _np
    names = ["Bag-of-Words", "N-grams\n(1–2)", "TF-IDF\n(1–2)", "Word2Vec", "fastText"]
    counts = []
    for key in ("bow", "ngram", "tfidf", "word2vec", "fasttext"):
        z = _np.load(os.path.join(ROOT, "data", "features", f"{key}.npz"), allow_pickle=True)
        counts.append(int(z["train_shape"][1]) if str(z["kind"]) == "sparse"
                      else int(z["train"].shape[1]))

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


def fig_confusion():
    """Where the reported model's errors go - Person 6."""
    path = os.path.join(ROOT, "reports", "model_results.csv")
    if not os.path.exists(path):
        print("skipping confusion figure - run scripts/train_models.py first")
        return None

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyse_results import refit
    from load_features import AVAILABLE, CLASSES

    res = pd.read_csv(path).dropna(subset=["test_f1"])
    res = res[res.representation.isin(AVAILABLE)]
    row = res.loc[res["test_f1"].idxmax()]
    _, pred, y_true = refit(row["model"], row["representation"], str(row["best_params"]))
    cm = confusion_matrix(y_true, pred, labels=CLASSES)
    share = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                    fontsize=15, color="#ffffff" if share[i, j] > 0.5 else INK)

    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=12, color=INK)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=12, color=INK)
    ax.set_xlabel("predicted", fontsize=12, color=INK_2, labelpad=10)
    ax.set_ylabel("actual", fontsize=12, color=INK_2, labelpad=10)
    ax.set_title("Where the errors go", fontsize=17, color=INK, pad=38,
                 loc="left", fontweight="bold")
    ax.text(0, 1.03, f"{row['model']} on {row['representation']} — it hedges toward "
                     "neutral rather than flipping polarity",
            transform=ax.transAxes, fontsize=11, color=INK_2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    out = os.path.join(FIGURES, "fig4_confusion.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_ceiling():
    """The model against the humans, in the same units - Person 6."""
    path = os.path.join(ROOT, "reports", "model_results.csv")
    if not os.path.exists(path):
        print("skipping ceiling figure - run scripts/train_models.py first")
        return None

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyse_results import human_scores

    rows, pooled = human_scores()
    if not rows:
        print("skipping ceiling figure - no adjudicated annotations")
        return None

    res = pd.read_csv(path).dropna(subset=["test_f1"])
    best = res["test_f1"].max()

    labels = [f"{r['pair']}\n{r['who']}" for r in rows] + ["all pairs\npooled", "our best model\non test"]
    vals = [r["f1"] for r in rows] + [pooled, best]
    colors = ["#9ec5f4"] * len(rows) + ["#9ec5f4", BLUE]

    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    for b in bars:
        b.set_joinstyle("round")

    ax.axhline(1 / 3, color="#9a9992", linewidth=1.4, linestyle="--")
    ax.text(-0.45, 1 / 3 + 0.015, "random baseline 0.333",
            fontsize=10, color=INK_2, ha="left")

    ax.set_xticks(range(len(labels)), labels, fontsize=11, color=INK)
    ax.set_ylabel("macro-F1", fontsize=11, color=INK_2, labelpad=10)
    ax.set_title("How well do humans do the same task?",
                 fontsize=17, color=INK, pad=38, loc="left", fontweight="bold")
    ax.text(0, 1.015, "one annotator predicting their partner's labels, scored exactly like a model",
            transform=ax.transAxes, fontsize=11, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.0)
    style(ax)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=12, color=INK)

    fig.tight_layout()
    out = os.path.join(FIGURES, "fig5_ceiling.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGURES, exist_ok=True)
    use_khmer_font()
    for path in (fig_sentiment_words(), fig_feature_counts(), fig_model_results(),
                 fig_confusion(), fig_ceiling()):
        if path:
            print("wrote", path)


if __name__ == "__main__":
    main()
