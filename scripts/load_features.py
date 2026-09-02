#!/usr/bin/env python3
"""
FOR PERSON 4 - load any of the five representations in one line.

    from load_features import load
    X_train, X_val, X_test, y_train, y_val, y_test = load("tfidf")

Names: "bow", "ngram", "tfidf", "word2vec", "fasttext"

  bow / ngram / tfidf  -> scipy sparse CSR matrices
  word2vec / fasttext  -> dense numpy arrays

Both kinds go straight into scikit-learn estimators, so the same training loop
works for all five:

    for name in AVAILABLE:
        X_tr, X_va, X_te, y_tr, y_va, y_te = load(name)
        model.fit(X_tr, y_tr)

Everything was fit on TRAIN ONLY, so there is no leakage to worry about here.
Labels are the strings "positive", "neutral" and "negative"; use to_numeric()
if a model needs integers instead.
"""
import os

import numpy as np
from scipy import sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(ROOT, "data", "features")

AVAILABLE = ["bow", "ngram", "tfidf", "word2vec", "fasttext"]
SPLITS = ["train", "val", "test"]
CLASSES = ["positive", "neutral", "negative"]


def _load_matrix(name: str):
    path = os.path.join(FEATURES, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\nRun: python scripts/representations.py"
        )
    z = np.load(path, allow_pickle=True)
    if str(z["kind"]) == "sparse":
        return {
            s: sparse.csr_matrix(
                (z[f"{s}_data"], z[f"{s}_indices"], z[f"{s}_indptr"]),
                shape=tuple(z[f"{s}_shape"]),
            )
            for s in SPLITS
        }
    return {s: z[s] for s in SPLITS}


def load(name: str):
    """Return X_train, X_val, X_test, y_train, y_val, y_test."""
    if name not in AVAILABLE:
        raise ValueError(f"Unknown representation {name!r}. Choose from {AVAILABLE}")
    X = _load_matrix(name)
    y = np.load(os.path.join(FEATURES, "labels.npz"), allow_pickle=True)
    return (X["train"], X["val"], X["test"],
            y["train"], y["val"], y["test"])


def to_numeric(y):
    """positive -> 0, neutral -> 1, negative -> 2, for models wanting integers."""
    lookup = {c: i for i, c in enumerate(CLASSES)}
    return np.array([lookup[v] for v in np.asarray(y)])


def feature_names(name: str):
    """Column names for bow / ngram / tfidf. Embeddings have no word per column,
    so this returns None for word2vec and fasttext - their dimensions are not
    individually interpretable."""
    z = np.load(os.path.join(FEATURES, f"{name}.npz"), allow_pickle=True)
    return z["vocab"] if "vocab" in z.files else None


if __name__ == "__main__":
    print("representation  X_train shape        type            n_features")
    for name in AVAILABLE:
        try:
            Xtr, Xva, Xte, ytr, yva, yte = load(name)
        except FileNotFoundError as e:
            print(f"{name:14}  -- not built yet --")
            continue
        kind = "sparse" if sparse.issparse(Xtr) else "dense"
        print(f"{name:14}  {str(Xtr.shape):<20} {kind:<15} {Xtr.shape[1]}")
    try:
        _, _, _, ytr, yva, yte = load("bow")
        import collections
        print(f"\nlabels  train {dict(collections.Counter(ytr))}")
        print(f"        val   {dict(collections.Counter(yva))}")
        print(f"        test  {dict(collections.Counter(yte))}")
    except FileNotFoundError:
        pass
