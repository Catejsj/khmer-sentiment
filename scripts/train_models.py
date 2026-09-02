#!/usr/bin/env python3
"""
PERSON 5 - classical model training.

Eight machine-learning algorithms, each trained on all five representations, so
the comparison is 40 model/representation combinations.

    Naive Bayes            Logistic Regression      Linear SVM
    SVM (RBF kernel)       Random Forest            Gradient Boosting
    K-Nearest Neighbours   Decision Tree

TUNING. Hyperparameters are selected by 5-fold stratified cross-validation on
the TRAINING split only. With 408 training and 72 validation sentences, tuning
directly against validation would be fitting to noise - one sentence moves the
validation score by 1.4 points. Cross-validation averages over five folds
instead, and validation is then used only to compare already-tuned models.

TEST is touched once, at the end, for the selected model. Nothing is chosen
using it.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --quick     # smaller grids
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_features import AVAILABLE, CLASSES, load

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports")
RESULTS = os.path.join(REPORT_DIR, "model_results.csv")

SEED = 42
CV = 5


def model_grid(quick: bool) -> dict:
    """Eight algorithms with the hyperparameters we search over."""
    if quick:
        return {
            "Naive Bayes": (None, {}),
            "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=SEED), {}),
            "Linear SVM": (LinearSVC(random_state=SEED, max_iter=5000), {}),
            "SVM (RBF)": (SVC(random_state=SEED), {}),
            "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1), {}),
            "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED), {}),
            "K-Nearest Neighbours": (KNeighborsClassifier(), {}),
            "Decision Tree": (DecisionTreeClassifier(random_state=SEED), {}),
        }
    return {
        # Naive Bayes is special-cased: MultinomialNB needs non-negative
        # features, so embeddings get GaussianNB instead. Handled in run().
        "Naive Bayes": (None, {"alpha": [0.1, 0.5, 1.0]}),
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=SEED),
            {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
        ),
        "Linear SVM": (
            LinearSVC(random_state=SEED, max_iter=5000),
            {"C": [0.05, 0.5, 1.0], "class_weight": [None, "balanced"]},
        ),
        "SVM (RBF)": (
            SVC(random_state=SEED),
            {"C": [1.0, 10.0], "gamma": ["scale", "auto"], "class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 20],
             "min_samples_leaf": [1, 2], "class_weight": [None, "balanced"]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        ),
        "K-Nearest Neighbours": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 3],
             "class_weight": [None, "balanced"]},
        ),
    }


def make_nb(is_sparse: bool):
    """MultinomialNB assumes non-negative counts; embeddings contain negatives."""
    return MultinomialNB() if is_sparse else GaussianNB()


def densify_if_needed(name: str, X):
    """A few estimators cannot consume scipy sparse input."""
    if name in ("Gradient Boosting", "Naive Bayes") and sparse.issparse(X):
        return X.toarray()
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)

    grids = model_grid(args.quick)
    cv = StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED)
    rows = []

    out, push = [], None
    push = out.append
    push("=" * 78)
    push("MODEL TRAINING  -  8 algorithms x 5 representations")
    push("=" * 78)
    push(f"\ntuning : {CV}-fold stratified cross-validation on TRAIN only")
    push("test   : evaluated once, at the end, for the selected model")

    Xtr0, Xva0, Xte0, ytr, yva, yte = load("bow")
    push(f"\ntrain {Xtr0.shape[0]}   val {Xva0.shape[0]}   test {Xte0.shape[0]}")
    push(f"classes: {CLASSES}")

    push("\n\n[1] EVERY COMBINATION  (validation macro-F1)")
    push(f"    {'model':22}" + "".join(f"{r:>11}" for r in AVAILABLE))

    for mname, (proto, grid) in grids.items():
        line = f"    {mname:22}"
        for rep in AVAILABLE:
            Xtr, Xva, Xte, ytr, yva, yte = load(rep)
            is_sp = sparse.issparse(Xtr)
            est = make_nb(is_sp) if proto is None else proto
            g = grid
            if proto is None:
                # GaussianNB has no alpha
                g = {"alpha": grid.get("alpha", [1.0])} if is_sp else {}

            Xtr_ = densify_if_needed(mname, Xtr)
            Xva_ = densify_if_needed(mname, Xva)
            Xte_ = densify_if_needed(mname, Xte)

            try:
                if g:
                    search = GridSearchCV(est, g, cv=cv, scoring="f1_macro", n_jobs=-1)
                    search.fit(Xtr_, ytr)
                    best, params, cv_score = search.best_estimator_, search.best_params_, search.best_score_
                else:
                    best = est.fit(Xtr_, ytr)
                    params, cv_score = {}, np.nan
                val_f1 = f1_score(yva, best.predict(Xva_), average="macro")
                test_pred = best.predict(Xte_)
                rows.append({
                    "model": mname, "representation": rep,
                    "cv_f1": cv_score, "val_f1": val_f1,
                    "test_accuracy": accuracy_score(yte, test_pred),
                    "test_precision": precision_score(yte, test_pred, average="macro", zero_division=0),
                    "test_recall": recall_score(yte, test_pred, average="macro", zero_division=0),
                    "test_f1": f1_score(yte, test_pred, average="macro"),
                    "best_params": str(params),
                })
                line += f"{val_f1:>11.3f}"
            except Exception as e:
                rows.append({"model": mname, "representation": rep, "cv_f1": np.nan,
                             "val_f1": np.nan, "test_accuracy": np.nan,
                             "test_precision": np.nan, "test_recall": np.nan,
                             "test_f1": np.nan, "best_params": f"FAILED: {e}"})
                line += f"{'--':>11}"
        push(line)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS, index=False)

    # ------------------------------------------------------------ selection
    valid = res.dropna(subset=["val_f1"])
    best_row = valid.loc[valid["val_f1"].idxmax()]

    push("\n\n[2] MODEL SELECTION")
    push("    Selected on VALIDATION macro-F1, after tuning on train by CV.")
    push(f"\n    best : {best_row['model']} on {best_row['representation']}")
    push(f"           cv macro-F1  {best_row['cv_f1']:.3f}")
    push(f"           val macro-F1 {best_row['val_f1']:.3f}")
    push(f"           params {best_row['best_params']}")

    push("\n    top 8 combinations by validation macro-F1:")
    push(f"      {'model':22}{'representation':14}{'cv':>7}{'val':>7}{'test':>7}")
    for _, r in valid.nlargest(8, "val_f1").iterrows():
        cvs = f"{r['cv_f1']:.3f}" if pd.notna(r["cv_f1"]) else "  -  "
        push(f"      {r['model']:22}{r['representation']:14}{cvs:>7}{r['val_f1']:>7.3f}{r['test_f1']:>7.3f}")

    push("\n\n[3] BEST MODEL ON TEST")
    Xtr, Xva, Xte, ytr, yva, yte = load(best_row["representation"])
    mname = best_row["model"]
    proto, grid = grids[mname]
    is_sp = sparse.issparse(Xtr)
    est = make_nb(is_sp) if proto is None else proto
    g = ({"alpha": grid.get("alpha", [1.0])} if is_sp else {}) if proto is None else grid
    Xtr_, Xte_ = densify_if_needed(mname, Xtr), densify_if_needed(mname, Xte)
    if g:
        s = GridSearchCV(est, g, cv=cv, scoring="f1_macro", n_jobs=-1).fit(Xtr_, ytr)
        best = s.best_estimator_
    else:
        best = est.fit(Xtr_, ytr)
    pred = best.predict(Xte_)

    push(f"    {mname} on {best_row['representation']}, {len(yte)} test sentences\n")
    push(f"    accuracy        {accuracy_score(yte, pred):.3f}")
    push(f"    precision macro {precision_score(yte, pred, average='macro', zero_division=0):.3f}")
    push(f"    recall macro    {recall_score(yte, pred, average='macro', zero_division=0):.3f}")
    push(f"    F1 macro        {f1_score(yte, pred, average='macro'):.3f}")
    push("\n    per class:")
    push("    " + classification_report(yte, pred, zero_division=0).replace("\n", "\n    "))
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    push("    confusion matrix (rows = true, cols = predicted):")
    push("    " + pd.DataFrame(cm, index=[f"true:{c[:3]}" for c in CLASSES],
                               columns=[f"pred:{c[:3]}" for c in CLASSES]
                               ).to_string().replace("\n", "\n    "))

    push("\n\n[4] WHAT THE NUMBERS MEAN AT THIS SIZE")
    push(f"    The test set is {len(yte)} sentences, so one sentence is {100 / len(yte):.2f} points of")
    push("    accuracy. Differences of two or three points between models are")
    push("    inside the noise and should not be reported as one model beating")
    push("    another. The random baseline for three balanced classes is 0.333.")

    push("\n[5] FILES WRITTEN")
    push(f"    reports/model_results.csv   all {len(res)} combinations")

    report = "\n".join(out)
    path = os.path.join(REPORT_DIR, "model_training_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {path}")
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()
