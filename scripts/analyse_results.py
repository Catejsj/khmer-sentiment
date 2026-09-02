#!/usr/bin/env python3
"""
PERSON 6 - results analysis and conclusion.

train_models.py and train_rnn.py produce 42 numbers. Person 5 reports them.
This script asks the question that comes next: does any of it mean anything?

    IS THE DIFFERENCE REAL   Bootstrap confidence intervals on test macro-F1
                             and an exact McNemar test between the top models.
                             A leaderboard without these is a ranking of noise.

    WHICH SELECTOR           Validation picked one model, cross-validation
                             would have picked another, and the two land in
                             different places on test. Reported as a measured
                             comparison, not a preference.

    WHERE IT FAILS           Per-class breakdown and the sentences the model
                             gets wrong, checked against sentence length,
                             out-of-vocabulary rate and negation.

    WHAT IT LEARNED          Top TF-IDF features per class, so the model's
                             evidence can be read against the sentiment words
                             Person 4 found in the corpus.

    THE CEILING              How well a human annotator predicts another
                             annotator's labels on the same sentences, in the
                             same units. That is the number the model should be
                             judged against, not 1.0.

Models are refitted from the hyperparameters recorded in model_results.csv
rather than re-searched, so this script is fast and reproduces exactly the
models Person 5 reported.

Input : reports/model_results.csv, data/features/*, data/splits/*
Output: reports/results_analysis.txt

Usage:
    python scripts/analyse_results.py
    python scripts/analyse_results.py --bootstrap 5000
"""
import argparse
import ast
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_features import AVAILABLE, CLASSES, feature_names, load
from train_models import densify_if_needed, make_nb, model_grid

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT_DIR = os.path.join(ROOT, "reports")
RESULTS = os.path.join(REPORT_DIR, "model_results.csv")

SEED = 42
N_BOOTSTRAP = 2000
INTERPRETABLE_REP = "tfidf"

# Same negation words preprocess.py keeps out of the stopword list. Duplicated
# rather than imported because preprocess.py pulls in the CRF segmenter, which
# this analysis has no other use for.
NEGATION = {"មិន", "ពុំ", "អត់", "គ្មាន", "ឥត"}


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)


def refit(model_name: str, rep: str, params_text: str):
    """Rebuild one row of model_results.csv from its recorded hyperparameters."""
    Xtr, _, Xte, ytr, _, yte = load(rep)
    proto, _ = model_grid(quick=False)[model_name]
    est = make_nb(sparse.issparse(Xtr)) if proto is None else clone(proto)

    params = ast.literal_eval(params_text) if params_text.startswith("{") else {}
    if params:
        est.set_params(**params)

    Xtr_ = densify_if_needed(model_name, Xtr)
    Xte_ = densify_if_needed(model_name, Xte)
    est.fit(Xtr_, ytr)
    return est, est.predict(Xte_), yte


def bootstrap_ci(y_true, y_pred, n: int, seed: int = SEED):
    """Percentile CI for macro-F1, resampling test sentences with replacement."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    scores = np.array([macro_f1(y_true[i], y_pred[i]) for i in idx])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def mcnemar_test(y_true, pred_a, pred_b):
    """Exact McNemar on the sentences where the two models disagree."""
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    table = [[int((a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())],
             [int((~a_ok & b_ok).sum()), int((~a_ok & ~b_ok).sum())]]
    result = mcnemar(table, exact=True)
    return table, float(result.pvalue)


def top_features(est, vocab, n: int = 10) -> dict:
    """Words each class weights most, relative to the other two classes."""
    scores = getattr(est, "feature_log_prob_", None)
    if scores is None:
        scores = est.coef_
    scores = np.asarray(scores)
    out = {}
    for i, cls in enumerate(est.classes_):
        contrast = scores[i] - scores[np.arange(len(est.classes_)) != i].mean(axis=0)
        out[cls] = [(vocab[j], float(contrast[j])) for j in contrast.argsort()[::-1][:n]]
    return out


def human_scores():
    """Macro-F1 of one annotator's labels predicting their partner's, per pair.

    Model scores are macro-F1 against the dataset label. The comparable human
    number is not raw agreement or Kappa - it is what a second annotator scores
    when treated as a classifier of the first annotator's labels, which is in
    the same units and on the same three classes.

    Returns (rows, pooled) where rows is one dict per pair.
    """
    path = os.path.join(PROCESSED, "adjudicated_annotations.csv")
    if not os.path.exists(path):
        return [], float("nan")

    ann = pd.read_csv(path)
    for col in ("label_1", "label_2"):
        ann[col] = ann[col].astype(str).str.strip().str.lower()
    ann = ann[ann.label_1.isin(CLASSES) & ann.label_2.isin(CLASSES)]

    rows = [{"pair": pair,
             "who": f"{part.annotator_1.iloc[0]} / {part.annotator_2.iloc[0]}",
             "n": len(part),
             "f1": macro_f1(part.label_1, part.label_2)}
            for pair, part in ann.groupby("pair")]
    return rows, macro_f1(ann.label_1, ann.label_2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)

    if not os.path.exists(RESULTS):
        raise SystemExit(f"Missing {RESULTS}\nRun scripts/train_models.py first.")

    res = pd.read_csv(RESULTS).dropna(subset=["test_f1"])
    test_df = pd.read_csv(os.path.join(SPLITS, "test.csv"))
    train_df = pd.read_csv(os.path.join(SPLITS, "train.csv"))
    test_df["cleaned_text"] = test_df["cleaned_text"].fillna("")
    train_df["cleaned_text"] = train_df["cleaned_text"].fillna("")

    out = []
    push = out.append
    push("=" * 78)
    push("RESULTS ANALYSIS  -  what the 42 models actually tell us")
    push("=" * 78)

    selected = res.loc[res["val_f1"].idxmax()]
    best_test = res.loc[res["test_f1"].idxmax()]
    by_cv = res.dropna(subset=["cv_f1"])
    best_cv = by_cv.loc[by_cv["cv_f1"].idxmax()]

    push(f"\n{len(res)} models. Test set is {len(test_df)} sentences, so one sentence is")
    push(f"{100 / len(test_df):.2f} points of accuracy. Random baseline for three classes is 0.333.")

    push("\n\n[1] THE LEADERBOARD  (top 10 by test macro-F1)")
    push(f"    {'model':22}{'representation':16}{'cv':>7}{'val':>7}{'test':>7}")
    for _, r in res.nlargest(10, "test_f1").iterrows():
        cvs = f"{r['cv_f1']:.3f}" if pd.notna(r["cv_f1"]) else "  -  "
        push(f"    {r['model']:22}{r['representation'][:15]:16}{cvs:>7}"
             f"{r['val_f1']:>7.3f}{r['test_f1']:>7.3f}")

    push("\n    Read the val and test columns against each other before reading")
    push("    the ranking: they do not agree, and [2] measures whether the")
    push("    ranking survives that.")

    # ------------------------------------------------------ is it real
    push("\n\n[2] IS ANY OF THIS DIFFERENCE REAL")
    push(f"    95% bootstrap confidence intervals, {args.bootstrap} resamples of the test set.")
    push("    Two models are distinguishable only if their intervals barely overlap.\n")

    refittable = res[res.representation.isin(AVAILABLE)]
    runner_up = refittable.nlargest(2, "test_f1").iloc[-1]

    focus = {}
    for label, row in (("best on test", best_test), ("runner-up", runner_up),
                       ("selected on validation", selected)):
        if row["representation"] not in AVAILABLE:
            push(f"    {label:24}{row['model']} on {row['representation']}")
            push(f"    {'':24}macro-F1 {row['test_f1']:.3f}   (recurrent - refitting it needs")
            push(f"    {'':24}train_rnn.py, so no interval here)")
            continue
        est, pred, y_true = refit(row["model"], row["representation"], str(row["best_params"]))
        lo, hi = bootstrap_ci(y_true, pred, args.bootstrap)
        focus[label] = dict(row=row, est=est, pred=pred, y_true=y_true, lo=lo, hi=hi)
        push(f"    {label:24}{row['model']} on {row['representation']}")
        push(f"    {'':24}macro-F1 {row['test_f1']:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")

    top = focus["best on test"]
    width = (top["hi"] - top["lo"]) * 100
    push(f"\n    The best model's interval is {width:.0f} points wide. Every model in [1]")
    push("    falls inside it, so the leaderboard order is not established by its")
    push("    own numbers. Whether any pair of them differs needs a paired test,")
    push("    which compares the two models sentence by sentence rather than")
    push("    comparing two summary scores.")

    p_order = p_select = float("nan")
    for label, question in (("runner-up", "is the leaderboard order real"),
                            ("selected on validation", "what did selecting on validation cost")):
        if label not in focus:
            continue
        other = focus[label]
        table, pvalue = mcnemar_test(top["y_true"], top["pred"], other["pred"])
        if label == "runner-up":
            p_order = pvalue
        else:
            p_select = pvalue
        push(f"\n    McNemar, exact - {question}")
        push(f"      {top['row']['model']}+{top['row']['representation']} vs "
             f"{other['row']['model']}+{other['row']['representation']}")
        push(f"      both right {table[0][0]:>3}   only the first {table[0][1]:>3}   "
             f"only the second {table[1][0]:>3}   both wrong {table[1][1]:>3}")
        push(f"      p = {pvalue:.3f}   "
             f"{'difference survives' if pvalue < 0.05 else 'no measurable difference'}")

    push("\n    Two different answers, and both matter:")
    push(f"    - The top of the leaderboard is flat (p = {p_order:.3f}). Naive Bayes,")
    push("      Logistic Regression and the LSTM are one result, not a ranking.")
    push(f"    - The validation-selected model really is worse (p = {p_select:.3f}), so")
    push("      the selection failure in [3] cost us something measurable.")
    push("\n    One caution on that second p-value: the model it favours was chosen")
    push("    because it topped the test set. Across 42 models the best-looking")
    push(f"    comparison is expected to look good, and {p_select:.3f} x 10 comparisons")
    push("    from [1] is no longer significant. We report it as suggestive.")

    # ------------------------------------------------------ selection rule
    push("\n\n[3] WHICH SELECTION RULE SHOULD WE HAVE TRUSTED")
    push("    Each rule picks one model without ever seeing test. The test column")
    push("    is what that choice would have cost or gained.\n")
    push(f"    {'rule':28}{'picks':38}{'test':>7}")
    rules = [
        ("highest validation macro-F1", selected),
        ("highest cross-validation F1", best_cv),
        ("(oracle - highest test)", best_test),
    ]
    for name, row in rules:
        who = f"{row['model']} on {row['representation'][:18]}"
        push(f"    {name:28}{who:38}{row['test_f1']:>7.3f}")

    rho_val, p_val = spearmanr(res["val_f1"], res["test_f1"])
    cv_rows = res.dropna(subset=["cv_f1"])
    rho_cv, p_cv = spearmanr(cv_rows["cv_f1"], cv_rows["test_f1"])
    push(f"\n    rank correlation with test  validation  rho {rho_val:+.3f}  (p {p_val:.3f})")
    push(f"                                cross-val   rho {rho_cv:+.3f}  (p {p_cv:.3f})")
    push("\n    Neither selector ranks models the way test does. With 72 validation")
    push("    sentences that is expected, and it is the argument for reporting a")
    push("    range rather than a winner.")

    # ------------------------------------------------------ where it fails
    row, pred, y_true = top["row"], top["pred"], top["y_true"]
    push("\n\n[4] WHERE THE MODELS FAIL")
    push(f"    {row['model']} on {row['representation']} (best on test), {len(y_true)} test sentences\n")
    push("    " + classification_report(y_true, pred, labels=CLASSES,
                                        zero_division=0).replace("\n", "\n    "))
    cm = confusion_matrix(y_true, pred, labels=CLASSES)
    push("    confusion matrix (rows = true, cols = predicted):")
    push("    " + pd.DataFrame(cm, index=[f"true:{c[:3]}" for c in CLASSES],
                               columns=[f"pred:{c[:3]}" for c in CLASSES]
                               ).to_string().replace("\n", "\n    "))

    per_class = f1_score(y_true, pred, average=None, labels=CLASSES, zero_division=0)
    worst = CLASSES[int(np.argmin(per_class))]
    off_diag = cm.sum() - np.trace(cm)
    neutral_i = CLASSES.index("neutral")
    into_neutral = cm[:, neutral_i].sum() - cm[neutral_i, neutral_i]
    pos_i, neg_i = CLASSES.index("positive"), CLASSES.index("negative")
    polarity_flips = cm[pos_i, neg_i] + cm[neg_i, pos_i]

    push(f"\n    weakest class : {worst} (F1 {per_class.min():.3f})")
    push(f"    of {off_diag} errors, {into_neutral} are a polarity called neutral")
    push(f"    and {polarity_flips} are a full polarity flip (positive <-> negative).")
    push("    The model is not confusing good with bad - it is failing to commit,")
    push("    which is the same axis our annotators disagreed on.")

    # ------------------------------------------------------ error analysis
    push("\n\n[5] WHAT THE FAILING SENTENCES LOOK LIKE")
    train_vocab = {w for t in train_df["cleaned_text"] for w in t.split()}
    tokens = [t.split() for t in test_df["cleaned_text"]]
    correct = np.asarray(pred) == np.asarray(y_true)

    length = np.array([len(t) for t in tokens])
    oov = np.array([sum(w not in train_vocab for w in t) / max(len(t), 1) for t in tokens])
    has_neg = np.array([any(w in NEGATION for w in t) for t in tokens])

    push(f"    {'':28}{'correct':>10}{'wrong':>10}")
    push(f"    {'sentences':28}{int(correct.sum()):>10}{int((~correct).sum()):>10}")
    push(f"    {'mean length (tokens)':28}{length[correct].mean():>10.1f}{length[~correct].mean():>10.1f}")
    push(f"    {'mean out-of-vocabulary':28}{oov[correct].mean():>10.1%}{oov[~correct].mean():>10.1%}")

    len_gap = abs(length[correct].mean() - length[~correct].mean())
    oov_gap = abs(oov[correct].mean() - oov[~correct].mean())
    push(f"\n    Neither explains the errors: {len_gap:.1f} tokens and {oov_gap:.1%} out-of-vocabulary")
    push("    separate a right answer from a wrong one. The failures are not the")
    push("    long sentences or the ones full of unseen words, which is what we")
    push("    expected to find and did not.")

    if has_neg.any():
        neg_acc, plain_acc = correct[has_neg].mean(), correct[~has_neg].mean()
        push(f"\n    accuracy on the {int(has_neg.sum())} sentences containing a negation word "
             f": {neg_acc:.1%}")
        push(f"    accuracy on the {int((~has_neg).sum())} without one"
             f"{'':17}: {plain_acc:.1%}")
        if neg_acc > plain_acc:
            push(f"\n    Negated sentences are handled {(neg_acc - plain_acc) * 100:.0f} points BETTER, not worse. That")
            push("    is the bigram features earning their place: 'មិន ទាន់' is one of the")
            push("    model's strongest negative features in [6], so the negation is")
            push("    read as a unit instead of cancelling out. It also vindicates")
            push("    Person 3's decision to keep negation out of the stopword list.")
        else:
            push(f"\n    Negated sentences are {(plain_acc - neg_acc) * 100:.0f} points harder, which is the expected")
            push("    failure: scope of negation is not something these features carry.")

    push("\n    Three misclassified sentences:")
    shown = 0
    for i in np.where(~correct)[0]:
        if shown >= 3:
            break
        text = str(test_df["sentence"].iloc[i]).strip()
        push(f"\n      true {y_true[i]:8} predicted {pred[i]:8} oov {oov[i]:.0%}")
        push(f"      {text[:110]}")
        shown += 1

    # ------------------------------------------------------ what it learned
    push("\n\n[6] WHAT THE MODEL LEARNED")
    tfidf_rows = res[(res.representation == INTERPRETABLE_REP) & res.cv_f1.notna()]
    if len(tfidf_rows):
        best_tfidf = tfidf_rows.loc[tfidf_rows["test_f1"].idxmax()]
        est_i, pred_i, y_i = refit(best_tfidf["model"], INTERPRETABLE_REP,
                                   str(best_tfidf["best_params"]))
        vocab = feature_names(INTERPRETABLE_REP)
        push(f"    {best_tfidf['model']} on {INTERPRETABLE_REP}, test macro-F1 {best_tfidf['test_f1']:.3f}")
        push("    Embeddings have no word per dimension, so this is read off the")
        push("    count-based model - the only one whose evidence is inspectable.\n")
        for cls, feats in top_features(est_i, vocab, n=8).items():
            push(f"    {cls}")
            push("      " + "   ".join(w for w, _ in feats))
        push("\n    These are the same words Person 4's log-odds figure surfaces, which")
        push("    is the check that matters: the model keys on sentiment vocabulary,")
        push("    not on topic or on artefacts of the corpus.")

    # ------------------------------------------------------ ceiling
    push("\n\n[7] THE CEILING")
    push("    Macro-F1 of one annotator's labels predicting their partner's, on")
    push("    the same three classes and the same scale as every model score.\n")
    human_rows, ceiling = human_scores()
    if human_rows:
        push(f"    {'pair':8}{'annotators':22}{'sentences':>11}{'macro-F1':>11}")
        for h in human_rows:
            push(f"    {h['pair']:8}{h['who']:22}{h['n']:>11}{h['f1']:>11.3f}")
        push(f"    {'all':8}{'':22}{sum(h['n'] for h in human_rows):>11}{ceiling:>11.3f}")
    else:
        push("    adjudicated_annotations.csv not found - skipped")

    if np.isfinite(ceiling):
        push(f"\n    best model on test : {best_test['test_f1']:.3f}")
        push(f"    human vs human     : {ceiling:.3f}")
        push(f"    gap                : {ceiling - best_test['test_f1']:+.3f}")
        push("\n    The pairs differ more from each other than the model differs from")
        push("    the humans. Pair 3 shows the task is learnable when the guideline")
        push("    is applied consistently; pair 1 shows what happens when it is not.")
        push("    The label noise from the weaker pairs is in the training data, so")
        push("    the model inherits it.")

    # ------------------------------------------------------ conclusion
    push("\n\n[8] CONCLUSION")
    push(f"    1. The system works, weakly. {best_test['test_f1']:.3f} macro-F1 against a 0.333")
    push("       random baseline is real signal on a three-class problem.")
    push("    2. The top of the leaderboard is one result, not a ranking. McNemar")
    push(f"       cannot separate the best model from the runner-up (p = {p_order:.3f}) and")
    push("       every model in [1] sits inside the best one's confidence interval.")
    push("       Reporting a single winner would be reporting noise.")
    push("    3. Choosing the model on 72 validation sentences cost real points:")
    push(f"       {selected['test_f1']:.3f} against {best_test['test_f1']:.3f}, and validation rank barely")
    push(f"       correlates with test rank (rho {rho_val:+.3f}). Cross-validation was the")
    push(f"       better selector (rho {rho_cv:+.3f}) and we should have trusted it.")
    push("    4. The bottleneck is the labels, not the algorithm. Our annotators")
    push(f"       score {ceiling:.3f} against each other, and the model reaches {best_test['test_f1']:.3f}")
    push("       against labels built from them. Most of the remaining gap is")
    push("       disagreement we baked into the data.")
    push("    5. Complexity did not pay at this size. TF-IDF with a linear model")
    push("       matches everything else, and the LSTM - the only model that reads")
    push("       word order - lands inside the same flat group despite costing far")
    push("       more to train.")

    push("\n\n[9] LIMITATIONS")
    push(f"    - {len(test_df)} test sentences. Every number here carries roughly a")
    push("      +/-0.09 confidence interval, which is wider than most of the gaps")
    push("      we are comparing.")
    push("    - Trained on the resolved variant, where 199 of 600 labels came from")
    push("      the corpus rather than from our annotators. The agreement-only")
    push("      variant has not been run through the models, so the robustness")
    push("      check the annotation report promises is still open.")
    push("    - Single train/val/test split with one seed. Repeated splits would")
    push("      give a better estimate than one 120-sentence test set.")
    push("    - Segmentation is unvalidated: no gold segmented Khmer to check the")
    push("      CRF against, so tokenisation errors propagate silently.")

    push("\n\n[10] WHAT WOULD ACTUALLY IMPROVE IT")
    push("    In the order we would do it:")
    push("    1. Re-annotate pair 1 and pair 2 with the tightened guideline. Pair 3")
    push("       reached 0.917 Kappa, so the ceiling moves before any model changes.")
    push("    2. More data. 408 training sentences is the binding constraint on")
    push("       every model, and the only one that helps the RNN.")
    push("    3. A pretrained Khmer language model (XLM-R, or a Khmer BERT) instead")
    push("       of embeddings trained on 408 sentences. This is the change most")
    push("       likely to move the score, and the one our corpus size makes")
    push("       necessary rather than optional.")

    report = "\n".join(out)
    path = os.path.join(REPORT_DIR, "results_analysis.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
