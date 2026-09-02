#!/usr/bin/env python3
"""
PERSON 3 (a) - merge the pair annotations and measure agreement.

Each pair of annotators labelled the same 200 sentences independently. This
script:

  * reads every filled sheet and normalises the labels (annotators typed
    "Neutral", "neutral" and "Positive " with a trailing space)
  * computes Cohen's Kappa within each pair - the number the rubric asks for
  * builds the adjudicated dataset: where a pair agrees, that is the label;
    where they disagree the row is flagged for the pair to resolve
  * reports how each annotator compares against the original corpus label

Pairs with only one annotator are reported but excluded from the adjudicated
set, because a single annotator gives no agreement to measure.

Usage:
    python scripts/collect_annotations.py
"""
import glob
import os
import re

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "data", "annotation")
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT_DIR = os.path.join(ROOT, "reports")
KEY = os.path.join(ANN, "annotation_KEY.xlsx")

LABELS = ["positive", "neutral", "negative"]
SHEET = "Annotation"
TEXT_COL = "ប្រយោគ  /  SENTENCE"

# filename -> (pair, annotator). Blank templates are skipped.
FILES = {
    "pair1_annotation_bath.xlsx": ("pair1", "Bath"),
    "pair1_annotation_label_Nacc.xlsx": ("pair1", "Nacc"),
    "Nita_pair2_annotation.xlsx": ("pair2", "Nita"),
    "Reaksa_pair2_annotation.xlsx": ("pair2", "Reaksa"),
    "pair3_seth.xlsx": ("pair3", "Seth"),
    "krisna_pair3.xlsx": ("pair3", "Krisna"),
}


def interpret(k: float) -> str:
    """Landis & Koch (1977)."""
    if k < 0:
        return "poor"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def normalise_label(v) -> str:
    """Annotators typed labels by hand, so casing and whitespace vary."""
    if pd.isna(v):
        return ""
    s = re.sub(r"\s+", " ", str(v)).strip().lower()
    return s if s in LABELS else ("" if not s else f"?{s}")


def load(path: str) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name=SHEET)
    text_col = TEXT_COL if TEXT_COL in d.columns else d.columns[1]
    return pd.DataFrame({
        "id": pd.to_numeric(d["id"], errors="coerce").astype("Int64"),
        "sentence": d[text_col].astype(str),
        "label": d["LABEL"].map(normalise_label),
    }).dropna(subset=["id"])


def main() -> None:
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    out, push = [], None
    push = out.append
    push("=" * 78)
    push("ANNOTATION AGREEMENT  -  Khmer sentiment corpus")
    push("=" * 78)

    # ------------------------------------------------------------- load
    frames = {}
    push("\n[1] FILES LOADED")
    for fname, (pair, who) in FILES.items():
        path = os.path.join(ANN, fname)
        if not os.path.exists(path):
            push(f"    [!] missing: {fname}")
            continue
        df = load(path)
        blank = int((df.label == "").sum())
        odd = sorted({l for l in df.label if l.startswith("?")})
        frames[(pair, who)] = df
        note = f"blank={blank}" if blank else "complete"
        if odd:
            note += f"  UNRECOGNISED={odd}"
        push(f"    {pair}  {who:8} {len(df):>4} rows   {note}")
        push(f"             {dict(df[df.label != ''].label.value_counts())}")

    push("\n    Note: labels were typed by hand, so 'Neutral', 'neutral' and")
    push("    'Positive ' with a trailing space all appeared. They are normalised")
    push("    to lowercase before any comparison - otherwise Cohen's Kappa would")
    push("    treat them as different categories and collapse to near zero.")

    # ------------------------------------------------- agreement per pair
    push("\n\n[2] COHEN'S KAPPA WITHIN EACH PAIR")
    push("    Two annotators, same 200 sentences, labelled independently.")
    pairs = sorted({p for p, _ in frames})
    adjudicated, pair_stats = [], []

    for pair in pairs:
        members = [(w, d) for (p, w), d in frames.items() if p == pair]
        if len(members) < 2:
            # Agreement needs two independent annotators by definition, so a
            # single-annotator pair is outside the scope of this analysis.
            push(f"\n    {pair}: single annotator on file - agreement requires two,")
            push(f"           so this pair is outside the scope of this analysis.")
            continue

        (w1, d1), (w2, d2) = members[0], members[1]
        m = d1.merge(d2, on="id", suffixes=("_1", "_2"))
        both = m[(m.label_1 != "") & (m.label_2 != "")]
        k = cohen_kappa_score(both.label_1, both.label_2, labels=LABELS)
        agree = (both.label_1 == both.label_2).mean()
        pair_stats.append((pair, w1, w2, len(both), k, agree))

        push(f"\n    {pair}   {w1} vs {w2}   ({len(both)} sentences)")
        push(f"      raw agreement : {agree:.1%}  ({int((both.label_1 == both.label_2).sum())}/{len(both)})")
        push(f"      Cohen's Kappa : {k:.3f}   ({interpret(k)})")

        cm = confusion_matrix(both.label_1, both.label_2, labels=LABELS)
        push(f"      confusion (rows={w1}, cols={w2}):")
        push("      " + pd.DataFrame(
            cm, index=[f"{w1[:6]}:{l[:3]}" for l in LABELS],
            columns=[f"{w2[:6]}:{l[:3]}" for l in LABELS]
        ).to_string().replace("\n", "\n      "))

        # adjudicated rows: agreement is the label, disagreement is flagged
        agreed = both[both.label_1 == both.label_2].copy()
        agreed["label"] = agreed.label_1
        agreed["status"] = "agreed"
        clash = both[both.label_1 != both.label_2].copy()
        clash["label"] = ""
        clash["status"] = "needs_adjudication"
        block = pd.concat([agreed, clash])
        block["pair"] = pair
        block["annotator_1"], block["annotator_2"] = w1, w2
        block = block.rename(columns={"sentence_1": "sentence"})
        adjudicated.append(block[["id", "pair", "sentence", "label", "status",
                                  "annotator_1", "annotator_2", "label_1", "label_2"]])

    if not pair_stats:
        raise SystemExit("No pair has two annotators - nothing to adjudicate.")

    mean_k = sum(s[4] for s in pair_stats) / len(pair_stats)
    push(f"\n    mean Cohen's Kappa across pairs : {mean_k:.3f}  ({interpret(mean_k)})")

    adj = pd.concat(adjudicated).sort_values("id").reset_index(drop=True)

    # ------------------------------------------------------ vs corpus label
    push("\n\n[3] EACH ANNOTATOR vs THE ORIGINAL CORPUS LABEL")
    if os.path.exists(KEY):
        key = pd.read_excel(KEY)[["id", "label"]].rename(columns={"label": "gold"})
        key["gold"] = key["gold"].map(normalise_label)
        push(f"    {'annotator':10} {'pair':7} {'kappa':>7} {'accuracy':>10}")
        for (pair, who), d in sorted(frames.items()):
            m = d.merge(key, on="id")
            m = m[m.label != ""]
            k = cohen_kappa_score(m.label, m.gold, labels=LABELS)
            push(f"    {who:10} {pair:7} {k:7.3f} {(m.label == m.gold).mean():9.1%}")
        push("\n    The corpus ships its own polarity label. We treat it as a")
        push("    reference point, not as truth: our annotators read full sentences")
        push("    in context, whereas the corpus label was assigned to a shorter cue")
        push("    phrase, so honest disagreement is expected.")
    else:
        push("    (annotation_KEY.xlsx not found - skipped)")

    # ---------------------------------------------------------- dataset
    push("\n\n[4] ADJUDICATED DATASET")
    n_agreed = int((adj.status == "agreed").sum())
    n_clash = int((adj.status == "needs_adjudication").sum())
    push(f"    sentences from pairs with two annotators : {len(adj)}")
    push(f"      both annotators agreed                 : {n_agreed}  ({n_agreed / len(adj):.1%})")
    push(f"      disagreed, needs the pair to resolve   : {n_clash}")
    push(f"\n    label distribution of the agreed rows:")
    push("      " + str(dict(adj[adj.status == "agreed"].label.value_counts())))

    push("\n    The agreed rows are the training data. Disagreements are left")
    push("    unlabelled rather than broken by a coin flip: the pair should settle")
    push("    them, and until then they are not evidence of anything.")

    adj.to_csv(os.path.join(PROCESSED, "adjudicated_annotations.csv"), index=False)
    gold_set = adj[adj.status == "agreed"][["id", "pair", "sentence", "label"]]
    gold_set.to_csv(os.path.join(PROCESSED, "labelled_dataset.csv"), index=False)

    # A second variant: the same 400 rows, with disagreements settled by the
    # corpus label instead of being dropped. 212 rows is thin for training seven
    # models, so both are produced and the choice is documented rather than
    # silently made here.
    resolved = None
    if os.path.exists(KEY):
        key = pd.read_excel(KEY)[["id", "label"]].rename(columns={"label": "gold"})
        key["gold"] = key["gold"].map(normalise_label)
        r = adj.merge(key, on="id", how="left")
        r["label"] = r.apply(
            lambda x: x["label"] if x["status"] == "agreed" else x["gold"], axis=1
        )
        r["resolution"] = r["status"].map(
            {"agreed": "annotators agreed", "needs_adjudication": "corpus label used"}
        )
        resolved = r[r.label != ""][["id", "pair", "sentence", "label", "resolution"]]
        resolved.to_csv(os.path.join(PROCESSED, "labelled_dataset_resolved.csv"), index=False)

    push("\n\n[5] TWO DATASET VARIANTS")
    push(f"    A. labelled_dataset.csv            {len(gold_set):>4} rows - only where both annotators agreed")
    push("       Cleanest signal, but small, and the surviving rows are the easy")
    push("       ones: keeping only agreements biases the set toward obvious cases.")
    if resolved is not None:
        push(f"    B. labelled_dataset_resolved.csv   {len(resolved):>4} rows - disagreements settled by the corpus label")
        push("       Nearly twice the data and keeps the hard cases, at the cost of")
        push("       leaning on an external label for 47% of rows.")
        push("       " + str(dict(resolved.label.value_counts())))
        push("\n    We train on B and report A as a robustness check. Person 5 should")
        push("    state which was used - the two are not interchangeable.")

    push("\n\n[6] FILES WRITTEN")
    push(f"    data/processed/adjudicated_annotations.csv   all {len(adj)} rows, with both votes")
    push(f"    data/processed/labelled_dataset.csv          {len(gold_set)} agreed rows")
    if resolved is not None:
        push(f"    data/processed/labelled_dataset_resolved.csv {len(resolved)} rows, disagreements resolved")

    report = "\n".join(out)
    path = os.path.join(REPORT_DIR, "annotation_agreement_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
