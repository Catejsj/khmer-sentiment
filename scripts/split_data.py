#!/usr/bin/env python3
"""
Split the labelled dataset into train / validation / test.

Runs BEFORE representations and training. Everything downstream is only
leak-free if the test set was carved out first.

The dataset is small (400 sentences), so the split is stratified on the label to
keep all three classes present in every part, and the test share is kept at 20%
so there are enough test sentences to compute per-class metrics at all.

Usage:
    python scripts/split_data.py
    python scripts/split_data.py --csv data/processed/preprocessed_agreed.csv --tag agreed
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
SPLITS = os.path.join(ROOT, "data", "splits")
REPORT_DIR = os.path.join(ROOT, "reports")

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(PROCESSED, "preprocessed.csv"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--val-size", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(SPLITS, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    df = pd.read_csv(args.csv)
    df["cleaned_text"] = df["cleaned_text"].fillna("")

    out, push = [], None
    push = out.append
    push("=" * 74)
    push("TRAIN / VALIDATION / TEST SPLIT")
    push("=" * 74)
    push(f"\nsource: {args.csv}")
    push(f"dataset: {len(df)} sentences  {df.label.value_counts().to_dict()}")

    # empty rows cannot be represented or classified
    empty = int((df.cleaned_text.str.strip() == "").sum())
    if empty:
        push(f"\n[!] {empty} sentences are empty after cleaning - dropped")
        df = df[df.cleaned_text.str.strip() != ""]

    train_val, test = train_test_split(
        df, test_size=args.test_size, random_state=SEED, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val, test_size=args.val_size, random_state=SEED,
        stratify=train_val["label"]
    )

    push("\n[1] SPLITS  (stratified on label)")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        dist = part.label.value_counts().to_dict()
        push(f"    {name:6} {len(part):>4} sentences   {dist}")

    push("\n[2] LEAKAGE CHECKS")
    ids = {n: set(p["id"]) for n, p in [("train", train), ("val", val), ("test", test)]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        push(f"    {a} n {b} id overlap : {len(ids[a] & ids[b])}  (must be 0)")
    texts = pd.concat([train, val, test])["cleaned_text"]
    push(f"    duplicate cleaned texts : {int(texts.duplicated().sum())}")

    push("\n[3] A NOTE ON SIZE")
    push(f"    {len(train)} training sentences across three classes is small. Person 5")
    push("    should prefer cross-validation on train+val for model selection and")
    push("    keep the test set for a single final measurement, rather than tuning")
    push(f"    against {len(val)} validation sentences where one example moves the score")
    push(f"    by {100 / max(len(val), 1):.1f} points.")

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(os.path.join(SPLITS, f"{name}{suffix}.csv"), index=False)
    push("\n[4] FILES WRITTEN")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        push(f"    data/splits/{name}{suffix}.csv   {len(part)} rows")

    report = "\n".join(out)
    path = os.path.join(REPORT_DIR, f"split_report{suffix}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
