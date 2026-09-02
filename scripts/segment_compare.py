#!/usr/bin/env python3
"""
PERSON 3 (c) - Khmer word segmentation: dictionary vs CRF.

Khmer is written without spaces, so segmentation is the first real decision in
the pipeline. There are two established families of approach, and the teacher
asked for both to be understood:

  DICTIONARY (maximum matching)
      Walk left to right, and at each position take the longest string that
      appears in a Khmer wordlist. Deterministic, inspectable, and only as good
      as the dictionary. Anything absent from the wordlist - a new name, a
      loanword, a typo - cannot be matched and falls out as unknown.

  CRF (conditional random field)
      A statistical model trained on hand-segmented Khmer. For each position
      between two characters it predicts whether a word boundary belongs there,
      using the surrounding characters as features. It generalises to words it
      has never seen, at the cost of being a black box.

This script implements maximum matching over a real dictionary, runs both on our
corpus, and reports where they differ.

Dictionary: SIL NRSI khmerlbdict (MIT licence), 34,398 unique entries assembled
from frequency wordlists, name lists and place lists. See data/dictionary/.

Usage:
    python scripts/segment_compare.py
"""
import glob
import os
import re
import time

import pandas as pd
from khmernltk import word_tokenize as crf_tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_DIR = os.path.join(ROOT, "data", "dictionary")
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT_DIR = os.path.join(ROOT, "reports")

# A Khmer orthographic cluster is a base character plus any following marks.
# COENG (U+17D2) binds the next consonant to the current cluster, so a
# subscript consonant must never be split off on its own.
KHMER_BASE = r"[ក-អឥ-ឳ]"
KHMER_MARK = r"[឴-៑៝]"
COENG = "្"
CLUSTER_RE = re.compile(
    f"(?:{KHMER_BASE}(?:{COENG}{KHMER_BASE})*(?:{KHMER_MARK})*)|."
)


def clusters(text: str) -> list:
    """Split into orthographic clusters - the smallest safe unit to cut at."""
    return [m.group(0) for m in CLUSTER_RE.finditer(text) if m.group(0)]


def load_dictionary() -> tuple:
    words = set()
    for path in sorted(glob.glob(os.path.join(DICT_DIR, "*.txt"))):
        if "LICENSE" in path:
            continue
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                w = line.split("\t")[0].strip()
                if w:
                    words.add(w)
    if not words:
        raise SystemExit(f"No wordlists found in {DICT_DIR}")
    return words, max(len(w) for w in words)


def maximum_matching(text: str, words: set, max_len: int) -> list:
    """Greedy longest-match segmentation over orthographic clusters.

    At each position, try the longest candidate that is in the dictionary. If
    nothing matches, emit one cluster as an unknown token and move on - the
    standard fallback, and the source of most of this method's errors.
    """
    cl = clusters(text)
    out, i = [], 0
    while i < len(cl):
        matched = False
        # longest first; never look further ahead than the longest dictionary word
        for j in range(min(len(cl), i + max_len), i, -1):
            candidate = "".join(cl[i:j])
            if len(candidate) > 1 and candidate in words:
                out.append(candidate)
                i = j
                matched = True
                break
        if not matched:
            out.append(cl[i])
            i += 1
    return out


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    words, max_len = load_dictionary()

    src = os.path.join(PROCESSED, "labelled_dataset_resolved.csv")
    df = pd.read_csv(src)
    sentences = [str(s) for s in df["sentence"]]

    out, push = [], None
    push = out.append
    push("=" * 78)
    push("KHMER WORD SEGMENTATION  -  dictionary vs CRF")
    push("=" * 78)

    push("\n[1] THE PROBLEM")
    push("    Khmer is written without spaces between words:")
    push(f"      {sentences[0][:72]}")
    push(f"    That is {len(sentences[0].split())} whitespace-separated chunks. Before anything else")
    push("    can happen - counting words, building features, training a model -")
    push("    the boundaries have to be found.")

    push("\n[2] THE TWO APPROACHES")
    push("""
    DICTIONARY (maximum matching)
      Walk left to right; at each position take the longest string present in a
      Khmer wordlist. Deterministic and fully inspectable - you can point at the
      dictionary and say why a cut was made. But it can only find words the
      dictionary already contains.

    CRF (conditional random field)
      A model trained on hand-segmented Khmer. For each gap between characters
      it predicts boundary or no-boundary from the surrounding characters.
      Handles unseen words, but you cannot inspect why it cut where it did.""")

    push("\n[3] THE DICTIONARY")
    push(f"    source  : SIL NRSI khmerlbdict (MIT licence)")
    push(f"    entries : {len(words):,} unique words")
    push(f"    longest : {max_len} characters")
    push("    assembled from frequency wordlists, name lists and place lists;")
    push("    the files and licence are kept in data/dictionary/")

    # ------------------------------------------------------ run both
    t0 = time.time()
    dict_tokens = [maximum_matching(s, words, max_len) for s in sentences]
    t_dict = time.time() - t0

    t0 = time.time()
    crf_tokens = [crf_tokenize(s) for s in sentences]
    t_crf = time.time() - t0
    crf_tokens = [[t for t in toks if t.strip()] for toks in crf_tokens]

    push("\n[4] BOTH RUN ON OUR 400 SENTENCES")
    d_all = [t for toks in dict_tokens for t in toks]
    c_all = [t for toks in crf_tokens for t in toks]
    push(f"    {'':14}{'tokens':>9}{'vocabulary':>12}{'mean/sentence':>15}{'seconds':>10}")
    push(f"    {'dictionary':14}{len(d_all):>9}{len(set(d_all)):>12}{len(d_all) / len(sentences):>15.1f}{t_dict:>10.1f}")
    push(f"    {'CRF':14}{len(c_all):>9}{len(set(c_all)):>12}{len(c_all) / len(sentences):>15.1f}{t_crf:>10.1f}")

    # unknown = single-cluster tokens the dictionary could not match
    unknown = [t for t in d_all if len(clusters(t)) == 1 and re.match(KHMER_BASE, t or " ")]
    push(f"\n    dictionary tokens left unmatched : {len(unknown)}  ({len(unknown) / len(d_all):.1%})")
    push("    Those are positions where no dictionary word fit, so a single")
    push("    cluster was emitted. They are the visible cost of the dictionary")
    push("    approach - every one is a word the wordlist does not contain.")

    push("\n[5] WHERE THEY DISAGREE")
    diffs = [(i, d, c) for i, (d, c) in enumerate(zip(dict_tokens, crf_tokens)) if d != c]
    push(f"    sentences segmented identically : {len(sentences) - len(diffs)} / {len(sentences)}")
    push(f"    sentences segmented differently : {len(diffs)}")
    push("\n    Three examples, dictionary first:")
    for i, d, c in diffs[:3]:
        push(f"\n      sentence {i}")
        push(f"        dict : {' · '.join(d[:16])}")
        push(f"        CRF  : {' · '.join(c[:16])}")

    push("\n[6] WHICH WE USE, AND WHY")
    push("    We use the CRF (khmer-nltk) for the pipeline.")
    push("""
    The dictionary approach is easier to explain and to audit, and on a fixed
    vocabulary it is perfectly good. Our corpus is news text, which is full of
    proper nouns, place names, organisation names and numbers - exactly the
    material a fixed wordlist misses. Every unmatched position becomes a
    single-cluster fragment, and those fragments then become features, so the
    error propagates all the way to the classifier.

    The CRF was trained on hand-segmented Khmer and generalises to words it has
    not seen. We keep the dictionary implementation because it is the honest
    comparison: it shows what the CRF is buying us, and it is a fallback if the
    model is unavailable.""")

    push("\n[7] LIMITATION")
    push("    Neither output is checked against a gold segmentation - we have no")
    push("    hand-segmented Khmer of our own. The comparison shows the two")
    push("    methods differ and by how much, not which is correct. Measuring")
    push("    that would need a segmented reference such as the khPOS corpus.")

    report = "\n".join(out)
    path = os.path.join(REPORT_DIR, "segmentation_comparison.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
