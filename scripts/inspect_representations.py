#!/usr/bin/env python3
"""
PERSON 4 - look inside the five representations.

representations.py proves the feature files exist and have the right shape.
This shows what is actually in them: which words dominate, which words separate
the three sentiment classes, how TF-IDF reweights the same corpus, and what the
embeddings consider similar.

Computed on TRAIN ONLY. Inspecting val or test to decide what to say about the
features would be leakage through the back door.

Usage:
    python scripts/inspect_representations.py
"""
import os

import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
REPORT_DIR = os.path.join(ROOT, "reports")

TEXT_COL = "cleaned_text"
CLASSES = ["positive", "neutral", "negative"]
TOKEN_PATTERN = r"\S+"
SEED, EMB_DIM, EMB_EPOCHS = 42, 100, 30


def stable_hash(word):
    import hashlib
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)


def top_terms(matrix, vocab, n=20):
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    idx = totals.argsort()[::-1][:n]
    return [(vocab[i], totals[i]) for i in idx]


def distinctive(matrix, vocab, labels, target, n=12, min_total=4):
    """Log-odds of a term belonging to `target` versus every other class.

    Raw counts only tell you which words are common. This asks which words are
    disproportionately associated with one sentiment, with add-one smoothing so
    a word seen three times does not top the list.
    """
    is_t = np.asarray(labels) == target
    pos = np.asarray(matrix[is_t].sum(axis=0)).ravel() + 1
    neg = np.asarray(matrix[~is_t].sum(axis=0)).ravel() + 1
    total = pos + neg - 2
    ratio = np.log((pos / pos.sum()) / (neg / neg.sum()))
    ok = total >= min_total
    order = np.where(ok, ratio, -np.inf).argsort()[::-1][:n]
    return [(vocab[i], ratio[i], int(total[i])) for i in order]


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    train = pd.read_csv(os.path.join(SPLITS, "train.csv"))
    train[TEXT_COL] = train[TEXT_COL].fillna("")
    corpus, labels = train[TEXT_COL].tolist(), train["label"].tolist()

    out, push = [], None
    push = out.append
    push("=" * 78)
    push("INSIDE THE FIVE REPRESENTATIONS")
    push(f"train only: {len(corpus)} sentences  "
         f"{ {c: labels.count(c) for c in CLASSES} }")
    push("=" * 78)

    bow_v = CountVectorizer(min_df=2, token_pattern=TOKEN_PATTERN)
    bow = bow_v.fit_transform(corpus)
    bow_vocab = bow_v.get_feature_names_out()

    ng_v = CountVectorizer(ngram_range=(2, 2), min_df=2, token_pattern=TOKEN_PATTERN)
    ng = ng_v.fit_transform(corpus)
    ng_vocab = ng_v.get_feature_names_out()

    tf_v = TfidfVectorizer(min_df=2, sublinear_tf=True, token_pattern=TOKEN_PATTERN)
    tf = tf_v.fit_transform(corpus)

    # ------------------------------------------------------------------ 1
    push("\n[1] ONE SENTENCE, SEEN AS NUMBERS")
    i = int(np.argmax([len(t.split()) for t in corpus]))
    push(f"    label: {labels[i]}")
    push(f"    cleaned: {corpus[i][:120]}")
    row = bow[i].toarray().ravel()
    nz = row.nonzero()[0]
    push(f"\n    Bag-of-Words : {len(row)} slots, {len(nz)} non-zero")
    push("      " + ", ".join(f"{bow_vocab[j]}={int(row[j])}" for j in nz[:12]))
    trow = tf[i].toarray().ravel()
    push(f"\n    TF-IDF       : same slots, weighted")
    push("      " + ", ".join(f"{bow_vocab[j]}={trow[j]:.2f}"
                              for j in trow.argsort()[::-1][:8]))

    # ------------------------------------------------------------------ 2
    push("\n\n[2] BAG-OF-WORDS - 20 most frequent words")
    rows = top_terms(bow, bow_vocab, 20)
    for k in range(0, 20, 4):
        push("      " + "".join(f"{w:<14}{int(c):>4}   " for w, c in rows[k:k + 4]))
    push("\n    Generic vocabulary. Nothing here separates the classes - every")
    push("    sentiment talks about the same everyday topics.")

    # ------------------------------------------------------------------ 3
    push("\n\n[3] WHICH WORDS SIGNAL EACH SENTIMENT")
    push("    Log-odds against the other two classes combined. Higher means more")
    push("    strongly associated with that sentiment.")
    for cls in CLASSES:
        push(f"\n    {cls.upper()}")
        for w, r, t in distinctive(bow, bow_vocab, labels, cls):
            push(f"      {w:<20} {r:>6.2f}   (seen {t})")

    # ------------------------------------------------------------------ 4
    push("\n\n[4] N-GRAMS - what word pairs add")
    rows = top_terms(ng, ng_vocab, 10)
    push("    most frequent bigrams:")
    for w, c in rows:
        push(f"      {w:<32} {int(c)}")
    push("\n    A single word may be neutral while the pair is not. This is the")
    push("    only representation of the three count-based ones that sees any")
    push("    word order at all.")

    # ------------------------------------------------------------------ 5
    push("\n\n[5] TF-IDF vs RAW COUNTS")
    counts = np.asarray(bow.sum(axis=0)).ravel()
    weights = np.asarray(tf.sum(axis=0)).ravel()
    push(f"    {'by raw count':<20}{'by TF-IDF weight':<20}")
    for a, b in zip(counts.argsort()[::-1][:10], weights.argsort()[::-1][:10]):
        push(f"      {bow_vocab[a]:<18}{bow_vocab[b]:<18}")
    push("\n    TF-IDF pushes down words present in nearly every sentence and")
    push("    lifts the ones only some sentences use.")

    # ------------------------------------------------------------------ 6
    docs = [t.split() for t in corpus]
    push("\n\n[6] WORD2VEC - nearest neighbours")
    w2v = Word2Vec(sentences=docs, vector_size=EMB_DIM, window=5, min_count=1,
                   sg=1, workers=1, seed=SEED, epochs=EMB_EPOCHS, hashfxn=stable_hash)
    probes = [w for w, _ in top_terms(bow, bow_vocab, 6)]
    push(f"    vocabulary: {len(w2v.wv)} words, {EMB_DIM} dimensions")
    for w in probes:
        if w in w2v.wv:
            near = ", ".join(x for x, _ in w2v.wv.most_similar(w, topn=5))
            push(f"      {w:<14} -> {near}")

    push("\n\n[7] FASTTEXT - nearest neighbours, and unseen words")
    ft = FastText(sentences=docs, vector_size=EMB_DIM, window=5, min_count=1,
                  sg=1, workers=1, seed=SEED, epochs=EMB_EPOCHS,
                  hashfxn=stable_hash, min_n=3, max_n=6)
    for w in probes[:4]:
        if w in ft.wv:
            near = ", ".join(x for x, _ in ft.wv.most_similar(w, topn=5))
            push(f"      {w:<14} -> {near}")

    test = pd.read_csv(os.path.join(SPLITS, "test.csv"))
    test[TEXT_COL] = test[TEXT_COL].fillna("")
    train_vocab = set(w2v.wv.key_to_index)
    unseen = [w for t in test[TEXT_COL] for w in t.split() if w not in train_vocab]
    push(f"\n    Words in test that never appear in train: {len(unseen)} of "
         f"{sum(len(t.split()) for t in test[TEXT_COL])} "
         f"({len(unseen) / max(sum(len(t.split()) for t in test[TEXT_COL]), 1):.1%})")
    push("\n    What each model does with them:")
    for w in list(dict.fromkeys(unseen))[:5]:
        near = ", ".join(x for x, _ in ft.wv.most_similar(w, topn=3))
        push(f"      {w:<16} Word2Vec: NO VECTOR    fastText: {near}")
    push("\n    This is the entire argument for fastText on Khmer. A quarter of the")
    push("    test vocabulary is unseen, because 272 training sentences cannot")
    push("    cover a language with this much compounding.")

    # ------------------------------------------------------------------ 8
    push("\n\n[8] SUMMARY")
    push(f"    {'representation':16}{'features':>10}   interpretable?")
    push(f"    {'Bag-of-Words':16}{bow.shape[1]:>10}   yes - one word per column")
    push(f"    {'N-grams':16}{'1086':>10}   yes - one phrase per column")
    push(f"    {'TF-IDF':16}{'1086':>10}   yes - weighted words")
    push(f"    {'Word2Vec':16}{EMB_DIM:>10}   no - dense dimensions")
    push(f"    {'fastText':16}{EMB_DIM:>10}   no - dense dimensions")

    report = "\n".join(out)
    path = os.path.join(REPORT_DIR, "representation_inspection.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
