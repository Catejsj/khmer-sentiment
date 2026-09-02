#!/usr/bin/env python3
"""
PERSON 4 - Text Representation.

Builds all five representations the rubric asks for and saves them so Person 5
can load any one in a single line:

    Bag-of-Words   CountVectorizer, unigrams
    N-grams        CountVectorizer, unigrams + bigrams
    TF-IDF         TfidfVectorizer, unigrams + bigrams
    Word2Vec       gensim skip-gram, averaged into one vector per sentence
    fastText       gensim skip-gram + character n-grams, averaged

THE RULE THAT MATTERS: every vectoriser and embedding model is FIT ON TRAIN
ONLY, then applied to validation and test. Fitting on the whole dataset would
leak test vocabulary into the features and every score afterwards would be
optimistic.

KHMER NOTE: the text arriving here is already word-segmented by
scripts/preprocess.py, with tokens joined by spaces. That is why the vectorisers
use token_pattern=r"\\S+" - scikit-learn's default pattern assumes English word
characters and would silently mangle Khmer script.

Input : data/splits/{train,val,test}.csv
Output: data/features/<name>.npz  +  labels.npz

Usage:
    python scripts/representations.py
    python scripts/representations.py --tag agreed
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
FEATURES = os.path.join(ROOT, "data", "features")
REPORT_DIR = os.path.join(ROOT, "reports")

SPLIT_NAMES = ["train", "val", "test"]
TEXT_COL = "cleaned_text"
LABEL_COL = "label"

SEED = 42
EMB_DIM = 100
EMB_WINDOW = 5
EMB_EPOCHS = 30
MIN_DF = 2
# Khmer tokens are space-joined by preprocess.py; the sklearn default
# token_pattern would drop them entirely.
TOKEN_PATTERN = r"\S+"


def stable_hash(word: str) -> int:
    """gensim seeds vectors with Python's hash(), which is randomised per
    process, so seed=42 alone does not reproduce. Pin it."""
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)


def load_splits(suffix: str) -> dict:
    data = {}
    for name in SPLIT_NAMES:
        path = os.path.join(SPLITS, f"{name}{suffix}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}\nRun scripts/split_data.py first.")
        df = pd.read_csv(path)
        df[TEXT_COL] = df[TEXT_COL].fillna("")
        data[name] = df
    return data


def save_sparse(path, mats, vocab):
    np.savez_compressed(
        path,
        **{f"{s}_data": mats[s].data for s in SPLIT_NAMES},
        **{f"{s}_indices": mats[s].indices for s in SPLIT_NAMES},
        **{f"{s}_indptr": mats[s].indptr for s in SPLIT_NAMES},
        **{f"{s}_shape": np.array(mats[s].shape) for s in SPLIT_NAMES},
        vocab=np.array(vocab, dtype=object),
        kind=np.array("sparse"),
    )


def build_sparse(name, vectorizer, data, suffix, push):
    """Fit on train, transform all three splits."""
    mats = {"train": vectorizer.fit_transform(data["train"][TEXT_COL])}
    for s in ("val", "test"):
        mats[s] = vectorizer.transform(data[s][TEXT_COL])
    vocab = vectorizer.get_feature_names_out()
    save_sparse(os.path.join(FEATURES, f"{name}{suffix}.npz"), mats, vocab)

    tr = mats["train"]
    sparsity = 1 - tr.nnz / (tr.shape[0] * tr.shape[1])
    push(f"    {name:12} {tr.shape[1]:>7} features   train {str(tr.shape):>12}   sparsity {sparsity:6.2%}")
    return mats


def build_embedding(name, cls, data, suffix, push):
    """Train on TRAIN tokens only, then average word vectors per sentence."""
    train_docs = [t.split() for t in data["train"][TEXT_COL]]
    params = dict(sentences=train_docs, vector_size=EMB_DIM, window=EMB_WINDOW,
                  min_count=1, sg=1, workers=1, seed=SEED, epochs=EMB_EPOCHS,
                  hashfxn=stable_hash)
    if cls is FastText:
        params.update(min_n=3, max_n=6)
    model = cls(**params)
    # FastText's saved model carries the whole character n-gram matrix, which is
    # far larger than the vectors we actually need. Only Word2Vec is kept.
    if cls is not FastText:
        model.save(os.path.join(FEATURES, f"{name}{suffix}.model"))

    mats = {}
    for s in SPLIT_NAMES:
        docs = [t.split() for t in data[s][TEXT_COL]]
        mats[s] = np.vstack([
            model.wv.get_mean_vector(d, pre_normalize=True, post_normalize=True)
            if d else np.zeros(EMB_DIM, dtype=np.float32)
            for d in docs
        ]).astype(np.float32)
    np.savez_compressed(
        os.path.join(FEATURES, f"{name}{suffix}.npz"),
        **{s: mats[s] for s in SPLIT_NAMES}, kind=np.array("dense"),
    )

    vocab = set(model.wv.key_to_index)
    test_words = [w for t in data["test"][TEXT_COL] for w in t.split()]
    oov = sum(1 for w in test_words if w not in vocab) / max(len(test_words), 1)
    push(f"    {name:12} {EMB_DIM:>7} dims       train {str(mats['train'].shape):>12}   "
         f"vocab {len(vocab)}, test OOV {oov:.1%}")
    return mats, oov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    os.makedirs(FEATURES, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    data = load_splits(suffix)

    out, push = [], None
    push = out.append
    push("=" * 78)
    push("TEXT REPRESENTATION  -  five feature sets for the classifier")
    push("=" * 78)

    push("\n[1] INPUT")
    for s in SPLIT_NAMES:
        push(f"    {s:6} {len(data[s]):>4} sentences   {data[s][LABEL_COL].value_counts().to_dict()}")
    push("\n    Text is already word-segmented by scripts/preprocess.py, with tokens")
    push("    joined by spaces. The vectorisers use token_pattern=\\S+ because")
    push("    scikit-learn's default assumes English word characters and would drop")
    push("    Khmer script entirely.")
    push("\n    All five are FIT ON TRAIN ONLY and then applied to val and test.")

    push("\n[2] REPRESENTATIONS BUILT")
    push(f"    {'name':12} {'size':>7}            {'train shape':>12}")

    build_sparse("bow", CountVectorizer(ngram_range=(1, 1), min_df=MIN_DF,
                                        token_pattern=TOKEN_PATTERN), data, suffix, push)
    build_sparse("ngram", CountVectorizer(ngram_range=(1, 2), min_df=MIN_DF,
                                          token_pattern=TOKEN_PATTERN), data, suffix, push)
    build_sparse("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=MIN_DF,
                                          sublinear_tf=True,
                                          token_pattern=TOKEN_PATTERN), data, suffix, push)
    _, oov_w2v = build_embedding("word2vec", Word2Vec, data, suffix, push)
    _, oov_ft = build_embedding("fasttext", FastText, data, suffix, push)

    np.savez_compressed(
        os.path.join(FEATURES, f"labels{suffix}.npz"),
        **{s: data[s][LABEL_COL].to_numpy() for s in SPLIT_NAMES},
    )

    push(f"\n[3] WHY min_df={MIN_DF}")
    push("    A word appearing in only one sentence cannot generalise - it can only")
    push("    ever fire for that one example, so the model memorises instead of")
    push("    learning. On a corpus this small that matters more, not less.")

    push("\n[4] WHAT EACH ONE CAPTURES")
    push("    Bag-of-Words  which words appear, and how often")
    push("    N-grams       adds word pairs, so ' មិន ល្អ ' (not good) survives as one feature")
    push("    TF-IDF        down-weights words common to every sentence, lifts the")
    push("                  distinctive ones")
    push("    Word2Vec      meaning from context - words used similarly sit close")
    push("    fastText      same, plus character n-grams, so a word never seen in")
    push("                  training still gets a vector built from its pieces")
    push(f"\n    That last point is measurable here: {oov_w2v:.1%} of the words in the test")
    push("    set never appear in training. Word2Vec drops every one of them;")
    push("    fastText builds a vector from character chunks instead.")

    push("\n[5] FOR PERSON 5 - loading these")
    push("""
    from load_features import load

    X_train, X_val, X_test, y_train, y_val, y_test = load("tfidf")

    Names: bow, ngram, tfidf, word2vec, fasttext
    Sparse ones return scipy CSR matrices, embeddings return dense numpy arrays.
    Both feed scikit-learn estimators directly, so one loop covers all five.""")

    report = "\n".join(out)
    path = os.path.join(REPORT_DIR, f"representation_report{suffix}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {path}")
    print(f"Saved: {FEATURES}/*.npz")


if __name__ == "__main__":
    main()
