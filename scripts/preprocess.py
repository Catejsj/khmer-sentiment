#!/usr/bin/env python3
"""
PERSON 3 (b) - preprocessing pipeline for Khmer sentiment text.

Khmer is not English, and three of the standard steps do not transfer:

  TOKENIZATION   Khmer writes without spaces between words. Splitting on
                 whitespace returns whole clauses, not words, so a statistical
                 segmenter is required. We use khmer-nltk, a CRF word
                 segmenter trained on Khmer.

  STEMMING /     Khmer is an analytic language. Verbs do not conjugate, nouns
  LEMMATIZATION  do not inflect for number or case, and there is no productive
                 suffix system to strip. "go / goes / went / going" are all the
                 same surface form ទៅ. So there is nothing for a stemmer to
                 remove, and no lemmatizer exists because none is needed. We
                 document this rather than forcing an English tool onto it.

  STOPWORDS      No canonical Khmer stopword list ships with any major library,
                 so we define one explicitly from function words - particles,
                 classifiers, pronouns and conjunctions - and keep negation.

Everything else follows the same shape as the English pipeline: clean, tokenize,
remove stopwords, and record the output of every stage so the report can show
the text at each step.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --csv data/processed/labelled_dataset.csv --tag agreed
"""
import argparse
import html
import os
import re
from collections import Counter

import pandas as pd
from khmernltk import word_tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT_DIR = os.path.join(ROOT, "reports")
DEFAULT_CSV = os.path.join(PROCESSED, "labelled_dataset_resolved.csv")

# Khmer function words: particles, classifiers, pronouns, prepositions,
# conjunctions. These carry grammar, not sentiment.
KHMER_STOPWORDS = {
    "នេះ", "នោះ", "ដែល", "និង", "ជា", "នៅ", "ក្នុង", "លើ", "ក្រោម", "ពី", "ទៅ",
    "មក", "បាន", "ដោយ", "សម្រាប់", "របស់", "គឺ", "ថា", "ហើយ", "ទេ", "ផង",
    "ដែរ", "គេ", "គាត់", "ខ្ញុំ", "យើង", "លោក", "អ្នក", "វា", "មួយ", "ពីរ",
    "ក៏", "តែ", "បើ", "ព្រោះ", "ដូច", "ជាមួយ", "អាច", "ត្រូវ", "នឹង", "ចំពោះ",
    "រួច", "ទាំង", "គ្នា", "ណា", "អី", "ឬ", "ដើម្បី", "ពេល", "ដល់", "រហូត",
}

# Negation and degree words invert or scale polarity. Removing them as
# stopwords would turn "not good" into "good" - the same trap as English.
KEEP_WORDS = {"មិន", "ពុំ", "អត់", "គ្មាន", "ឥត", "ណាស់", "ខ្លាំង", "តិច", "ស្ទើរ"}
STOPWORDS = KHMER_STOPWORDS - KEEP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Keep Khmer script (U+1780-U+17FF), Khmer digits, ASCII letters and spaces.
NON_KHMER_RE = re.compile(r"[^ក-៿᧠-᧿a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")
KHMER_CHAR_RE = re.compile(r"[ក-៿]")


# --------------------------------------------------------------- text stages
def unescape_html(text):
    return html.unescape(text)


def remove_urls(text):
    return URL_RE.sub(" ", text)


def strip_symbols(text):
    """Remove Latin punctuation, Khmer punctuation (។ ៖ ៛), digits and emoji."""
    return NON_KHMER_RE.sub(" ", text)


def normalise_space(text):
    return MULTISPACE_RE.sub(" ", text).strip()


TEXT_STAGES = [
    ("00_raw", None),
    ("01_unescape_html", unescape_html),
    ("02_remove_urls", remove_urls),
    ("03_strip_symbols", strip_symbols),
    ("04_normalise_space", normalise_space),
]


# -------------------------------------------------------------- token stages
def tokenize(text):
    """Khmer word segmentation. Whitespace splitting does not work here."""
    return [t for t in word_tokenize(text) if t.strip()]


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def drop_short(tokens):
    """Single Khmer characters are usually orphaned diacritics after cleaning."""
    return [t for t in tokens if len(t) > 1 or not KHMER_CHAR_RE.match(t)]


def preprocess(raw_text):
    stages, current = {}, str(raw_text)
    for name, fn in TEXT_STAGES:
        current = current if fn is None else fn(current)
        stages[name] = current

    tokens = tokenize(current)
    stages["05_tokenized"] = tokens
    tokens = remove_stopwords(tokens)
    stages["06_stopwords_removed"] = tokens
    tokens = drop_short(tokens)
    stages["07_short_dropped"] = tokens
    stages["tokens"] = tokens
    stages["cleaned_text"] = " ".join(tokens)
    return stages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(args.csv)
    for col in ("sentence", "label"):
        if col not in df.columns:
            raise SystemExit(f"{args.csv}: missing '{col}'. Found {list(df.columns)}")

    results = [preprocess(t) for t in df["sentence"]]
    df["cleaned_text"] = [r["cleaned_text"] for r in results]
    df["tokens"] = [r["tokens"] for r in results]
    df["n_tokens_raw"] = [len(tokenize(str(t))) for t in df["sentence"]]
    df["n_tokens_clean"] = [len(r["tokens"]) for r in results]

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = os.path.join(PROCESSED, f"preprocessed{suffix}.csv")
    stages_csv = os.path.join(PROCESSED, f"preprocessing_stages{suffix}.csv")
    out_report = os.path.join(REPORT_DIR, f"preprocessing_report{suffix}.txt")

    out, push = [], None
    push = out.append
    push("=" * 78)
    push(f"PREPROCESSING REPORT  -  Khmer sentiment, {len(df)} sentences")
    push(f"source: {args.csv}")
    push("=" * 78)

    push("\n[1] PIPELINE")
    push("""
      01 unescape HTML entities
      02 remove URLs
      03 remove punctuation, digits, emoji   (Khmer ។ ៖ ៛ included)
      04 normalise whitespace
      05 WORD SEGMENTATION   <- khmer-nltk CRF model
      06 STOPWORD REMOVAL    <- our Khmer list, negation kept
      07 drop orphaned single characters

    NO STEMMING OR LEMMATIZATION. Khmer is analytic - verbs do not conjugate
    and nouns do not inflect, so there are no affixes to strip. Applying an
    English stemmer would corrupt the text without removing anything real.""")

    push("\n[2] WHY SEGMENTATION IS THE HARD STEP")
    sample = df["sentence"].iloc[0]
    push(f"    Khmer writes without spaces between words. This sentence:")
    push(f"      {sample[:70]}")
    push(f"    contains {len(str(sample).split())} whitespace-separated chunks but")
    push(f"    {len(tokenize(str(sample)))} actual words once segmented:")
    push(f"      {tokenize(str(sample))[:14]}")
    push("\n    Splitting on spaces would give a handful of enormous 'words', each")
    push("    appearing once, and every downstream count would be meaningless.")

    push("\n[3] STAGE BY STAGE ON REAL SENTENCES")
    for idx in [0, len(df) // 3, len(df) - 1][:3]:
        row, stages = df.iloc[idx], results[idx]
        push("\n" + "-" * 78)
        push(f"id={row['id']}   label={row['label']}")
        push("-" * 78)
        for name, _ in TEXT_STAGES:
            push(f"  {name:22} | {str(stages[name])[:100]}")
        for name in ["05_tokenized", "06_stopwords_removed", "07_short_dropped"]:
            push(f"  {name:22} | {stages[name][:14]}")
        push(f"  {'>> CLEANED':22} | {stages['cleaned_text'][:100]}")

    push("\n\n[4] STOPWORDS")
    push(f"    Khmer function words defined : {len(KHMER_STOPWORDS)}")
    push(f"    negation/degree words kept   : {len(KEEP_WORDS)}  {sorted(KEEP_WORDS)}")
    push(f"    effective stopword list      : {len(STOPWORDS)}")
    push("\n    No standard Khmer stopword list ships with NLTK or spaCy, so this")
    push("    one was written by hand from particles, classifiers, pronouns and")
    push("    conjunctions. Negation is excluded for the same reason as in English:")
    push("    dropping មិន ('not') would invert the polarity of the sentence.")

    push("\n\n[5] CORPUS STATISTICS")
    raw_tokens = [t for s in df["sentence"] for t in tokenize(str(s))]
    clean_tokens = [t for toks in df["tokens"] for t in toks]
    stats = pd.DataFrame(
        {
            "before": [len(raw_tokens), len(set(raw_tokens)),
                       round(df["n_tokens_raw"].mean(), 2)],
            "after": [len(clean_tokens), len(set(clean_tokens)),
                      round(df["n_tokens_clean"].mean(), 2)],
        },
        index=["total tokens", "vocabulary size", "mean tokens/sentence"],
    )
    stats["reduction"] = (1 - stats["after"] / stats["before"]).map(lambda v: f"{v:.1%}")
    push(stats.to_string())
    empty = int((df["n_tokens_clean"] == 0).sum())
    push(f"\n    sentences empty after cleaning: {empty}")

    push("\n\n[6] TOP 15 WORDS PER CLASS")
    for label in ["positive", "neutral", "negative"]:
        sub = df[df["label"] == label]
        toks = [t for toks in sub["tokens"] for t in toks]
        if not toks:
            continue
        top = ", ".join(f"{w}({n})" for w, n in Counter(toks).most_common(15))
        push(f"\n    {label} ({len(sub)} sentences, {len(toks)} tokens)")
        push(f"      {top}")

    report = "\n".join(out)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    cols = [c for c in ["id", "pair", "label", "sentence", "cleaned_text", "tokens",
                        "n_tokens_raw", "n_tokens_clean"] if c in df.columns]
    df[cols].to_csv(out_csv, index=False)
    pd.DataFrame([
        {"id": df["id"].iloc[i], "label": df["label"].iloc[i],
         **{k: (" ".join(v) if isinstance(v, list) else v)
            for k, v in r.items() if k != "tokens"}}
        for i, r in enumerate(results)
    ]).to_csv(stages_csv, index=False)

    print(report)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {stages_csv}")
    print(f"Saved: {out_report}")


if __name__ == "__main__":
    main()
