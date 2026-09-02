# Khmer Sentiment Analysis

Text classification project on Khmer-language sentences.
Corpus: [kh-polarity](https://github.com/ye-kyaw-thu/kh-polarity) (Ye Kyaw Thu et al.).

## Pipeline

```
annotate (pairs) → agreement → preprocess → split → REPRESENT → train → evaluate
                   └── Person 3 ──┘                  └ Person 4 ┘   P5      P6
```

## Presenter split

| Person | Section | Document |
|---|---|---|
| 1 | Introduction & research | — |
| 2 | Dataset & ethics | — |
| **3** | **Annotation & preprocessing** | `docs/PERSON3_ANNOTATION_PREPROCESSING.md` ✅ |
| **4** | **Text representation** | `docs/PERSON4_TEXT_REPRESENTATION.md` ✅ |
| **5** | **ML models & training** | `docs/PERSON5_MODELS_TRAINING.md` ✅ |
| 6 | Results & conclusion | `reports/model_results.csv` — 42 models ready |

## For Person 5 — loading the features

```python
from load_features import load
X_train, X_val, X_test, y_train, y_val, y_test = load("tfidf")
```

Names: `bow`, `ngram`, `tfidf`, `word2vec`, `fasttext`

| Representation | Features | Type |
|---|---|---|
| Bag-of-Words | 1,187 | sparse CSR |
| N-grams (1–2) | 1,565 | sparse CSR |
| TF-IDF (1–2) | 1,565 | sparse CSR |
| Word2Vec | 100 dims | dense array |
| fastText | 100 dims | dense array |

All five were **fit on train only**, then applied to val and test. Labels are
identical across all five — same rows, same order.

| Split | Sentences | positive / neutral / negative |
|---|---|---|
| train | 408 | 124 / 155 / 129 |
| val | 72 | 22 / 27 / 23 |
| test | 120 | 36 / 46 / 38 |

`to_numeric(y)` converts labels to integers. `feature_names(rep)` gives column
names for the count-based three.

**Report macro-F1**, not accuracy — three classes, and the set is small enough
that one test sentence moves accuracy by 0.83 points.

## Key results so far

**Annotation.** Cohen's Kappa within each pair: pair1 (Bath, Nacc) **0.157**,
pair2 (Nita, Reaksa) **0.380**, pair3 (Seth, Krisna) **0.917**, mean **0.485**.
401 of 600 sentences had both annotators agree. Disagreement is concentrated on
neutral versus a weak polarity, not positive versus negative. The spread between
pairs is wider than the spread within any pair, which points at the guideline
rather than the annotators.

**Segmentation.** Dictionary maximum matching leaves 15.2% of positions
unmatched and agrees with the CRF on only 3 of 400 sentences. We use the CRF.

**Representation.** 22.8% of test-set word occurrences never appear in training —
the argument for fastText. Word2Vec drops all of them.

**Models.** 42 trained: 8 algorithms × 5 representations, plus SimpleRNN and
LSTM. Best on test is **Naive Bayes + TF-IDF at macro-F1 0.544**; the **LSTM is
third at 0.529**. Validation and test rank the models differently — selecting on
validation gives Linear SVM + fastText, which scores 0.416 on test. With 72
validation sentences the top ten models are statistically indistinguishable.

## Scripts

| Script | Does |
|---|---|
| `collect_annotations.py` | merge pair sheets, Cohen's Kappa, build the dataset |
| `segment_compare.py` | dictionary vs CRF word segmentation |
| `preprocess.py` | Khmer cleaning pipeline (segment, stopwords, no stemming) |
| `split_data.py` | stratified train/val/test split |
| `representations.py` | **builds all five feature sets** |
| `load_features.py` | **one-line loader for Person 5** |
| `inspect_representations.py` | what the features actually contain |
| `train_models.py` | **8 algorithms × 5 representations, CV-tuned** |
| `train_rnn.py` | **SimpleRNN and LSTM (PyTorch)** |
| `make_figures.py` | slide figures (needs Noto Sans Khmer) |
| `build_annotation_sheets.py` | generate the blank annotation workbooks |

## Setup

`gensim` does not build on Python 3.14, so this uses 3.11:

```bash
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

## Running it

```bash
./.venv/bin/python scripts/collect_annotations.py
./.venv/bin/python scripts/preprocess.py
./.venv/bin/python scripts/split_data.py
./.venv/bin/python scripts/representations.py
./.venv/bin/python scripts/inspect_representations.py
./.venv/bin/python scripts/train_models.py
./.venv/bin/python scripts/train_rnn.py
./.venv/bin/python scripts/make_figures.py
```

## Khmer-specific notes

**Word segmentation is the first real problem.** Khmer is written without spaces
between words, so whitespace splitting returns whole clauses. We use
`khmer-nltk`, a CRF segmenter, and keep a dictionary implementation in
`scripts/segment_compare.py` for comparison.

**No stemming or lemmatization, deliberately.** Khmer is an analytic language:
verbs do not conjugate, nouns do not inflect for number or case. There are no
affixes to strip, which is why no Khmer lemmatizer exists.

**Stopwords are hand-built.** No Khmer stopword list ships with NLTK or spaCy,
so `preprocess.py` defines 50 function words. Negation words (មិន, ពុំ, អត់,
គ្មាន) are deliberately kept — មិן is the strongest frequent negative feature in
the corpus, appearing 90 times (64 of them in negative sentences).

**Figures need a Khmer font.** `make_figures.py` requires Noto Sans Khmer and
fails loudly if it is missing, rather than rendering empty boxes.

## Data

- `data/annotation/` — the annotators' workbooks and the answer key
- `data/dictionary/` — SIL NRSI khmerlbdict wordlists (MIT licence, attribution
  and licence text included)
- `data/processed/` — adjudicated annotations, cleaned text, per-stage output
- `data/splits/` — train / val / test
- `data/features/` — the five representations

`data/raw/kh-polar.txt` is the source corpus (5.4 MB) and is gitignored; it is
publicly downloadable from the link above.
