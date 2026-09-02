# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university group project: three-class sentiment classification (positive /
neutral / negative) on Khmer sentences from the kh-polarity corpus. It is a
sequence of standalone data-processing scripts, not an application or library —
there is no package, no test suite, no linter, and no build step. `README.md`
holds the current numbers and the presenter split; `docs/` holds the slide
scripts each person presents from.

Work is divided by "Person N" (see README): 3 owns annotation + preprocessing,
4 representation, 5 models + training, 6 results + conclusion. Persons 5 and 6
reach the data only through `scripts/load_features.py` and
`reports/model_results.csv`, never by rebuilding features themselves.

## Environment

`gensim` has no wheel for Python 3.14, so the project pins 3.11:

```bash
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

Every command is run as `./.venv/bin/python scripts/<name>.py`. `.venv/` is
gitignored and may not exist yet on a fresh clone.

## Pipeline

Each script reads files written by the previous one and refuses to run if they
are missing. Run in this order after changing anything upstream:

```
data/raw/kh-polar.txt            (gitignored, download from the kh-polarity repo)
  └ build_annotation_sheets.py → data/annotation/pair{1,2,3}_annotation.xlsx  (blank sheets)
       ↓ annotators fill them in by hand
     collect_annotations.py     → data/processed/labelled_dataset.csv           (agreement-only)
                                  data/processed/labelled_dataset_resolved.csv  (disagreements settled)
                                  data/processed/adjudicated_annotations.csv
     preprocess.py              → data/processed/preprocessed.csv
     split_data.py              → data/splits/{train,val,test}.csv
     representations.py         → data/features/{bow,ngram,tfidf,word2vec,fasttext}.npz + labels.npz
     inspect_representations.py   (analysis only, writes no data)
     train_models.py            → reports/model_results.csv + model_training_report.txt
     train_rnn.py               → appends 2 rows to model_results.csv, rnn_training_report.txt
     analyse_results.py         → reports/results_analysis.txt
     make_figures.py            → reports/figures/*.png
```

`segment_compare.py` is a standalone comparison (dictionary maximum matching vs
the CRF segmenter) and is not part of the data flow.

Only `build_annotation_sheets.py` needs the raw corpus; everything downstream
starts from the committed CSVs.

## The two dataset variants and the `--tag` convention

The same pipeline is run twice over two versions of the labelled data:

| Variant | Source CSV | Command |
|---|---|---|
| resolved (default) | `labelled_dataset_resolved.csv` — all rows, disagreements settled by the corpus label | `preprocess.py` then `split_data.py` then `representations.py`, no flags |
| agreed | `labelled_dataset.csv` — only rows where both annotators agreed | `preprocess.py --csv data/processed/labelled_dataset.csv --tag agreed`, then `split_data.py --csv data/processed/preprocessed_agreed.csv --tag agreed`, then `representations.py --tag agreed` |

`--tag X` appends `_X` to every output filename that stage writes — processed
CSV, split CSVs, feature `.npz` files and the report. The tag must be threaded
through all three stages or a later stage will silently read the other
variant's files. Nothing else varies between the two runs.

## Invariants that will break silently if violated

- **`token_pattern=r"\S+"` on every scikit-learn vectorizer.** Text is already
  word-segmented with tokens joined by spaces. sklearn's default pattern
  assumes English word characters and drops Khmer script entirely, producing an
  empty vocabulary rather than an error.
- **Fit on train only.** Vectorizers and embedding models are fit on `train`
  and applied to `val`/`test`. `inspect_representations.py` likewise computes
  everything on train only — inspecting val/test to decide what to report is
  leakage through the back door.
- **`stable_hash` for gensim.** `seed=42` alone does not reproduce because
  gensim seeds vectors with Python's per-process-randomised `hash()`;
  `representations.py` passes `hashfxn=stable_hash` (SHA-256) instead. Both
  `representations.py` and `inspect_representations.py` must use the same
  `SEED`/`EMB_DIM`/`EMB_EPOCHS` values, since the latter retrains the
  embeddings rather than loading them.
- **`STOPWORDS = KHMER_STOPWORDS - KEEP_WORDS`** in `preprocess.py`. Negation
  and degree words (មិន, ពុំ, អត់, គ្មាន, ណាស់ …) are deliberately kept — មិន is
  the strongest frequent negative feature in the corpus. Removing them turns
  "not good" into "good".
- **No stemming or lemmatization, on purpose.** Khmer is analytic: no
  conjugation, no inflection, no affixes to strip, and no Khmer lemmatizer
  exists. Do not add one to "complete" the pipeline.
- **`khmer-nltk`'s CRF `word_tokenize` is the segmenter.** Dictionary maximum
  matching exists only in `segment_compare.py` for the comparison; it leaves
  15% of positions unmatched.
- **Noto Sans Khmer is required** by `make_figures.py` and
  `build_annotation_sheets.py`; both fail loudly rather than render empty
  boxes.

## Feature file format

`data/features/*.npz` is not a plain array dump — sparse matrices are stored as
their CSR component arrays plus a `kind` marker, and dense embeddings as one
array per split. Always go through `scripts/load_features.py`:

```python
from load_features import load
X_train, X_val, X_test, y_train, y_val, y_test = load("tfidf")
```

`load()` returns scipy CSR for `bow`/`ngram`/`tfidf` and dense float32 arrays
for `word2vec`/`fasttext`; both feed scikit-learn estimators directly, so one
training loop covers all five. Labels are the strings in
`load_features.CLASSES`; `to_numeric()` converts them. `feature_names(rep)`
returns the vocabulary for the count-based three and `None` for the embeddings.

Only the Word2Vec `.model` is saved (FastText's carries the whole character
n-gram matrix); `.model`/`.npy` files are gitignored and regenerated by
`representations.py`.

## Script conventions

New scripts should follow the shape the existing ones share:

- Module docstring explaining the *why*, including what is Khmer-specific about
  the step, and a `Usage:` block at the end.
- `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`, all
  paths derived from it; scripts are run from the repo root but never rely on
  the cwd.
- Constants (`SEED = 42`, `TOKEN_PATTERN`, column names) at module top.
- Build the report as a list of lines via `out, push = [], out.append`, then
  write it to `reports/<name><suffix>.txt` **and** print it. The report is the
  deliverable — the numbers quoted in `README.md` and `docs/` come from these
  files, so re-run the affected stage and update both when behaviour changes.
- `raise SystemExit("Missing <path>\nRun scripts/<upstream>.py first.")` when an
  input is absent.

## Evaluating models (Person 5/6 work)

Report **macro-F1**, not accuracy: three classes, 408 train / 72 val / 120 test
sentences, so one test sentence moves accuracy by 0.83 points. Hyperparameters
are tuned by cross-validation inside train; test is evaluated once.

Two results established by `analyse_results.py` govern how any new number here
should be reported:

- **Differences below ~0.09 macro-F1 are not real.** The best model's bootstrap
  interval is 18 points wide and McNemar cannot separate it from the runner-up
  (p = 0.815). Never present the leaderboard order as a ranking.
- **Validation rank does not predict test rank** (ρ = −0.016 over 42 models);
  cross-validation does, weakly (ρ = +0.389). Selecting on the 72-sentence
  validation split is what cost the project ~6 points.

Models are rebuilt from the `best_params` recorded in `model_results.csv` rather
than re-searched — `analyse_results.refit()` is the single place that does this,
and `make_figures.py` imports it rather than repeating the logic.
