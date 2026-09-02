# Person 6 — Results & Conclusion

Two slides, script, and answers. ~3 minutes.

Everything here comes from `reports/results_analysis.txt`, produced by
`scripts/analyse_results.py`.

---

## SLIDE 1 — Is any of this difference real?

**Title:** Results — one number, and how much to trust it

**Bullets**
- Best model: **Naive Bayes + TF-IDF, macro-F1 0.544**
- 95% confidence interval **[0.452, 0.630]** — 18 points wide
- The top of the leaderboard is **one result, not a ranking**

**Visual:** `reports/figures/fig4_confusion.png`

**Say (~90 sec):**

> "Person 5 gave you 42 models and a leaderboard. My job is to tell you which
> parts of it are real.
>
> Our best model is Naive Bayes on TF-IDF at **0.544 macro-F1**, against a random
> baseline of 0.333. That is genuine signal — the model has learned something.
>
> But a single number on 120 test sentences is not a result. So we bootstrapped
> it: resample the test set 2,000 times and recompute. The 95% confidence
> interval is **0.452 to 0.630** — eighteen points wide. **Every model in the top
> ten falls inside that interval.**
>
> To check the ranking properly you need a paired test, one that compares two
> models sentence by sentence instead of comparing two summary scores. We ran
> McNemar's test between the best model and the runner-up: **p = 0.815**. There
> is no measurable difference between them. Naive Bayes, Logistic Regression and
> the LSTM are **one result**, not first, second and third.
>
> [PAUSE — point at the figure]
>
> Now where it fails. Of 54 errors, **22 are a real polarity called neutral** and
> only 13 are a full flip from positive to negative. The model is not confusing
> good with bad — it is refusing to commit. That is the exact axis our annotators
> disagreed on, which is the first hint that the problem is not the algorithm."

---

## SLIDE 2 — The ceiling

**Title:** How well do humans do the same task?

**Bullets**
- (none — the figure is the slide)

**Visual:** `reports/figures/fig5_ceiling.png`, full bleed

**Say (~90 sec):**

> "This is the slide I want you to remember.
>
> We scored our own annotators exactly the way we scored a model: take one
> annotator's labels as the truth, their partner's as the prediction, macro-F1.
> Same units, same three classes, directly comparable.
>
> **Pooled across all three pairs, humans score 0.666.** Our model scores
> **0.544**. The gap is **12 points** — not the 46 points you would assume if you
> measured against a perfect 1.0.
>
> And look at the spread. Pair 3 scored **0.946** against each other. Pair 1
> scored **0.438** — barely above the random baseline, and *below our model*. The
> difference between our annotator pairs is bigger than the difference between
> our model and our annotators.
>
> [PAUSE]
>
> That is the conclusion of this project. **The bottleneck is the labels, not the
> algorithm.** We trained 42 models, tuned them properly, and included a neural
> network. None of it matters as much as the fact that two of our three pairs
> could not agree on what a positive sentence is. Pair 3 proves the task is doable
> — the guideline works when it is applied consistently.
>
> So if we had one more week, we would not train a 43rd model. We would
> re-annotate pairs 1 and 2."

---

## The numbers

| | |
|---|---|
| Best on test | Naive Bayes + TF-IDF, **0.544** |
| 95% CI (2,000 bootstrap resamples) | **[0.452, 0.630]** |
| Best vs runner-up | McNemar **p = 0.815** — indistinguishable |
| Selected on validation | Linear SVM + fastText → **0.416** |
| Best vs selected | McNemar **p = 0.029** — but post-hoc, see below |
| Human vs human (pooled) | **0.666** |
| Human, best pair (Seth / Krisna) | **0.946** |
| Human, worst pair (Bath / Nacc) | **0.438** |
| Random baseline | 0.333 |

**Which selection rule should we have trusted:**

| Rule | Picks | Test |
|---|---|---|
| Highest validation macro-F1 | Linear SVM + fastText | 0.416 |
| Highest cross-validation F1 | Random Forest + n-grams | **0.472** |
| *(oracle — highest test)* | Naive Bayes + TF-IDF | 0.544 |

Rank correlation with test: validation **ρ = −0.016** (p = 0.92), cross-validation
**ρ = +0.389** (p = 0.016). Validation rank carries *no* information about test
rank. Cross-validation carries some. We selected on validation and it cost us
about six points — the honest lesson from this project's methodology.

---

## Where the errors come from

Two things we expected to explain the failures, and neither does:

| | correct | wrong |
|---|---|---|
| mean length (tokens) | 19.3 | 20.5 |
| mean out-of-vocabulary | 22.9% | 23.3% |

A 1.2-token and 0.4-point difference. The model does not fail on long sentences
or on sentences full of unseen words.

What *did* show up, in the opposite direction to the prediction:

| | accuracy |
|---|---|
| 32 sentences containing a negation word | **62.5%** |
| 88 sentences without one | 52.3% |

Negated sentences are **10 points easier**, not harder. The bigram features are
doing their job — `មិន ទាន់` is among the model's strongest negative features —
so the negation is read as a unit rather than cancelling out. This is direct
evidence for Person 3's decision to keep negation out of the stopword list, and
for Person 4's decision to include bigrams.

**Top features the model learned** (Naive Bayes on TF-IDF, per class):

| Class | Words |
|---|---|
| negative | ប៉ះពាល់ · ប៉ះ · មិន · មានអារម្មណ៍ · ប្រឈម · មិន ទាន់ · ពិបាក |
| neutral | ផ្លាស់ប្តូរ · ចាប់ផ្តើម · ប៉ុណ្ណោះ · សវនាការ · លើក · កីឡា · ជាទូទៅ |
| positive | ដ៏ ល្អ · ប្រកបដោយ · ការអប់រំ · ស្នេហា · ប្រសើរឡើង · ជួយ |

These are the same words Person 4's log-odds figure surfaced independently. The
model keys on sentiment vocabulary, not on topic artefacts — which is the check
that matters before reporting any score.

---

## Limitations

- **120 test sentences.** Every number carries roughly a ±0.09 interval, wider
  than most of the gaps being compared.
- **Trained on the resolved variant**, where 199 of 600 labels came from the
  corpus rather than from our annotators. The agreement-only variant has not been
  run through the models, so the robustness check the annotation report promises
  is still open.
- **One split, one seed.** Repeated splits would estimate this better than a
  single 120-sentence test set.
- **Segmentation is unvalidated** — no gold segmented Khmer to check the CRF
  against, so tokenisation errors propagate silently into every representation.

---

## Questions

**"Why report 0.544 when you selected Linear SVM on validation?"**
> "Both numbers are in the report and they answer different questions. 0.416 is
> what our protocol produced — select on validation, evaluate once on test. 0.544
> is the best any of our models achieved. We report the protocol number as our
> result and the best number as the ceiling of our search, and we show the gap
> rather than hiding it. Quoting 0.544 alone would be selecting on test."

**"McNemar says p = 0.029 for the best versus the selected model. Isn't that significant?"**
> "Nominally, yes — but that model was chosen *because* it topped the test set.
> Across 42 models, the best-looking comparison is expected to look good by
> chance. Corrected for even the ten comparisons on the leaderboard, 0.029 × 10
> is no longer significant. We report it as suggestive, not established."

**"Why is human agreement your ceiling and not 100%?"**
> "Because the model is trained and tested on labels those humans produced. If
> two annotators only agree at 0.666 macro-F1, the labels themselves contain
> disagreement, and a classifier cannot be more consistent than the thing it is
> imitating. Measuring against 1.0 would be measuring against a standard the data
> does not contain."

**"Pair 1 scored 0.438 — below your model. What does that mean?"**
> "That the model is more self-consistent than that pair was with each other. It
> does not mean the model is better at Khmer sentiment; it means pair 1's labels
> are close to noise, and roughly a third of our training data comes from that
> pair. Their rows are actively hurting the model."

**"Why does the model favour neutral?"**
> "Neutral is the largest class — 228 of 600 sentences — and it is also the class
> our annotators disagreed about most. So the model sees the most examples of
> the least consistently defined category. Predicting neutral is the safe move,
> and 22 of our 54 errors are exactly that."

**"You included an LSTM. Was it worth it?"**
> "As a result, no — 0.529, inside the flat group, for far more compute. As
> evidence, yes. It is the only model that reads word order, so its failure to
> beat Naive Bayes tells us the ceiling here is not about model capacity. That is
> a finding, not a null result."

**"What would you do next?"**
> "In order. One: re-annotate pairs 1 and 2 with the tightened guideline — pair 3
> hit 0.917 Kappa, so we know the ceiling can move. Two: more data; 408 training
> sentences is the binding constraint on every model, and the only thing that
> would let the RNN work. Three: a pretrained Khmer language model instead of
> embeddings trained on 408 sentences. That is the change most likely to move the
> score, and our corpus size is what makes it necessary rather than optional."
