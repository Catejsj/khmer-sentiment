# Person 5 — Models & Training

Two slides, script, and answers. ~3 minutes.

---

## SLIDE 1 — What we trained

**Bullets**
- 8 classical algorithms × 5 representations = **40 combinations**
- Plus **SimpleRNN and LSTM** — 42 in total
- Tuned by 5-fold cross-validation on **train only**
- Test touched once, at the end

**Visual — the algorithm list:**

| Classical (8) | Recurrent (2) |
|---|---|
| Naive Bayes · Logistic Regression | SimpleRNN (`nn.RNN`) |
| Linear SVM · SVM (RBF) | LSTM (`nn.LSTM`) |
| Random Forest · Gradient Boosting | |
| K-Nearest Neighbours · Decision Tree | |

**Say (~90 sec):**

> "We trained eight classical algorithms on all five representations — that's 40
> combinations — plus two recurrent networks, so 42 models in total.
>
> **On tuning.** Our training set is 408 sentences and validation is 72. Tuning
> directly against 72 sentences would be fitting to noise — one sentence moves
> the validation score by 1.4 points. So hyperparameters were selected by 5-fold
> stratified cross-validation on the training split only, which averages over
> five folds instead of trusting one small sample.
>
> **On leakage.** The vectorisers were already fit on train only by Person 4.
> We never call fit on validation or test. The test set is evaluated once, at the
> very end, for the selected model — nothing is chosen using it.
>
> **On the RNN.** The eight classical models all read a fixed-length vector per
> sentence, so they cannot see word order. A recurrent network reads word by word
> and carries a hidden state forward. In Khmer that matters for negation — មិន
> placed before a positive word inverts the sentence, and only the RNN can see
> that ordering. We seeded its embedding layer from the Word2Vec vectors, because
> 408 sentences cannot train an embedding table from scratch."

---

## SLIDE 2 — Results, and a warning

**Bullets**
- (none — the figure is the slide)

**Visual:** `reports/figures/fig3_model_results.png`, full bleed

**Say (~90 sec):**

> "Light blue is validation, dark blue is test, and the dashed line is the random
> baseline at 0.333 for three classes.
>
> The best model on test is **Naive Bayes on TF-IDF at 0.544 macro-F1**. The
> **LSTM comes third at 0.529** — the best of the two recurrent models, and
> comfortably ahead of the SimpleRNN at 0.405.
>
> [PAUSE]
>
> But look at the two bars for each model. **Validation and test disagree,
> badly.**
>
> If we select on validation, as you are supposed to, we pick Linear SVM on
> fastText — validation 0.511. On test that model scores **0.416**, which is
> eighth. Meanwhile Naive Bayes on TF-IDF scored only 0.388 on validation and
> came top on test.
>
> That is not a bug, it's sample size. Seventy-two validation sentences cannot
> rank 42 models reliably. The honest conclusion is that **most of these models
> are statistically indistinguishable on this data** — everything from 0.49 to
> 0.54 is inside the noise — and that the correct fix is more annotated data, not
> more model tuning."

---

## The numbers

| | |
|---|---|
| Models trained | 42 (8 algorithms × 5 representations + 2 RNN) |
| Tuning | 5-fold stratified CV on train |
| Train / val / test | 408 / 72 / 120 |
| Random baseline (3 classes) | 0.333 |
| **Best on test** | **Naive Bayes + TF-IDF, macro-F1 0.544** |
| Best recurrent | LSTM, macro-F1 0.529 |
| Selected on validation | Linear SVM + fastText → test 0.416 |
| One test sentence is worth | 0.83 accuracy points |

**Top 5 by test macro-F1:**

| Model | Representation | Val | Test |
|---|---|---|---|
| Naive Bayes | TF-IDF | 0.388 | **0.544** |
| Logistic Regression | TF-IDF | 0.418 | 0.531 |
| **LSTM** | learned embedding | 0.437 | **0.529** |
| Linear SVM | TF-IDF | 0.406 | 0.519 |
| Gradient Boosting | Bag-of-Words | 0.392 | 0.516 |

---

## The RNN in detail

| | SimpleRNN | LSTM |
|---|---|---|
| Test accuracy | 0.417 | **0.542** |
| Test macro-F1 | 0.405 | **0.529** |
| Best epoch | 7 | 6 |
| Stopped at | 17 | 16 |

Both peak within about six epochs and then overfit — training loss falls to 0.09
while validation F1 stops improving. Early stopping on validation macro-F1 keeps
the best checkpoint.

> "The LSTM beats the SimpleRNN by 12 F1 points, which is the expected result:
> gating lets it carry information across a longer sentence, where the plain
> recurrent unit loses it. But both stop learning after six epochs. 408 sentences
> is simply not enough to train a recurrent network, and we report that rather
> than tuning until it looks better."

---

## Questions

**"Why did you use cross-validation instead of the validation set for tuning?"**
> "72 validation sentences. One sentence is 1.4 points of macro-F1, so tuning
> against it would be fitting to noise. Five-fold CV on the 408 training
> sentences averages over five splits, which is far more stable. We still use
> validation to compare already-tuned models, just not to pick their
> hyperparameters."

**"Why does your best test model have a poor validation score?"**
> "Because 72 sentences cannot rank 42 models. That mismatch is the most
> important thing on the slide — it says the differences between our top ten
> models are not real. We report the validation-selected model and the
> best-on-test model separately, rather than pretending the ranking is stable."

**"Isn't picking the best test score cheating?"**
> "It would be if we used it to select. We didn't — selection was on validation,
> and that gives Linear SVM on fastText at 0.416. We report the full table so the
> gap between selection and outcome is visible."

**"How did you avoid data leakage?"**
> "Three ways. The vectorisers were fit on train only and merely applied to val
> and test. Hyperparameters were chosen by cross-validation inside the training
> split. And test was evaluated once, at the end. No preprocessing step ever saw
> the test labels."

**"Why is everything so close to the random baseline?"**
> "Three classes, 408 training sentences, and a task where our own annotators
> only reached Cohen's Kappa 0.485. If humans agree that little, a classifier
> cannot do much better — the labels themselves carry noise. Around 0.54 against
> a 0.333 baseline is a real signal, but a weak one."

**"What would improve it?"**
> "More annotated data, and better annotation. Pair 3 reached 0.917 agreement, so
> consistent labelling is achievable — a re-annotation with the tightened
> guideline would raise the ceiling before any model change would."
