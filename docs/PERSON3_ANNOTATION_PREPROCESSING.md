# Person 3 — Annotation & Preprocessing

Two slides plus one backup, script, and answers. ~3 minutes.

---

## SLIDE 1 — Annotation & agreement

**Bullets**
- 600 sentences, 3 pairs, 200 each
- Two annotators per pair, working independently
- Labels: positive / neutral / negative
- Cohen's Kappa within each pair

**Visual — this table:**

| Pair | Annotators | Raw agreement | Cohen's κ | |
|---|---|---|---|---|
| pair1 | Bath, Nacc | 44.0% | **0.157** | slight |
| pair2 | Nita, Reaksa | 62.0% | **0.380** | fair |
| pair3 | Seth, Krisna | 94.5% | **0.917** | almost perfect |
| | | **mean** | **0.485** | moderate |

**Say (~90 sec):**

> "We annotated 600 Khmer sentences. Six people in three pairs, 200 each. Both
> members of a pair labelled the same sentences independently, so we could
> measure how much they agreed.
>
> The results vary enormously. Pair 3 reached Cohen's Kappa of **0.917** —
> almost perfect. Pair 1 reached **0.157**, which is barely above chance. Same
> guideline, same task, same three labels.
>
> [PAUSE]
>
> That spread is the finding. It isn't that Khmer sentiment is impossible to
> annotate — pair 3 proves it can be done consistently. It's that our guideline
> left room for interpretation, and the pairs who discussed it beforehand
> converged while the others didn't.
>
> The confusion matrices show where it broke down: disagreement is almost
> entirely **neutral versus a weak polarity**, not positive versus negative.
> Annotators rarely swapped those two — they disagreed about whether a sentence
> was evaluative at all.
>
> 401 of 600 sentences had both annotators agree. Rather than break the ties with
> a coin flip, we kept two datasets: the 401 agreed sentences, and a
> 600-sentence version where disagreements fall back to the corpus's own label.
> We train on the larger one and report the smaller as a check."

---

## SLIDE 2 — Preprocessing Khmer

**Bullets**
- Khmer has **no spaces between words**
- Word segmentation: khmer-nltk CRF model
- Stopwords: hand-built list, negation kept
- **No stemming or lemmatization** — and that is correct

**Visual — this before/after:**

```
raw       បឹងកេតធ្លាប់បានក្លាយជាជើងឯកនៃពានរង្វាន់សម្តេចតេជោ
          3 whitespace chunks

segmented បឹង · កេត · ធ្លាប់ · បាន · ក្លាយជា · ជើងឯក · នៃ · ពានរង្វាន់ …
          27 actual words
```

**Say (~90 sec):**

> "Khmer preprocessing is not English preprocessing with different characters.
> Three of the standard steps do not transfer.
>
> **Tokenization.** Khmer is written without spaces between words. Splitting on
> whitespace gives three enormous chunks where there are actually 27 words. Every
> downstream count would be meaningless. We use khmer-nltk, a CRF segmenter
> trained on Khmer.
>
> **Stemming and lemmatization — we do neither, and that is the right call.**
> Khmer is an analytic language. Verbs do not conjugate, nouns do not inflect for
> number or case. 'Go', 'goes', 'went' and 'going' are all the same surface form,
> ទៅ. There are no affixes to strip, which is why no Khmer lemmatizer exists —
> not because the tooling is missing, but because there is nothing for it to do.
>
> **Stopwords.** No standard Khmer stopword list ships with NLTK or spaCy, so we
> wrote one: 50 function words. We deliberately kept the negation words.
> Dropping មិន — 'not' — would invert the polarity of exactly the sentences
> labelled negative.
>
> That decision is measurable. មិន is the strongest frequent negative feature in
> the corpus. Treating it as a stopword would have deleted our best feature."

---

## BACKUP SLIDE — Dictionary vs CRF segmentation

Show only if asked how segmentation works.

| | Tokens | Vocabulary | Unmatched |
|---|---|---|---|
| Dictionary (34,398 words) | 14,488 | 3,143 | **15.2%** |
| CRF (khmer-nltk) | 11,615 | 3,034 | — |

Only **3 of 400** sentences segmented identically by both.

**The example that settles it** — the number "2,073":

```
dictionary   ២ · , · ០ · ៧ · ៣        six fragments
CRF          ២,០៧៣                    one token
```

> "There are two standard approaches. **Dictionary-based** maximum matching takes
> the longest string present in a Khmer wordlist — deterministic and inspectable.
> **CRF** is a statistical model that predicts each boundary from surrounding
> characters.
>
> We implemented both. Using a 34,000-word SIL dictionary, 15% of positions had
> no match at all. This example is the clearest: the number 2,073. The dictionary
> shatters it into six fragments because the digits aren't in the wordlist. The
> CRF keeps it whole.
>
> Our corpus is news text — names, places, numbers — exactly what a fixed
> wordlist misses. So we use the CRF, and keep the dictionary as the comparison
> that shows what the CRF buys us."

---

## The numbers

| | |
|---|---|
| Sentences annotated | 600 (3 pairs × 200) |
| Cohen's κ | 0.157 / 0.380 / 0.917, mean **0.485** |
| Both annotators agreed | 401 / 600 (66.8%) |
| Dataset used for training | 600 (disagreements resolved via corpus label) |
| Khmer stopwords defined | 50, minus 9 negation/degree words kept |
| Vocabulary after cleaning | 2,883 words |

---

## Questions

**"Why is pair 1's Kappa so much lower than pair 3's?"**
> "Same guideline, very different outcome — 0.157 against 0.917. The confusion
> matrix shows pair 1 disagreed mostly on neutral versus a weak polarity, which
> is exactly where our guideline was underspecified. Pair 3 evidently settled
> that boundary between themselves before starting. The fix is to tighten the
> guideline's threshold for 'evaluative enough to count' and re-annotate."

**"Why didn't you use stemming or lemmatization?"**
> "Because Khmer has no inflection to remove. It's analytic — verbs don't
> conjugate, nouns don't take plural or case endings. An English stemmer would
> cut characters off Khmer words without removing any affix, because there are
> none."

**"How does the CRF segmenter work?"**
> "It predicts, for each position between two characters, whether a word boundary
> falls there, using the surrounding characters as features. It was trained on
> hand-segmented Khmer. We use it as a library rather than reimplementing it."

**"Why keep the disagreed sentences at all?"**
> "Because keeping only agreements biases the dataset toward easy cases. The 401
> agreed sentences are the obvious ones; the hard ones are what a classifier
> needs to see. We report both datasets so the effect is visible rather than
> hidden."

**"Is 0.485 good?"**
> "It's 'moderate' on the Landis and Koch scale. For three-class sentiment that's
> acceptable but not strong. The honest reading is that it's an average of one
> excellent pair and one poor one, which says more about our process than about
> the task."
