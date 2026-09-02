# Person 3 — Annotation & Preprocessing

Two slides, script, and answers. ~3 minutes.

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
| pair3 | Seth | — | — | partner not submitted |
| | | **mean** | **0.269** | fair |

**Say (~90 sec):**

> "We annotated 600 Khmer sentences from the kh-polarity corpus. Six people in
> three pairs, 200 sentences each. Both members of a pair labelled the same 200
> independently, so we could measure how much they agreed.
>
> The guideline defines three labels — positive, neutral, negative — with a
> five-step decision procedure and seven pre-decided edge cases.
>
> Agreement came out low. Pair 1 reached Cohen's Kappa of 0.157, which is
> 'slight'. Pair 2 reached 0.380, 'fair'. Mean 0.269.
>
> [PAUSE]
>
> That is a real result, not a failure to report. Three-class sentiment is hard,
> and it is harder in Khmer where there is no widely-agreed annotation standard
> to anchor to. The confusion matrix shows where it broke down: the disagreement
> is overwhelmingly **neutral versus a weak polarity**, not positive versus
> negative. Annotators rarely swapped those two — they disagreed about whether
> something counted as evaluative at all.
>
> Only 212 of the 400 sentences had both annotators agree. Rather than break the
> ties with a coin flip, we kept two datasets: the 212 agreed sentences, and a
> 400-sentence version where disagreements fall back to the corpus's own label.
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
> whitespace gives you three enormous chunks where there are actually 27 words.
> Every downstream count would be meaningless. We use khmer-nltk, a CRF
> segmenter trained on Khmer.
>
> **Stemming and lemmatization — we do neither, and that is the right call.**
> Khmer is an analytic language. Verbs do not conjugate, nouns do not inflect for
> number or case. 'Go', 'goes', 'went' and 'going' are all the same surface form,
> ទៅ. There are no affixes to strip, which is why no Khmer lemmatizer exists —
> not because the tooling is missing, but because there is nothing for it to do.
>
> **Stopwords.** No standard Khmer stopword list ships with NLTK or spaCy, so we
> wrote one: 50 function words — particles, classifiers, pronouns, conjunctions.
> We deliberately kept the negation words. Dropping មិន, 'not', would invert the
> polarity of exactly the sentences labelled negative.
>
> That decision is measurable. In the feature analysis, មិន is the single
> strongest frequent negative signal, appearing 68 times in the training set. If
> we had treated it as a stopword we would have deleted our best feature."

---

## SLIDE 2b (or backup) — Dictionary vs CRF segmentation

**Bullets**
- Two ways to find word boundaries in Khmer
- Dictionary: longest match against a wordlist
- CRF: statistical model, predicts each boundary
- We use CRF — here is why

**Visual — this comparison:**

| | Tokens | Vocabulary | Unmatched |
|---|---|---|---|
| Dictionary (34,398 words) | 14,488 | 3,143 | **15.2%** |
| CRF (khmer-nltk) | 11,615 | 3,034 | — |

Only **3 of 400** sentences were segmented identically by both.

**The example that settles it** — the number "2,073":

```
dictionary   ២ · , · ០ · ៧ · ៣        six fragments
CRF          ២,០៧៣                    one token
```

**Say (~45 sec):**

> "There are two standard approaches. **Dictionary-based** maximum matching walks
> left to right and takes the longest string that appears in a Khmer wordlist —
> deterministic, and you can point at the dictionary and say why a cut was made.
> **CRF** is a statistical model trained on hand-segmented Khmer that predicts
> each boundary from surrounding characters.
>
> We implemented both. Using a 34,000-word SIL dictionary, 15% of positions had
> no dictionary match at all, and the two methods agreed on only 3 of our 400
> sentences.
>
> This example is the clearest: the number two-thousand-and-seventy-three. The
> dictionary shatters it into six fragments because the digits aren't in the
> wordlist. The CRF keeps it as one token.
>
> Our corpus is news text — full of names, places and numbers, exactly what a
> fixed wordlist misses. So we use the CRF, and keep the dictionary
> implementation as the comparison that shows what the CRF is buying us."

---

## The numbers

| | |
|---|---|
| Sentences annotated | 600 (3 pairs × 200) |
| Pairs with two annotators | 2 |
| Both annotators agreed | 212 / 400 (53.0%) |
| Dataset used for training | 400 (disagreements resolved via corpus label) |
| Khmer stopwords defined | 50, minus 9 negation/degree words kept |
| Vocabulary after cleaning | 2,330 words |

---

## Questions

**"Why is your Kappa so low?"**
> "Three-class sentiment is intrinsically harder than binary, and the confusion
> matrix shows the disagreement is almost entirely neutral versus a weak
> polarity — not positive versus negative. Our guideline defined the labels but
> underspecified the threshold for 'evaluative enough to count'. That is a
> guideline problem, and the fix is to tighten section 5 and re-annotate."

**"Why didn't you use stemming or lemmatization?"**
> "Because Khmer has no inflection to remove. It is an analytic language — verbs
> don't conjugate, nouns don't take plural or case endings. Applying an English
> stemmer would cut characters off Khmer words and corrupt them without removing
> any affix, because there are no affixes."

**"How does the segmenter work?"**
> "khmer-nltk uses a conditional random field trained on segmented Khmer. It
> predicts, for each character position, whether a word boundary falls there. We
> use it as a library rather than reimplementing it."

**"Why keep the disagreed sentences at all?"**
> "Because keeping only agreements biases the dataset toward easy cases. The 212
> agreed sentences are the obvious ones; the hard ones are exactly what a
> classifier needs to see. We report both datasets so the effect is visible
> rather than hidden."

**"What happened to pair 3?"**
> "One annotator submitted, the other has not yet. With a single annotator there
> is no agreement to measure, so those 200 sentences are excluded from the
> dataset. They can be added once the second file arrives."
