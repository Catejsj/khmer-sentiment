# Person 4 — Text Representation

Two slides, script, and answers. ~3 minutes.

---

## SLIDE 1 — The five representations

**Title:** Text Representation — turning Khmer sentences into numbers

**Bullets**
- Models do arithmetic, not reading
- Five methods, each captures something different
- All fit on **training data only** — no leakage

**Visual — one table, the whole slide:**

| | What it captures | Features | Word order | Unseen words |
|---|---|---|---|---|
| Bag-of-Words | which words appear | 1,187 | ✗ | dropped |
| N-grams (1–2) | + word pairs | 1,565 | partial | dropped |
| TF-IDF (1–2) | + how distinctive | 1,565 | partial | dropped |
| Word2Vec | meaning from context | 100 | ✗ | dropped |
| fastText | + character chunks | 100 | ✗ | **handled** |

**Say (~90 sec):**

> "A model cannot read Khmer, or anything else. It does arithmetic. So my job was
> turning our cleaned sentences into numbers — five different ways, because it
> isn't obvious in advance which works best.
>
> **Bag-of-Words** counts words. One column per word, 1,187 of them. It throws
> away word order, which matters here: មិន ល្អ — 'not good' — and ល្អ — 'good' —
> both contain ល្អ, so Bag-of-Words can't separate them.
>
> **N-grams** fix that by adding word pairs, so 'មិន ល្អ' becomes its own
> feature. That takes us to 1,565 features.
>
> **TF-IDF** keeps the same features but weights instead of counting. Words in
> nearly every sentence get pushed down, distinctive ones lifted.
>
> The last two are different in kind. **Word2Vec** learns meaning from context —
> words used similarly end up close together, in 100 dimensions instead of
> hundreds of columns. **fastText** does the same and also splits each word into
> character chunks, so a word it has never seen still gets a vector.
>
> [PAUSE — this is the number that matters]
>
> That last point is not theoretical for us. **23% of the words in our test set
> never appear in training.** Word2Vec drops nearly a quarter of the test
> vocabulary entirely. fastText builds vectors for them from character pieces.
>
> One thing true of all five: every one was fit on training data only, then
> applied to validation and test. Fitting on everything would leak test
> vocabulary into the features."

---

### If you have time, or get asked, expand any of these

**Bag-of-Words, worked through.** Three toy sentences, the whole idea:

```
                    ល្អ   មិន   ណាស់
"ល្អ ណាស់"  good    1     0      1
"មិន ល្អ"   not     1     1      0
```

Both rows contain ល្អ. To Bag-of-Words they look similar, even though one is
positive and one is negative. That is the failure n-grams exist to fix.

**N-grams.** Adding the pair `មិន ល្អ` as its own column separates them:

```
                    ល្អ   មិន   មិន ល្អ
"ល្អ ណាស់"           1     0       0
"មិន ល្អ"            1     1       1
```

378 bigrams survived our `min_df=2` filter, on top of the 1,187 single words.

**TF-IDF, in one sentence.** Term Frequency × Inverse Document Frequency —
how often a word appears *here*, divided by how many sentences contain it at
all. A word in every sentence scores near zero; a word in three sentences scores
high. Raw counts say the most common word is the most important; TF-IDF says the
opposite.

**Word2Vec.** Slide a five-word window along the text. Given a word, predict its
neighbours. Words that keep similar company end up with similar vectors. 100
numbers per word, then averaged to 100 numbers per sentence.

**fastText.** Same, but each word is also split into 3-to-6 character chunks. A
word's vector is the sum of its chunks, so an unseen word still gets one built
from pieces it shares with words that were seen.

---

## SLIDE 2 — What the features contain

**Title:** Do the features carry sentiment?

**Bullets**
- (none — the figure is the slide)

**Visual:** `reports/figures/fig1_sentiment_words.png`, full bleed

**Say (~90 sec):**

> "This is every word scored by how strongly it belongs to one class. Blue is
> disproportionately positive, red disproportionately negative.
>
> And unlike a lot of these analyses, these are **real sentiment words**.
>
> On the positive side: ប្រកបដោយ — 'endowed with', which prefixes praise.
> អប់រំ and ការអប់រំ — 'to educate' and 'education'. ជួយ — 'to help'.
>
> On the negative side: បញ្ហា — 'problem'. ប៉ះពាល់ — 'to affect adversely'.
> ករណី — 'case', as in an incident. ពេក — 'excessively'.
>
> [PAUSE]
>
> These are not topic words that happen to correlate. They are the words a Khmer
> speaker would point at if you asked which parts of the sentence carry the
> feeling. That is the evidence the features work.
>
> And separately — មិន, 'not', appears **90 times in training, 64 of them in
> negative sentences.** That is direct evidence for Person 3's decision to keep
> negation out of the stopword list. Treating it as a stopword would have deleted
> our single best negative feature.
>
> So the features do carry sentiment. Whether the models can use it on only 408
> training sentences is Person 5 and 6's slide."

---

## ⭐ Where slide 2's numbers come from

The `+2.2`, `−2.4` beside each bar are **log-odds**. This is the follow-up most
likely to be asked, so know it cold.

### What log-odds means

> "It's how many times more likely that word is to appear in one class than the
> other. Zero means equally common in both. The further from zero, the more
> lopsided."

Because it's a logarithm, small numbers mean big differences:

| log-odds | means | |
|---|---|---|
| 0 | 1× | equally common |
| 1 | ~3× | more likely |
| 2 | ~7× | more likely |
| 2.4 | ~11× | more likely |

### The actual bars, with their raw counts

| Word | Meaning | On the chart | In words | pos | neg |
|---|---|---|---|---|---|
| ប្រកបដោយ | endowed with | **+2.19** | ~9× more likely positive | 8 | **0** |
| អប់រំ | to educate | +2.08 | ~8× more likely positive | 7 | 0 |
| ការអប់រំ | education | +1.94 | ~7× more likely positive | 6 | 0 |
| បញ្ហា | problem | **−2.40** | ~11× more likely negative | **0** | 10 |
| ប៉ះពាល់ | to affect adversely | −2.20 | ~9× more likely negative | 0 | 8 |
| ករណី | case, incident | −1.95 | ~7× more likely negative | 0 | 6 |

> "So បញ្ហា — 'problem' — appears ten times in negative sentences and never once
> in a positive one. ប្រកបដោយ is the mirror image: eight times positive, never
> negative."

**Quoting the raw counts alongside the log-odds is what makes it land.** The bar
height is abstract; "ten times versus zero" is not.

### The formula

```
                  share of that word among POSITIVE sentences
   log-odds = log ──────────────────────────────────────────
                  share of that word among NEGATIVE sentences
```

Computed on the **408 training sentences only** — using test to decide what to
say about the features would be leakage through the back door.

### Three details you will be asked about

**"Why proportions, not raw counts?"**
> "The classes aren't the same size — 2,143 word occurrences in positive
> sentences against 2,130 in negative. Close here, but if one class had twice the
> text every word would look twice as common in it. Log-odds compares shares, so
> class size cancels out."

**"How is it a ratio if one count is zero?"**
> "Add-one smoothing. We add 1 to every count before dividing, otherwise a word
> appearing zero times in one class gives division by zero. It's why បញ្ហា is
> −2.40 and not infinity."

**"Why only some words?"**
> "Minimum 4 occurrences overall. 317 of the 1,187 words meet that. Without the
> threshold the top of the list is just rare words that happened to land in one
> class once."

**"Why a log at all?"**
> "It makes the scale symmetric. Twice as likely each way becomes +0.7 and −0.7 —
> equal distance from zero. A plain ratio gives 2 and 0.5, which doesn't plot
> symmetrically."

---

## The numbers, and where they come from

| Number | Meaning | Derivation |
|---|---|---|
| **1,187** | Bag-of-Words features | words appearing in ≥2 training sentences |
| **1,565** | n-gram / TF-IDF features | 1,187 unigrams + 378 bigrams |
| **100** | embedding dimensions | a chosen hyperparameter, not derived |
| **22.8%** | out-of-vocabulary rate | word occurrences in test absent from training |
| **2,883** | training vocabulary | distinct words the embeddings learned |
| **408 / 72 / 120** | train / val / test | stratified split of 600 |
| **317** | words on the slide-2 shortlist | those appearing ≥4 times overall |

**On the 100:** be honest that it's a choice, not a calculation.

> "That's a hyperparameter — `vector_size=100`. A standard starting point for a
> corpus this small. Larger vectors need more data to fill meaningfully; with 408
> sentences, going higher would mostly add noise."

Inventing a derivation for it invites a question you can't survive.

---

## How it was built

**The three count-based ones** — scikit-learn, same pattern:

```python
vec = CountVectorizer(ngram_range=(1, 1), min_df=2, token_pattern=r"\S+")
X_train = vec.fit_transform(train_text)   # LEARNS the vocabulary
X_val   = vec.transform(val_text)         # only APPLIES it
X_test  = vec.transform(test_text)
```

> "`fit_transform` on train is where the vocabulary is learned. On val and test
> we only call `transform`. A word that only appears in test is simply ignored,
> because the vectoriser was never told it exists. That is the whole no-leakage
> argument."

**One Khmer-specific detail worth mentioning:** `token_pattern=r"\S+"`.
scikit-learn's default pattern assumes English word characters and would drop
Khmer script silently. Our text is already segmented by Person 3's pipeline with
tokens joined by spaces, so we split on whitespace instead.

**The two embeddings** — gensim, train then average:

```python
model = Word2Vec(sentences=train_tokens, vector_size=100,
                 window=5, sg=1, epochs=30, min_count=1)
model.wv.get_mean_vector(tokens, pre_normalize=True, post_normalize=True)
```

The model gives a vector per word; the classifier needs one per sentence, so the
word vectors are averaged. `pre_normalize` scales each word to length 1 first so
one extreme word can't dominate; `post_normalize` scales the finished sentence
vector so long and short sentences stay comparable.

---

## Handoff to Person 5

```python
from load_features import load
X_train, X_val, X_test, y_train, y_val, y_test = load("tfidf")
```

Names: `bow`, `ngram`, `tfidf`, `word2vec`, `fasttext`. Sparse matrices for the
first three, dense arrays for the embeddings — both feed scikit-learn directly,
so one loop covers all five.

`to_numeric(y)` converts the labels to integers if a model needs them.
`feature_names(rep)` gives column names for the count-based three, for Person 6's
feature-importance analysis.

**Baseline already run** (plain Logistic Regression, untuned, validation macro-F1):

| | |
|---|---|
| fastText | 0.436 |
| Bag-of-Words | 0.412 |
| N-grams | 0.412 |
| TF-IDF | 0.396 |
| Word2Vec | 0.363 |

These are **not results** — they prove the files load and train. Person 5 tunes
properly and reports the real numbers. Worth noting that all five sit close
together and only a little above the 0.333 random baseline, which is what 408
training sentences across three classes buys you.

---

## Questions

**"Which representation is best?"**
> "Person 6's slide. After tuning it turned out to be TF-IDF — the three best
> models on test all use it. Our untuned baseline had ranked fastText first,
> which is a good illustration of how little an untuned single run tells you."

**"Why is the OOV rate 23% when English projects see under 10%?"**
> "Two reasons. Our training set is only 408 sentences, and Khmer compounds
> heavily, so the vocabulary grows faster per sentence than English. It is the
> strongest argument for fastText in this project."

**"23% of what exactly?"**
> "Word occurrences in the test set — not distinct words. Of every word token in
> test, 23% are words the training data never contained."

**"Do the 100 dimensions mean anything?"**
> "Not individually. No single dimension is 'positive'. Only distances between
> whole vectors carry meaning, which is why embeddings are harder to explain than
> Bag-of-Words."

**"Why `min_df=2`?"**
> "A word appearing in one sentence cannot generalise — the model would memorise
> that sentence. On a corpus this small that matters more, not less."

**"Why token_pattern=\\S+?"**
> "scikit-learn's default tokenizer pattern is built for English word characters
> and drops Khmer script entirely. Our text is pre-segmented with spaces between
> tokens, so we split on whitespace."

**"Can you show where those numbers come from?"**
> "Yes — one command regenerates the whole analysis."

```bash
./.venv/bin/python scripts/inspect_representations.py
```
