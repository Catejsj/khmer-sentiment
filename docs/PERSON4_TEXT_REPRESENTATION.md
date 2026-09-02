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
> **Bag-of-Words** counts words. One column per word, 1,187 of them. It throws away
> word order, which matters here: មិន ល្អ — 'not good' — and ល្អ — 'good' — both
> contain ល្អ, so Bag-of-Words can't separate them.
>
> **N-grams** fix that by adding word pairs, so 'មិន ល្អ' becomes its own feature.
> That takes us to 1,565 features.
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
> On the positive side: ប្រឹងប្រែង — to strive, to make an effort. ប្រសើរឡើង —
> to improve. ធានា — to guarantee. ទទួលបាន — to gain, to receive.
>
> On the negative side: ប៉ះពាល់ — to affect adversely. ប្រឈម — to face, to
> confront. ពេក — excessively. ចោល — to discard.
>
> [PAUSE]
>
> And the strongest frequent negative signal is **មិន — 'not' — appearing 90
> times in training, 64 of them in negative sentences.** That is direct evidence for Person 3's decision to
> keep negation out of the stopword list. If it had been treated as a stopword,
> we would have deleted our single best negative feature.
>
> So the features do carry sentiment. Whether the models can use it on only 408
> training sentences is Person 5 and 6's slide."

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
> we only call `transform`. That is the whole no-leakage argument."

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
word vectors are averaged and normalised.

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
