# Annotation Guideline — Khmer Sentiment Corpus

**Version 1.0** · Sentiment polarity annotation for Khmer sentences

---

## 1. Purpose and scope

This document defines the labelling scheme, decision procedure and quality
criteria for sentiment annotation of Khmer-language sentences drawn from the
`kh-polarity` corpus (Ye Kyaw Thu et al.).

Annotation is performed in **pairs**. Both members of a pair independently label
the same 200 sentences without consulting each other. Inter-annotator agreement
is then measured with Cohen's Kappa, and disagreements are resolved by
discussion between the pair.

---

## 2. Label set

Each sentence receives **exactly one** of three labels.

| Label | Definition |
|---|---|
| **positive** | The sentence expresses approval, satisfaction, success, improvement, praise, or a favourable outcome. |
| **neutral** | The sentence reports information without evaluative stance — factual statements, descriptions, procedural or administrative content. |
| **negative** | The sentence expresses criticism, dissatisfaction, harm, loss, failure, conflict, or an unfavourable outcome. |

No sentence may be left blank. No sentence may receive two labels.

---

## 3. Unit of annotation

The unit is the **full sentence as displayed**, judged on its own.

Annotators must not:

- consult the surrounding article or source document
- search for the sentence online
- infer sentiment from what they know about the topic or the people named

The label reflects the sentiment expressed **in the text presented**, not the
annotator's own view of the subject matter.

---

## 4. Decision procedure

Apply these tests in order and stop at the first that resolves the sentence.

**Step 1 — Is there an evaluative expression?**
Look for words that praise, criticise, or express a judgement. If none are
present and the sentence merely reports facts, label **neutral**.

**Step 2 — What is the direction of the evaluation?**
If the evaluation is favourable → **positive**. If unfavourable → **negative**.

**Step 3 — Is the evaluation negated?**
Khmer negation (មិន, ពុំ, អត់, គ្មាន) inverts polarity. *"មិនល្អ"* ("not good")
is **negative**, not positive. Read the negation carefully; it frequently
appears at a distance from the word it negates.

**Step 4 — Mixed sentiment.**
Where a sentence contains both favourable and unfavourable evaluation, label the
**dominant** one — the sentiment the sentence as a whole leaves the reader with.
If neither dominates, label **neutral**.

**Step 5 — Genuine uncertainty.**
If the sentence remains ambiguous after the steps above, label **neutral** and
record the reason in the notes column. Do not guess between positive and
negative.

---

## 5. Adjudicated conventions

The following cases are decided in advance. They are the recurring sources of
disagreement, and fixing them beforehand is what keeps agreement measurable.

| Case | Label | Rationale |
|---|---|---|
| Factual report of a negative event (accident, arrest, decline) with no evaluative language | **negative** | The event itself carries the polarity. |
| Factual report of an achievement (award, growth, completion) | **positive** | As above, in the other direction. |
| Administrative or procedural announcement | **neutral** | Reports process, not evaluation. |
| Statistics or figures with no framing | **neutral** | Numbers alone are not evaluative. |
| Quoted criticism attributed to a third party | **negative** | The sentence conveys the criticism. |
| Aspiration or plan for improvement ("we will strive to…") | **positive** | Expresses favourable intent. |
| Weather, dates, locations, routine notices | **neutral** | — |

---

## 6. Quality requirements

1. **Annotate all 200 sentences.** Partial files cannot be scored.
2. **Work independently.** Do not discuss individual sentences with your partner
   until both files are submitted. Comparing notes beforehand invalidates the
   agreement measurement entirely.
3. **Use the dropdown.** Type nothing into the LABEL column — select from the
   list. Hand-typed labels introduce casing and whitespace variants
   (`Neutral`, `neutral`, `Positive `) that must be repaired before scoring.
4. **Do not re-save the file as CSV.** Spreadsheet software re-encodes Khmer
   incorrectly, and the text will no longer match your partner's copy.
5. **Do not reorder or sort rows.** The `id` column is what aligns the two
   files.
6. Use the notes column for anything you found genuinely difficult. Those notes
   drive the adjudication discussion.

---

## 7. Agreement and adjudication

After both files are submitted:

1. **Cohen's Kappa** is computed within each pair.
2. Sentences where both annotators agree enter the dataset directly.
3. Sentences where they disagree are returned to the pair for discussion. The
   pair must reach a single agreed label; where they cannot, the sentence is
   excluded rather than settled arbitrarily.

Kappa is interpreted on the Landis & Koch (1977) scale:

| κ | Interpretation |
|---|---|
| < 0.20 | slight |
| 0.21 – 0.40 | fair |
| 0.41 – 0.60 | moderate |
| 0.61 – 0.80 | substantial |
| > 0.80 | almost perfect |

Sentiment annotation on a three-class scheme typically reaches moderate
agreement. A result below `0.40` indicates the guideline is underspecified for
this data rather than that the annotators were careless, and should prompt
revision of §5 before further annotation.

---

## 8. Submission

Save the completed workbook under your own name — `Nita_pair2_annotation.xlsx` —
and return it to the coordinator. Do not overwrite the blank template.
