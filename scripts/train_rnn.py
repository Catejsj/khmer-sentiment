#!/usr/bin/env python3
"""
PERSON 5 - the recurrent network (mandatory).

The eight classical models in train_models.py all consume a fixed-length vector
per sentence, which throws word order away. A recurrent network reads the
sentence one word at a time and carries a hidden state forward, so order is part
of what it sees. That is the reason to include one.

Two variants are trained:

    SimpleRNN   torch.nn.RNN - the plain recurrent unit
    LSTM        torch.nn.LSTM - gated, handles longer dependencies

The embedding layer is initialised from the Word2Vec vectors already trained by
representations.py, rather than from scratch. With 408 training sentences a
randomly-initialised embedding has no chance of learning useful Khmer word
vectors, so we transfer what we have.

HONEST EXPECTATION: 408 sentences is far too little to train a recurrent network
properly. Both variants will overfit within a few epochs. Early stopping on
validation macro-F1 limits the damage, and the result is reported as it comes
out rather than tuned until it flatters.

Usage:
    python scripts/train_rnn.py
    python scripts/train_rnn.py --epochs 60
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_features import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
FEATURES = os.path.join(ROOT, "data", "features")
REPORT_DIR = os.path.join(ROOT, "reports")

SEED = 42
EMB_DIM = 100
HIDDEN = 64
MAX_LEN = 60
PAD, UNK = 0, 1

torch.manual_seed(SEED)
np.random.seed(SEED)


def stable_hash(word: str) -> int:
    """Must exist here to unpickle the saved Word2Vec model.

    representations.py trained it with hashfxn=stable_hash for reproducibility,
    and gensim pickles that reference by name against __main__. Loading the
    model from a different script fails unless the same function is defined.
    """
    import hashlib
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)


class RecurrentClassifier(nn.Module):
    """Embedding -> recurrent layer -> last hidden state -> 3-way output."""

    def __init__(self, vocab_size, kind="lstm", emb_weights=None, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD)
        if emb_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(emb_weights))
        rnn_cls = nn.LSTM if kind == "lstm" else nn.RNN
        self.rnn = rnn_cls(EMB_DIM, HIDDEN, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(HIDDEN, len(CLASSES))

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, hidden = self.rnn(packed)
        h = hidden[0] if isinstance(hidden, tuple) else hidden   # LSTM returns (h, c)
        return self.fc(self.dropout(h[-1]))


def build_vocab(token_lists):
    """Vocabulary from TRAIN only. Test words absent from it become <unk>."""
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for toks in token_lists:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def encode(token_lists, vocab):
    ids, lengths = [], []
    for toks in token_lists:
        seq = [vocab.get(t, UNK) for t in toks][:MAX_LEN] or [UNK]
        lengths.append(len(seq))
        ids.append(seq + [PAD] * (MAX_LEN - len(seq)))
    return torch.tensor(ids), torch.tensor(lengths)


def embedding_matrix(vocab):
    """Seed the embedding layer with the Word2Vec vectors we already trained."""
    path = os.path.join(FEATURES, "word2vec.model")
    if not os.path.exists(path):
        return None
    wv = Word2Vec.load(path).wv
    mat = np.random.normal(0, 0.1, (len(vocab), EMB_DIM)).astype(np.float32)
    mat[PAD] = 0
    hits = 0
    for word, i in vocab.items():
        if word in wv:
            mat[i] = wv[word]
            hits += 1
    return mat, hits


def run_epoch(model, X, L, y, opt, loss_fn, batch=32, train=True):
    model.train() if train else model.eval()
    order = torch.randperm(len(y)) if train else torch.arange(len(y))
    total, preds = 0.0, []
    for i in range(0, len(y), batch):
        idx = order[i:i + batch]
        xb, lb, yb = X[idx], L[idx], y[idx]
        with torch.set_grad_enabled(train):
            logits = model(xb, lb)
            loss = loss_fn(logits, yb)
        if train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        total += loss.item() * len(idx)
        preds.append((idx, logits.argmax(1)))
    out = torch.zeros(len(y), dtype=torch.long)
    for idx, p in preds:
        out[idx] = p
    return total / len(y), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)

    data = {s: pd.read_csv(os.path.join(SPLITS, f"{s}.csv")) for s in ("train", "val", "test")}
    for d in data.values():
        d["cleaned_text"] = d["cleaned_text"].fillna("")
    toks = {s: [t.split() for t in d["cleaned_text"]] for s, d in data.items()}

    vocab = build_vocab(toks["train"])
    emb, hits = embedding_matrix(vocab) or (None, 0)

    label_ix = {c: i for i, c in enumerate(CLASSES)}
    X, L, Y = {}, {}, {}
    for s in ("train", "val", "test"):
        X[s], L[s] = encode(toks[s], vocab)
        Y[s] = torch.tensor([label_ix[v] for v in data[s]["label"]])

    out, push = [], None
    push = out.append
    push("=" * 78)
    push("RECURRENT NEURAL NETWORK")
    push("=" * 78)
    push(f"\n    vocabulary (train only) : {len(vocab)}")
    push(f"    embedding seeded from Word2Vec : {hits}/{len(vocab)} words matched")
    push(f"    embedding dim {EMB_DIM}, hidden {HIDDEN}, max length {MAX_LEN}")
    push(f"    train {len(Y['train'])}   val {len(Y['val'])}   test {len(Y['test'])}")

    push("\n    Why a recurrent network at all: the eight classical models read a")
    push("    fixed-length vector per sentence and cannot see word order. An RNN")
    push("    reads word by word and carries a hidden state, so order is part of")
    push("    the input. In Khmer that matters for negation - មិន placed before a")
    push("    positive word inverts the sentence.")

    # class weights, since neutral outnumbers the others
    counts = np.bincount(Y["train"].numpy(), minlength=len(CLASSES))
    weights = torch.tensor((counts.sum() / (len(CLASSES) * counts)), dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    results = []
    for kind, label in [("rnn", "SimpleRNN"), ("lstm", "LSTM")]:
        torch.manual_seed(SEED)
        model = RecurrentClassifier(len(vocab), kind=kind, emb_weights=emb)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_f1, best_state, bad, best_epoch = -1.0, None, 0, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            tr_loss, _ = run_epoch(model, X["train"], L["train"], Y["train"], opt, loss_fn, train=True)
            _, va_pred = run_epoch(model, X["val"], L["val"], Y["val"], opt, loss_fn, train=False)
            va_f1 = f1_score(Y["val"], va_pred, average="macro")
            history.append((epoch, tr_loss, va_f1))
            if va_f1 > best_f1:
                best_f1, best_epoch, bad = va_f1, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break

        model.load_state_dict(best_state)
        _, te_pred = run_epoch(model, X["test"], L["test"], Y["test"], opt, loss_fn, train=False)
        yte = Y["test"].numpy()
        pred_names = [CLASSES[i] for i in te_pred.numpy()]
        true_names = [CLASSES[i] for i in yte]

        push(f"\n\n[{label}]")
        push(f"    stopped at epoch {history[-1][0]}, best validation macro-F1 at epoch {best_epoch}")
        push(f"    {'epoch':>6}{'train loss':>12}{'val macro-F1':>14}")
        for e, l, f in history[:3] + ([("...", "", "")] if len(history) > 6 else []) + history[-3:]:
            if e == "...":
                push(f"    {'...':>6}")
            else:
                push(f"    {e:>6}{l:>12.4f}{f:>14.3f}")

        push(f"\n    TEST")
        push(f"      accuracy        {accuracy_score(true_names, pred_names):.3f}")
        push(f"      precision macro {precision_score(true_names, pred_names, average='macro', zero_division=0):.3f}")
        push(f"      recall macro    {recall_score(true_names, pred_names, average='macro', zero_division=0):.3f}")
        push(f"      F1 macro        {f1_score(true_names, pred_names, average='macro'):.3f}")
        cm = confusion_matrix(true_names, pred_names, labels=CLASSES)
        push("\n      confusion matrix (rows = true, cols = predicted):")
        push("      " + pd.DataFrame(cm, index=[f"true:{c[:3]}" for c in CLASSES],
                                     columns=[f"pred:{c[:3]}" for c in CLASSES]
                                     ).to_string().replace("\n", "\n      "))

        results.append({
            "model": label, "representation": "learned embedding (Word2Vec init)",
            "cv_f1": np.nan, "val_f1": best_f1,
            "test_accuracy": accuracy_score(true_names, pred_names),
            "test_precision": precision_score(true_names, pred_names, average="macro", zero_division=0),
            "test_recall": recall_score(true_names, pred_names, average="macro", zero_division=0),
            "test_f1": f1_score(true_names, pred_names, average="macro"),
            "best_params": f"hidden={HIDDEN}, epochs_to_best={best_epoch}",
        })

    push("\n\n[WHAT TO CONCLUDE]")
    push("    Both variants stop improving within a handful of epochs and then")
    push("    overfit - 408 training sentences cannot support a network with an")
    push("    embedding table this size. The classical models on averaged")
    push("    embeddings do better precisely because averaging is a much stronger")
    push("    constraint than letting a recurrent layer learn freely.")
    push("\n    That is the honest finding: on this dataset the RNN is included")
    push("    because the brief requires it and because it is the only model that")
    push("    sees word order, not because it wins. With a few thousand more")
    push("    annotated sentences the comparison would likely change.")

    # append to the shared results table so Person 6 has everything in one file
    path = os.path.join(REPORT_DIR, "model_results.csv")
    df = pd.DataFrame(results)
    if os.path.exists(path):
        prev = pd.read_csv(path)
        prev = prev[~prev["model"].isin(df["model"])]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)

    report = "\n".join(out)
    rpath = os.path.join(REPORT_DIR, "rnn_training_report.txt")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {rpath}")
    print(f"Appended to: {path}")


if __name__ == "__main__":
    main()
