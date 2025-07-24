#!/usr/bin/env python3
import os
import json
import pickle
import argparse
from collections import defaultdict, Counter

from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

# ─────── CONFIG & ARGS ───────
def get_args():
    p = argparse.ArgumentParser(description="Train a linear-chain CRF over your JSON-extracted tokens.")
    p.add_argument("--input-json", "-i", default="data/synthetic_kusu.json", help="Path to the stitched JSON file of token features.")
    p.add_argument("--model-out", "-m", default="models/crf_model.pkl", help="Where to pickle the trained CRF model.")
    p.add_argument("--report-out", "-r", default="models/classification_report.txt", help="Where to save the classification report.")
    p.add_argument("--test-size", "-t", type=float, default=0.2, help="Fraction of data to reserve for testing.")
    p.add_argument("--c1", type=float, default=0.1, help="L1 penalty in CRF (sparsity).")
    p.add_argument("--c2", type=float, default=0.1, help="L2 penalty in CRF (smoothness).")
    p.add_argument("--max-iter", type=int, default=100, help="Maximum number of iterations for LBFGS.")
    p.add_argument("--cv", action="store_true", help="If set, run GridSearchCV over c1/c2 grid (slower).")
    return p.parse_args()

# ─────── UTILS ───────
def group_tokens_into_lines(entries):
    by_page_y = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if not isinstance(e, dict):
            continue
        page = e.get("page", 0)
        y = round(e.get("y_position", 0), 1)
        by_page_y[page][y].append(e)

    lines = []
    for page in sorted(by_page_y):
        for y in sorted(by_page_y[page], reverse=True):
            tokens = sorted(by_page_y[page][y], key=lambda x: x.get("x_position", 0))
            lines.append(tokens)
    return lines

def extract_features(tokens):
    feats = []
    n = len(tokens)
    for i, tok in enumerate(tokens):
        txt = tok.get("text", "")
        base = {
            "word": txt,
            "lower": txt.lower(),
            "length": len(txt),
            "is_upper": tok.get("is_all_caps", txt.isupper()),
            "is_title": tok.get("is_title_case", txt.istitle()),
            "is_digit": txt.isdigit(),
            "suffix1": txt[-1:] or "",
            "font_size": tok.get("font_size", 0),
            "rel_font_size": tok.get("relative_font_size", 0.0),
            "x_pos": tok.get("x_position", 0.0),
            "y_pos": tok.get("y_position", 0.0),
            "line_height": tok.get("line_height", 0.0),
            "line_space_ratio": tok.get("line_spacing_ratio", 0.0),
            "page_ratio": tok.get("page_position_ratio", 0.0),
            "is_centered": tok.get("is_centered", False),
            "word_count": tok.get("word_count", len(txt.split())),
            "char_count": tok.get("char_count", len(txt)),
            "starts_with_numbering": tok.get("starts_with_numbering", False),
            "ends_with_punctuation": tok.get("ends_with_punctuation", False),
        }
        if len(txt) >= 2:
            base["prefix2"] = txt[:2]
            base["suffix2"] = txt[-2:]
        if i > 0:
            prev = tokens[i - 1]
            base["prev_lower"] = prev.get("text", "").lower()
            base["prev_font"] = prev.get("font_size", 0)
        else:
            base["BOS"] = True
        if i < n - 1:
            nxt = tokens[i + 1]
            base["next_lower"] = nxt.get("text", "").lower()
            base["next_font"] = nxt.get("font_size", 0)
        else:
            base["EOS"] = True
        feats.append({k: str(v) for k, v in base.items()})
    return feats

def load_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No such file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("Expected top-level JSON list.")

    lines = group_tokens_into_lines(entries)
    X, y = [], []
    label_counts = Counter()
    for line in lines:
        feats = extract_features(line)
        labels = [str(tok.get("label", "")) for tok in line]
        paired = [(f, l) for f, l in zip(feats, labels) if l]
        for _, lbl in paired:
            label_counts[lbl] += 1
        if paired:
            X.append([f for f, _ in paired])
            y.append([l for _, l in paired])

    print(f"[+] Loaded {len(X)} lines; {sum(len(s) for s in X)} tokens total.")
    print(f"    Label distribution: {dict(label_counts)}")
    return X, y, list(label_counts.keys())

# ─────── TRAIN & EVAL ───────
def train_and_evaluate(args):
    X, y, all_labels = load_data(args.input_json)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=42)
    print(f"[+] {len(Xtr)} train sequences; {len(Xte)} test sequences")

    def make_crf(c1, c2):
        return CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=args.max_iter,
            all_possible_transitions=True,
            all_possible_states=True
        )

    if args.cv:
        param_grid = {'c1': [0.01, 0.1, 1.0], 'c2': [0.01, 0.1, 1.0]}
        gs = GridSearchCV(
            estimator=make_crf(args.c1, args.c2),
            param_grid=param_grid,
            cv=KFold(3, shuffle=True, random_state=42),
            verbose=1,
            n_jobs=-1,
            scoring='f1_weighted'
        )
        gs.fit(Xtr, ytr)
        crf = gs.best_estimator_
        print(f"[+] GridSearch best params: {gs.best_params_}")
    else:
        crf = make_crf(args.c1, args.c2)
        crf.fit(Xtr, ytr)

    print("[+] Predicting on test set…")
    y_pred = crf.predict(Xte)

    report = metrics.flat_classification_report(
        yte, y_pred,
        labels=all_labels,
        digits=3
    )
    print(report)

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[+] Classification report saved to {args.report_out}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(crf, f)
    print(f"[+] Model pickled to {args.model_out}")

if __name__ == "__main__":
    args = get_args()
    train_and_evaluate(args)
