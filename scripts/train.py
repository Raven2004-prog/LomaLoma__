import os
import json
import pickle
from glob import glob
from collections import defaultdict

from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import train_test_split

# ======== CONFIGURATION =========
LABEL_JSON_PATH = r"C:\documents\adobe_1a\LomaLoma__\data\synthetic_combined.json"
# =================================

def group_tokens_into_lines(entries):
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        page = entry.get("page", 0)
        y_pos = round(entry.get("y_position", 0), 1)
        grouped[page][y_pos].append(entry)

    lines = []
    for page in sorted(grouped.keys()):
        for y_pos in sorted(grouped[page].keys(), reverse=True):
            tokens = sorted(
                grouped[page][y_pos],
                key=lambda x: x.get("x_position", 0)
            )
            lines.append(tokens)
    return lines

def extract_features(token):
    # build raw features
    feat = {
        'word': token.get("text", ""),
        'lower': token.get("text", "").lower(),
        'length': len(token.get("text", "")),
        'font_size': token.get("font_size", 0),
        'line_width': token.get("line_width", 0),
        'line_height': token.get("line_height", 0),
        'char_count': token.get("char_count", 0),
        'y_position': token.get("y_position", 0),
        'page': token.get("page", 0),
        'is_upper': token.get("text", "").isupper(),
        'is_title': token.get("text", "").istitle(),
        'is_digit': token.get("text", "").isdigit(),
        'prefix1': token.get("text", "")[:1],
        'prefix2': token.get("text", "")[:2],
        'suffix1': token.get("text", "")[-1:],
        'suffix2': token.get("text", "")[-2:],
    }
    # convert all values to strings
    return {k: str(v) for k, v in feat.items()}

def load_dataset(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("JSON should contain a list")

    lines = group_tokens_into_lines(entries)
    X, y = [], []
    for line in lines:
        feats = []
        labels = []
        for token in line:
            lbl = token.get("label")
            # skip tokens without text or label
            if lbl is None or token.get("text") is None:
                continue
            feats.append(extract_features(token))
            labels.append(str(lbl))
        if feats:
            X.append(feats)
            y.append(labels)

    print(f"Loaded {len(X)} sequences ({sum(len(s) for s in X)} tokens total)")
    return X, y

def train_crf(X_train, y_train):
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf

if __name__ == "__main__":
    print(f"Loading data from JSON: {LABEL_JSON_PATH}")
    X, y = load_dataset(LABEL_JSON_PATH)
    if not X:
        print("No data found; exiting.")
        exit(1)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training on {len(X_train)} sequences...")
    crf = train_crf(X_train, y_train)

    print("Evaluating on test set:")
    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(y_test, y_pred, digits=3))

    os.makedirs("models", exist_ok=True)
    with open("models/crf_model.pkl", "wb") as f:
        pickle.dump(crf, f)
    print("Model saved to models/crf_model.pkl")
