import os
import json
import pickle
from glob import glob
from collections import defaultdict

from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import train_test_split

# ======== CONFIGURATION =========
# Path to your labels JSON file
LABEL_JSON_PATH = r"C:\documents\adobe_1a\LomaLoma__\data\synthetic_combined.json"  # CHANGE THIS TO YOUR JSON PATH
# =================================

def group_tokens_into_lines(entries):
    """
    Group tokens into lines based on page and y_position
    """
    # Group tokens by page and line (using y_position)
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        page = entry.get("page", 0)
        y_pos = round(entry.get("y_position", 0), 1)  # Group by rounded y_position
        grouped[page][y_pos].append(entry)
    
    # Sort lines by y_position (top to bottom) and tokens by x_position (left to right)
    lines = []
    for page in sorted(grouped.keys()):
        for y_pos in sorted(grouped[page].keys(), reverse=True):
            # Sort tokens by x_position if available, otherwise by appearance
            tokens = sorted(grouped[page][y_pos], 
                           key=lambda x: x.get("x_position", 0))
            lines.append(tokens)
    return lines

def extract_features(token):
    """Extract features from a token dictionary"""
    text = token.get("text", "")
    return {
        'word': text,
        'word.lower()': text.lower(),
        'word.length': len(text),
        'font_size': token.get("font_size", 0),
        'line_width': token.get("line_width", 0),
        'line_height': token.get("line_height", 0),
        'char_count': token.get("char_count", 0),
        'y_position': token.get("y_position", 0),
        'page': token.get("page", 0),
        'is_upper': text.isupper(),
        'is_title': text.istitle(),
        'is_digit': text.isdigit(),
        'prefix1': text[:1] if text else '',
        'prefix2': text[:2] if len(text) >= 2 else '',
        'suffix1': text[-1:] if text else '',
        'suffix2': text[-2:] if len(text) >= 2 else '',
    }

def load_dataset(json_path):
    """Load dataset from JSON with per-token features"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading JSON: {e}")
    
    if not isinstance(entries, list):
        raise ValueError("JSON should contain a list of token objects")
    
    # Group tokens into lines
    lines = group_tokens_into_lines(entries)
    
    # Extract features and labels
    X, y = [], []
    for line in lines:
        line_features = []
        line_labels = []
        for token in line:
            if "text" not in token or "label" not in token:
                continue
            line_features.append(extract_features(token))
            line_labels.append(token["label"])
        
        if line_features and line_labels:
            X.append(line_features)
            y.append(line_labels)
    
    print(f"Loaded {len(X)} lines with {sum(len(line) for line in X)} tokens")
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
        print("No valid data loaded. Exiting.")
        exit(1)
        
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training CRF model on {len(X_train)} sequences...")
    crf = train_crf(X_train, y_train)

    print("Evaluating model...")
    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    with open("models/crf_model.pkl", "wb") as f:
        pickle.dump(crf, f)
    print("Model saved to models/crf_model.pkl")