# train.py
import os
import json
import pickle
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset.jsonl"


def load_dataset(jsonl_path="dataset.jsonl"):
    X, y = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                items = json.loads(line)

                if not isinstance(items, list):
                    print(f"⚠️ Skipping line {line_num}: not a list")
                    continue

                # Synthetic format
                if "features" in items[0]:
                    x_seq = [item["features"] for item in items]
                    y_seq = [item["label"].upper() for item in items]

                # Manual format
                else:
                    feature_keys = set(items[0].keys()) - {"text", "label"}
                    x_seq = [{k: item[k] for k in feature_keys} for item in items]
                    y_seq = [item["label"].upper() for item in items]

                if len(x_seq) != len(y_seq):
                    print(f"⚠️ Skipping line {line_num}: X and Y length mismatch")
                    continue

                X.append(x_seq)
                y.append(y_seq)

            except Exception as e:
                print(f"⚠️ Skipping line {line_num} due to error: {e}")

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
    print("Loading dataset...")
    X, y = load_dataset()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training CRF model...")
    crf = train_crf(X_train, y_train)

    print("Evaluating model...")
    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    with open("models/crf_model.pkl", "wb") as f:
        pickle.dump(crf, f)
    print("Model saved to models/crf_model.pkl")
