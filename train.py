import os
import json
import pickle
from glob import glob
from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import train_test_split

DATA_DIR = "data/synthetic2"


def load_dataset():
    X, y = [], []
    all_files = glob("data/synthetic*/doc_*.json")

    for file_path in all_files:
        with open(file_path, 'r') as f:
            try:
                lines = json.load(f)
                if not isinstance(lines, list):
                    continue
                features_seq = []
                labels_seq = []
                for entry in lines:
                    try:
                        features = {k: v for k, v in entry.items() if k not in {"label", "text"}}
                        label = entry["label"].upper()
                        features_seq.append(features)
                        labels_seq.append(label)
                    except:
                        continue
                if features_seq and labels_seq and len(features_seq) == len(labels_seq):
                    X.append(features_seq)
                    y.append(labels_seq)
            except:
                continue
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
