import os
import json
import pickle
from glob import glob

from utils.pdf_util import extract_lines_from_pdf
from utils.feature_utils import document_to_feature_sequence

from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

DATA_DIR = "data/synthetic"


def load_dataset():
    pdf_files = sorted(glob(os.path.join(DATA_DIR, "*.pdf")))
    X, y = [], []
    for pdf_file in pdf_files:
        label_file = pdf_file.replace(".pdf", ".labels.json")
        if not os.path.exists(label_file):
            continue

        lines = extract_lines_from_pdf(pdf_file)
        features = document_to_feature_sequence(lines)

        with open(label_file, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
            labels = [entry["label"] for entry in labels_data]

        if len(features) != len(labels):
            print(f"Warning: Mismatch in {pdf_file}")
            continue

        X.append(features)
        y.append(labels)

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
    print("Loading data...")
    X, y = load_dataset()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training CRF model...")
    crf = train_crf(X_train, y_train)

    print("Evaluating...")
    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    with open("models/crf_model.pkl", "wb") as f:
        pickle.dump(crf, f)
    print("Model saved to models/crf_model.pkl")
