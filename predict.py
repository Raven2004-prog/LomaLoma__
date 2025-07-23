import sys
import pickle
from utils.pdf_util import extract_lines_from_pdf
from utils.feature_utils import document_to_feature_sequence

MODEL_PATH = "models/crf_model.pkl"

def predict_labels(pdf_path):
    with open(MODEL_PATH, "rb") as f:
        crf = pickle.load(f)

    lines = extract_lines_from_pdf(pdf_path)
    features = document_to_feature_sequence(lines)
    labels = crf.predict([features])[0]  # Single sequence
    return lines, labels

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    lines, labels = predict_labels(pdf_path)
    for line, label in zip(lines, labels):
        print(f"[{label}] {line['text']}")
