# postprocess.py

import sys
import os
import json
import pickle
from utils.pdf_util import extract_lines_from_pdf
from utils.feature_utils import document_to_feature_sequence

MODEL_PATH = "models/crf_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_labels(model, lines):
    features = document_to_feature_sequence(lines)
    predictions = model.predict_single(features)
    return predictions

def build_outline(lines, labels):
    outline = []
    stack = []

    for line, label in zip(lines, labels):
        if label == "BODY":
            continue

        level = label.upper()
        entry = {
            "level": level,
            "text": line["text"].strip(),
            "page": line["page"]
        }

        while stack and stack[-1]["level"] >= level:
            stack.pop()

        if not stack:
            outline.append(entry)
        else:
            parent = stack[-1]
            parent.setdefault("children", []).append(entry)

        stack.append(entry)

    return outline

def main(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    model = load_model()

    lines = extract_lines_from_pdf(pdf_path)
    labels = predict_labels(model, lines)

    title = basename
    outline = build_outline(lines, labels)

    output = {
        "title": title,
        "outline": outline
    }

    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"{basename}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Outline written to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python postprocess.py <path_to_pdf>")
    else:
        main(sys.argv[1])
