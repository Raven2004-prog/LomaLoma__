# task1a_output.py

import os
import json
import pickle
from glob import glob
from tqdm import tqdm

from utils.pdf_util import extract_lines_from_pdf
from utils.feature_utils import document_to_feature_sequence

MODEL_PATH = "models/crf_model.pkl"
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_outline_for_file(model, pdf_path):
    basename = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        lines = extract_lines_from_pdf(pdf_path)
        if not lines:
            raise ValueError("No lines extracted from PDF.")

        features = document_to_feature_sequence(lines)
        labels = model.predict_single(features)

        outline = []
        for line, label in zip(lines, labels):
            if label != "BODY":
                outline.append({
                    "level": label.upper(),
                    "text": line["text"].strip(),
                    "page": line["page"]
                })

        return {
            "title": basename,
            "outline": outline
        }

    except Exception as e:
        print(f"⚠️ Failed to process {basename}: {e}")
        return {
            "title": basename,
            "outline": []
        }

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model = load_model()

    pdf_files = sorted(glob(os.path.join(INPUT_FOLDER, "*.pdf")))
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        result = predict_outline_for_file(model, pdf_path)
        out_file = os.path.join(OUTPUT_FOLDER, os.path.splitext(os.path.basename(pdf_path))[0] + ".json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(f"\n✅ Processed {len(pdf_files)} PDFs into '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    main()
