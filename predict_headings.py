# predict_headings.py

import json
import joblib
import numpy as np
from pathlib import Path

# Load model and label encoder
model = joblib.load("models/heading_classifier.joblib")
le = joblib.load("models/label_encoder.joblib")

# Load features from JSON
with open("output/features.json", "r", encoding="utf-8") as f:
    features = json.load(f)

# Convert to feature vectors
def feature_vector(feat):
    return [
        feat["font_size"],
        feat["line_width"],
        feat["line_height"],
        feat["char_count"],
        feat["y_position"],
    ]

X = [feature_vector(f) for f in features]

# Predict labels
preds = model.predict(X)
labels = le.inverse_transform(preds)

# Build output JSON
outline = []
title = None

for f, label in zip(features, labels):
    if label in {"H1", "H2", "H3"}:
        if title is None and label == "H1":
            title = f["text"]
        outline.append({
            "level": label,
            "text": f["text"],
            "page": f["page"]
        })

result = {
    "title": title if title else "Untitled Document",
    "outline": outline
}

# Save to output.json
with open("output.json", "w", encoding="utf-8") as out:
    json.dump(result, out, indent=2, ensure_ascii=False)

print("✅ Headings extracted to output.json")
