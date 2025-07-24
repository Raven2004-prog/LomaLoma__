import sys
import json
import os
from predict import predict_labels


def generate_flat_outline(lines, labels, pdf_path):
    outline = []
    title = None

    # find title from first 2 heading-like entries
    heading_blocks = [(line, label) for line, label in zip(lines, labels) if label.startswith("H")]
    for line, label in heading_blocks[:2]:
        text = line['text'].strip()
        if len(text.split()) > 3 and text != "•":
            title = text
            break

    # fallback: use file name
    if not title:
        title = os.path.splitext(os.path.basename(pdf_path))[0]

    for line, label in zip(lines, labels):
        if label in ["H1", "H2", "H3"]:
            text = line['text'].strip()
            if text and text != "•":
                outline.append({
                    "level": label,
                    "text": text,
                    "page": line['page_num']
                })

    return {"title": title, "outline": outline}


if __name__ == "__main__":
    input_dir = r"C:\Users\lenovo\Desktop\loma_final\LomaLoma__\input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")

            lines, labels = predict_labels(pdf_path)
            output = generate_flat_outline(lines, labels, pdf_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)

            print(f"Processed {filename} → {output_path}")