import fitz  # PyMuPDF
import numpy as np
import re


def clean_text(text):
    # Remove whitespace and special characters (including dots)
    return re.sub(r'[^A-Za-z0-9]', '', text.strip())


def extract_lines_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        lines = []

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = "".join([span["text"] for span in line["spans"]]).strip()
                if not text:
                    continue

                # Strip special characters and filter out short lines
                cleaned = clean_text(text)
                if len(cleaned) < 3:
                    continue

                span = line["spans"][0]
                x0 = span["bbox"][0]
                x1 = span["bbox"][2]
                width = x1 - x0

                lines.append({
                    "text": text,
                    "font_size": span["size"],
                    "x0": x0,
                    "x1": x1,
                    "width": width,
                    "y0": span["bbox"][1],
                    "y1": span["bbox"][3],
                    "page": page_num,
                    "page_width": page.rect.width,
                    "page_height": page.rect.height,
                })

        # Sort lines top to bottom
        lines.sort(key=lambda l: l["y0"])
        font_sizes = [line["font_size"] for line in lines]
        median_font = np.median(font_sizes) if font_sizes else 1.0

        for i, line in enumerate(lines):
            prev = lines[i - 1] if i > 0 else None

            vertical_gap = line["y0"] - prev["y1"] if prev else 0
            size_vs_prev = line["font_size"] / prev["font_size"] if prev else 1.0

            features = {
                "y_position": round(line["y0"], 2),
                "width_ratio": round(line["width"] / line["page_width"], 2),
                "vertical_gap": round(vertical_gap, 2),
                "font_size_ratio": round(line["font_size"] / median_font, 2),
                "has_numeric_prefix": int(bool(re.match(r'^[0-9]+(\.[0-9]+)*', line["text"]))),
                "word_count": len(line["text"].strip().split()),
                "center_deviation": round(abs((line["page_width"] / 2) - (line["x0"] + line["width"] / 2)) / line["page_width"], 2),
                "size_vs_prev": round(size_vs_prev, 2),
                "page_top_distance": round(line["y0"] / line["page_height"], 2)
            }

            line.update(features)  # Flatten all features into line


        all_lines.extend(lines)

    return all_lines
