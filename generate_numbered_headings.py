import os
import json
import random
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import inch

output_dir = "data/synthetic_numbered"
os.makedirs(output_dir, exist_ok=True)

FONT_NAME = "Helvetica"
BASE_FONT_SIZE = 11
MAX_LINE_CHARS = 30
styles = getSampleStyleSheet()

def make_style(indent=0, bold=False, space_after=6, align=TA_LEFT):
    return ParagraphStyle(
        name=f"Style_{indent}_{bold}_{align}",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=BASE_FONT_SIZE,
        alignment=align,
        spaceAfter=space_after,
        leftIndent=indent
    )

def generate_numbered_heading(level):
    base = random.choice(["Overview", "Objectives", "Scope", "Features", "References", "Goals", "Topics"])
    number = ""
    if level == "H1":
        number = f"{random.randint(1, 9)}. "
    elif level == "H2":
        number = f"{random.randint(1, 9)}.{random.randint(1, 5)} "
    elif level == "H3":
        number = f"{random.randint(1, 9)}.{random.randint(1, 5)}.{random.randint(1, 3)} "
    return (number + base)[:MAX_LINE_CHARS]

def random_block_sequence():
    sequence = []
    for _ in range(random.randint(10, 16)):
        sequence.append(random.choices(["H1", "H2", "H3", "BODY"], weights=[0.25, 0.25, 0.25, 0.25])[0])
    return sequence

def simulate_numbered_pdf(doc_id):
    story = []
    raw_lines = []

    for block_type in random_block_sequence():
        align = random.choice([TA_LEFT, TA_CENTER])
        indent = 0 if align == TA_CENTER else random.choice([0, 15, 30])

        if block_type == "H1":
            text = generate_numbered_heading("H1")
            style = make_style(indent, bold=True, space_after=14, align=align)
        elif block_type == "H2":
            text = generate_numbered_heading("H2")
            style = make_style(indent, bold=False, space_after=10, align=align)
        elif block_type == "H3":
            text = generate_numbered_heading("H3")
            style = make_style(indent, bold=False, space_after=8, align=align)
        else:
            text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            style = make_style(indent, bold=False, space_after=4, align=align)

        story.append(Paragraph(text, style))
        story.append(Spacer(1, 0.05 * inch))
        raw_lines.append((text, block_type, BASE_FONT_SIZE))

    return story, raw_lines

def extract_features(line, page_height, page_width, median_font_size, prev_line=None):
    text = line["text"]
    y0 = line["y0"]
    x0 = line["x0"]
    width = line["x1"] - x0

    return {
        "y_position": round(y0, 2),
        "width_ratio": round(width / page_width, 2),
        "vertical_gap": round(y0 - prev_line["y1"], 2) if prev_line else 0.0,
        "font_size_ratio": round(line["font_size"] / median_font_size, 2),
        "has_numeric_prefix": int(bool(re.match(r'^[0-9]+(\.[0-9]+)*', text))),
        "word_count": len(text.strip().split()),
        "center_deviation": round(abs((page_width / 2) - (x0 + width / 2)) / page_width, 2),
        "size_vs_prev": round(line["font_size"] / prev_line["font_size"], 2) if prev_line else 1.0,
        "page_top_distance": round(y0 / page_height, 2)
    }

def match_label(text, raw_lines):
    for raw_text, label, _ in raw_lines:
        if text.strip() in raw_text.strip():
            return label
    return "BODY"

def generate_and_save_features(doc_id):
    import fitz  # PyMuPDF

    pdf_path = os.path.join(output_dir, f"doc_{doc_id}.pdf")
    json_path = os.path.join(output_dir, f"doc_{doc_id}.json")
    story, raw_lines = simulate_numbered_pdf(doc_id)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(story)

    pdf = fitz.open(pdf_path)
    json_entries = []

    for page_num, page in enumerate(pdf):
        blocks = page.get_text("dict")["blocks"]
        lines = []

        for block in blocks:
            for line in block.get("lines", []):
                text = "".join([span["text"] for span in line["spans"]]).strip()
                if len(re.sub(r'[^A-Za-z0-9]', '', text)) < 3:
                    continue
                span = line["spans"][0]
                lines.append({
                    "text": text,
                    "font_size": span["size"],
                    "x0": span["bbox"][0],
                    "x1": span["bbox"][2],
                    "y0": span["bbox"][1],
                    "y1": span["bbox"][3],
                    "page": page_num,
                    "page_width": page.rect.width,
                    "page_height": page.rect.height
                })

        font_sizes = [l["font_size"] for l in lines]
        median = sum(font_sizes) / len(font_sizes) if font_sizes else BASE_FONT_SIZE
        prev = None

        for line in lines:
            features = extract_features(line, line["page_height"], line["page_width"], median, prev)
            matched_label = match_label(line["text"], raw_lines)
            json_entries.append({
                "text": line["text"],
                "label": matched_label,
                **features
            })
            prev = line

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_entries, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()

    for i in range(args.num):
        generate_and_save_features(i)

    print(f"{args.num} randomized numbered PDFs created in {output_dir}")
