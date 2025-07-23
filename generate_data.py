import os
import json
import random
import re
from glob import glob
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import inch

output_dir = "data/synthetic"
os.makedirs(output_dir, exist_ok=True)
OUTPUT_JSONL = os.path.join(output_dir, "synthetic_data.jsonl")

FONT_NAMES = ["Helvetica", "Times-Roman", "Courier"]
ALIGNMENTS = [TA_LEFT, TA_CENTER]

styles = getSampleStyleSheet()

def make_font_name(base, bold, italic):
    if base == "Times-Roman":
        return "Times-BoldItalic" if bold and italic else \
               "Times-Bold" if bold else \
               "Times-Italic" if italic else "Times-Roman"
    elif base == "Courier":
        return "Courier-BoldOblique" if bold and italic else \
               "Courier-Bold" if bold else \
               "Courier-Oblique" if italic else "Courier"
    else:
        return "Helvetica-BoldOblique" if bold and italic else \
               "Helvetica-Bold" if bold else \
               "Helvetica-Oblique" if italic else "Helvetica"

def make_style(base, size, font, align, bold=True, italic=False):
    return ParagraphStyle(
        name=f"{font}_{size}_{bold}_{italic}",
        parent=base,
        fontName=make_font_name(font, bold, italic),
        fontSize=size,
        alignment=align,
        spaceAfter=random.uniform(5, 20),
        leftIndent=random.choice([0, 15, 30])
    )

def extract_features(line, page_height, page_width, median_font_size, prev_line=None):
    text = line["text"]
    y0 = line["y0"]
    x0 = line["x0"]
    width = line["width"]

    features = {
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
    return features

def simulate_pdf_lines(doc_id):
    styles = getSampleStyleSheet()
    font = random.choice(FONT_NAMES)
    align = random.choice(ALIGNMENTS)
    italic = random.choice([True, False])
    base_size = 11

    heading_styles = {
        "TITLE": make_style(styles["Heading1"], 26, font, align, bold=True, italic=italic),
        "H1": make_style(styles["Heading2"], 22, font, align, bold=True, italic=italic),
        "H2": make_style(styles["Heading3"], 18, font, align, bold=False, italic=italic),
        "H3": make_style(styles["Heading4"], 14, font, align, bold=False, italic=italic),
        "BODY": make_style(styles["BodyText"], base_size, font, TA_LEFT, bold=False, italic=italic)
    }

    story = []
    raw_lines = []

    def add_line(text, label, style):
        para = Paragraph(text, style)
        story.append(para)
        story.append(Spacer(1, 0.1 * inch))
        raw_lines.append((text, label, style.fontSize))

    add_line("Sample Document Title", "TITLE", heading_styles["TITLE"])

    for i in range(3):
        add_line(f"{i+1}. Section {i+1}", "H1", heading_styles["H1"])
        for j in range(2):
            add_line(f"{i+1}.{j+1} Subsection", "H2", heading_styles["H2"])
            for k in range(2):
                add_line(f"{i+1}.{j+1}.{k+1} Detail Header", "H3", heading_styles["H3"])
                for _ in range(3):
                    add_line("Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "BODY", heading_styles["BODY"])

    return story, raw_lines

def generate_and_save_features(doc_id):
    from reportlab.pdfgen import canvas
    from PyPDF2 import PdfReader

    pdf_path = os.path.join(output_dir, f"doc_{doc_id}.pdf")
    json_entries = []

    story, raw_lines = simulate_pdf_lines(doc_id)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(story)

    # Re-parse rendered PDF to extract bbox and page info
    import fitz  # PyMuPDF
    pdf = fitz.open(pdf_path)
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
                    "width": span["bbox"][2] - span["bbox"][0],
                    "page": page_num,
                    "page_width": page.rect.width,
                    "page_height": page.rect.height
                })

        font_sizes = [l["font_size"] for l in lines]
        median = sum(font_sizes) / len(font_sizes) if font_sizes else 12
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

    # Append to synthetic_data.jsonl
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        for entry in json_entries:
            json.dump(entry, f)
            f.write("\n")

def match_label(text, raw_lines):
    for raw_text, label, _ in raw_lines:
        if text.strip() in raw_text.strip():
            return label
    return "BODY"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="Number of documents")
    args = parser.parse_args()

    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)

    for i in range(args.num):
        generate_and_save_features(i)

    print(f"{args.num} synthetic PDFs created and saved to {OUTPUT_JSONL}")
