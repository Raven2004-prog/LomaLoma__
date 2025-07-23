import os
import json
import random
import re
import lorem
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import inch

output_dir = "data/synthetic2"
os.makedirs(output_dir, exist_ok=True)

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
        leftIndent=random.choice([0, 10, 20, 30])
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
        "font_size_ratio": round(line["font_size"] / median_font_size, 2) if median_font_size else 1.0,
        "has_numeric_prefix": int(bool(re.match(r'^[0-9]+(\.[0-9]+)*', text))),
        "word_count": len(text.strip().split()),
        "center_deviation": round(abs((page_width / 2) - (x0 + width / 2)) / page_width, 2),
        "size_vs_prev": round(line["font_size"] / prev_line["font_size"], 2) if prev_line else 1.0,
        "page_top_distance": round(y0 / page_height, 2)
    }
    return features


def random_heading(max_len=30):
    """Generates a heading of length ≤ max_len"""
    while True:
        text = lorem.sentence().split(".")[0]
        clean_text = re.sub(r'\s+', ' ', text).strip()
        if len(clean_text) <= max_len:
            return clean_text


def simulate_pdf_lines(doc_id):
    font = random.choice(FONT_NAMES)
    italic = random.choice([False, True])
    base_size = 11
    align = random.choice(ALIGNMENTS)

    def heading_font(size_offset):
        return random.choice([base_size + size_offset + delta for delta in [-1, 0, 1]])

    heading_styles = {
        "TITLE": make_style(styles["Heading1"], heading_font(4), font, align, bold=True, italic=italic),
        "H1": make_style(styles["Heading2"], heading_font(2), font, align, bold=random.choice([True, False]), italic=italic),
        "H2": make_style(styles["Heading3"], heading_font(1), font, align, bold=random.choice([True, False]), italic=italic),
        "H3": make_style(styles["Heading4"], heading_font(0), font, align, bold=random.choice([True, False]), italic=italic),
        "BODY": make_style(styles["BodyText"], base_size, font, TA_LEFT, bold=False, italic=False),
    }

    story = []
    raw_lines = []

    def add_line(text, label, style):
        para = Paragraph(text, style)
        story.append(para)
        story.append(Spacer(1, 0.1 * inch))
        raw_lines.append((text, label, style.fontSize))

    # Title
    add_line(random_heading(), "TITLE", heading_styles["TITLE"])

    section_count = 0
    for _ in range(4):  # Top-level H1s
        add_line(random_heading(), "H1", heading_styles["H1"])
        section_count += 1

        for _ in range(random.randint(1, 3)):
            add_line(random_heading(), "H2", heading_styles["H2"])
            section_count += 1

            for _ in range(random.randint(1, 2)):
                add_line(random_heading(), "H3", heading_styles["H3"])
                section_count += 1

                for _ in range(2):
                    add_line(lorem.paragraph(), "BODY", heading_styles["BODY"])

        if section_count >= random.randint(3, 5):
            story.append(PageBreak())
            section_count = 0

    return story, raw_lines


def generate_and_save_features(doc_id):
    import fitz  # PyMuPDF

    pdf_path = os.path.join(output_dir, f"doc_{doc_id}.pdf")
    json_path = os.path.join(output_dir, f"doc_{doc_id}.json")
    json_entries = []

    story, raw_lines = simulate_pdf_lines(doc_id)
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(story)

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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_entries, f, indent=2)


def match_label(text, raw_lines):
    for raw_text, label, _ in raw_lines:
        if text.strip() in raw_text.strip():
            return label
    return "BODY"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="Number of documents to generate")
    args = parser.parse_args()

    for i in range(args.num):
        generate_and_save_features(i)

    print(f"{args.num} synthetic PDFs and matching JSON files saved to {output_dir}")
