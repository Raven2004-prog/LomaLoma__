import argparse
import json
import os
import glob
import random
import re

import fitz                # PyMuPDF
from faker import Faker
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from statistics import mean, stdev

# ─── Setup ───────────────────────────────────────────────────────────────────────
fake = Faker()
Faker.seed(42)
OUT_DIR = "data/synthetic"
os.makedirs(OUT_DIR, exist_ok=True)

FONTS      = ["Helvetica","Courier","Times-Roman"]
MIN_PAGES, MAX_PAGES = 10, 15

# ─── Style Helpers ───────────────────────────────────────────────────────────────
def make_font_name(base, bold=False, italic=False):
    low = base.lower()
    if "times" in low:
        if bold and italic: return "Times-BoldItalic"
        if bold:           return "Times-Bold"
        if italic:         return "Times-Italic"
        return "Times-Roman"
    if "courier" in low:
        if bold and italic: return "Courier-BoldOblique"
        if bold:           return "Courier-Bold"
        if italic:         return "Courier-Oblique"
        return "Courier"
    if bold and italic: return "Helvetica-BoldOblique"
    if bold:           return "Helvetica-Bold"
    if italic:         return "Helvetica-Oblique"
    return "Helvetica"

def make_style(name, parent, family, size, alignment,
               bold=False, italic=False,
               space_after=10, leading_factor=1.2, left_indent=0):
    return ParagraphStyle(
        name=name,
        parent=parent,
        fontName=make_font_name(family, bold, italic),
        fontSize=size,
        leading=size * leading_factor,
        spaceAfter=space_after,
        leftIndent=left_indent,
        alignment=alignment
    )

# ─── Feature Extraction ─────────────────────────────────────────────────────────
FEATURE_KEYS = [
    "text", "font_size", "line_width", "line_height", "char_count",
    "page", "y_position", "label", "is_all_caps", "is_title_case",
    "starts_with_number", "contains_colon", "contains_year",
    "word_count", "avg_word_len", "named_entity_ratio", "pdf_path"
]

def extract_features(item, mean_fs, std_fs, pdf_path):
    txt = item["text"]
    raw_size = item["font_size"]
    return {
        "text": txt,
        "font_size": raw_size,
        "line_width": item.get("line_width"),
        "line_height": item.get("line_height"),
        "char_count": len(txt),
        "page": item.get("page"),
        "y_position": item.get("y_position"),
        "label": item["label"],
        "is_all_caps": txt.isupper(),
        "is_title_case": txt.istitle(),
        "starts_with_number": bool(re.match(r'^[0-9]', txt.strip())),
        "contains_colon": ':' in txt,
        "contains_year": bool(re.search(r'\b19\d{2}\b|\b20\d{2}\b', txt)),
        "word_count": len(txt.split()),
        "avg_word_len": sum(len(w) for w in txt.split()) / max(len(txt.split()), 1),
        "named_entity_ratio": 0.0,
        "pdf_path": pdf_path
    }

def extract_layout_features(pdf_path, records):
    doc = fitz.open(pdf_path)
    unmatched = list(records)
    for pnum in range(len(doc)):
        page = doc.load_page(pnum)
        blocks = page.get_text("blocks")
        for x0, y0, x1, y1, txt, _, btype in blocks:
            if btype != 0 or not txt.strip():
                continue
            snippet = " ".join(txt.split())
            for item in unmatched:
                cand = " ".join(item["text"].split())
                if snippet in cand or cand in snippet:
                    item.update({
                        "page": pnum + 1,
                        "line_width": x1 - x0,
                        "line_height": y1 - y0,
                        "y_position": y0
                    })
                    unmatched.remove(item)
                    break
    if unmatched:
        print(f"Warning: {len(unmatched)} items missing layout in {pdf_path}")
    return records

# ─── Generate One Document ─────────────────────────────────────────────────────
def generate_document(doc_id):
    base = getSampleStyleSheet()
    family = random.choice(FONTS)
    title_st = make_style("Title", base["h1"], family, random.randint(22,28), TA_CENTER, bold=True, space_after=18)
    h1_st    = make_style("H1",    base["h1"], family, 18, TA_LEFT, bold=True, space_after=12)
    p_st     = make_style("Body",  base["BodyText"], family, 11, TA_LEFT, leading_factor=1.4, space_after=8)

    story, records = [], []
    story.append(PageBreak())

    pages = random.randint(MIN_PAGES, MAX_PAGES)
    for i in range(pages):
        # H1
        h1_txt = f"Section {i+1}: {fake.catch_phrase()}"
        story.append(Paragraph(h1_txt, h1_st))
        records.append({"text": h1_txt, "label": "H1", "font_size": h1_st.fontSize})

        # Body paragraphs
        for _ in range(random.randint(2,4)):
            p_txt = fake.paragraph(nb_sentences=random.randint(5,10))
            story.append(Paragraph(p_txt, p_st))
            records.append({"text": p_txt, "label": "BODY", "font_size": p_st.fontSize})

        if i < pages - 1:
            story.append(PageBreak())

    # Build PDF
    pdf_path = os.path.join(OUT_DIR, f"doc_{doc_id}.pdf")
    SimpleDocTemplate(pdf_path, pagesize=A4).build(story)

    # Extract, feature-engineer, and write JSON
    recs = extract_layout_features(pdf_path, records)
    sizes = [r["font_size"] for r in recs]
    m, s = mean(sizes), stdev(sizes) if len(sizes) > 1 else 0
    features = [extract_features(r, m, s, pdf_path) for r in recs]

    json_path = os.path.join(OUT_DIR, f"doc_{doc_id}.labels.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print(f"Wrote {json_path}")
    return True

# ─── Combine JSONs ───────────────────────────────────────────────────────────────
def combine_jsons(json_dir, output_path):
    all_records = []
    for filepath in glob.glob(os.path.join(json_dir, '*.labels.json')):
        with open(filepath, 'r', encoding='utf-8') as f:
            all_records.extend(json.load(f))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"Combined {len(all_records)} records into {output_path}")

# ─── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=60, help="Number of docs to generate")
    parser.add_argument("--combine", action="store_true",
                        help="After generation, combine all .labels.json into one file")
    parser.add_argument("--output-json", default=os.path.join(OUT_DIR, "synthetic_combined.json"),
                        help="Path for combined JSON")
    args = parser.parse_args()

    for i in range(args.num):
        generate_document(i)
    print(f"Generated {args.num} PDFs and JSONs")

    if args.combine:
        combine_jsons(OUT_DIR, args.output_json)
