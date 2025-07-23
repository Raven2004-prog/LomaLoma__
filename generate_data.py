import argparse
import os
import random
import json
import re
import fitz                # PyMuPDF
from faker import Faker
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─── Setup ───────────────────────────────────────────────────────────────────────
fake = Faker(); Faker.seed(42)
OUT_DIR    = "data/synthetic"
os.makedirs(OUT_DIR, exist_ok=True)

DOC_TYPES  = ["Report","Analysis","Study","Review","Proposal","Manual","Briefing"]
DOC_SCOPES = ["Annual","Quarterly","Monthly","Internal","Confidential","Draft","External"]
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
def extract_features(item):
    txt = item["text"]
    return {
        "text": txt,
        "font_size": item["font_size"],
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
        "avg_word_len": sum(len(w) for w in txt.split()) / max(len(txt.split()),1),
        "named_entity_ratio": 0.0
    }

def extract_layout_features(pdf_path, records):
    doc = fitz.open(pdf_path)
    unmatched = records.copy()
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
    h2_st    = make_style("H2",    base["h2"], family, 14, TA_LEFT, bold=True, space_after=10)
    h3_st    = make_style("H3",    base["h3"], family, 12, TA_LEFT, bold=True, space_after=8)
    p_st     = make_style("Body",  base["BodyText"], family, 11, TA_LEFT, leading_factor=1.4, space_after=8)

    story, records = [], []

    # Title metadata (not kept) -- skip label collection
    story.append(PageBreak())

    # Main pages
    pages = random.randint(MIN_PAGES, MAX_PAGES)
    for i in range(pages):
        # H1
        h1_txt = f"Section {i+1}: {fake.catch_phrase()}"
        story.append(Paragraph(h1_txt, h1_st))
        records.append({"text":h1_txt, "label":"H1", "font_size":h1_st.fontSize})

        # BODY paragraphs
        for _ in range(random.randint(2,4)):
            p_txt = fake.paragraph(nb_sentences=random.randint(5,10))
            story.append(Paragraph(p_txt, p_st))
            records.append({"text":p_txt, "label":"BODY", "font_size":p_st.fontSize})

        # optional H2
        if random.random()>0.5:
            h2_txt = fake.bs().title()
            story.append(Paragraph(h2_txt, h2_st))
            records.append({"text":h2_txt, "label":"H2", "font_size":h2_st.fontSize})

            # optional H3 under H2
            if random.random()>0.5:
                h3_txt = fake.catch_phrase()
                story.append(Paragraph(h3_txt, h3_st))
                records.append({"text":h3_txt, "label":"H3", "font_size":h3_st.fontSize})

        if i < pages-1:
            story.append(PageBreak())

    # Build PDF
    pdf_path = os.path.join(OUT_DIR, f"doc_{doc_id}.pdf")
    SimpleDocTemplate(pdf_path, pagesize=A4).build(story)

    # Extract layout
    recs = extract_layout_features(pdf_path, records)

    # Write per-file JSON
    json_path = pdf_path.replace('.pdf', '.labels.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([extract_features(r) for r in recs], f, ensure_ascii=False, indent=2)
    print(f"Wrote {json_path}")

    return True

# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=60, help="Number of docs to generate")
    args = p.parse_args()

    for i in range(args.num):
        generate_document(i)
    print(f"Generated {args.num} PDFs with individual JSON labels.")
