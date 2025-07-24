import argparse
import os
import random
import json
import re
import fitz  # PyMuPDF
from faker import Faker
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    PageBreak, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors

# Initialize Faker
fake = Faker()
Faker.seed(42)

# --- Configuration ---
OUT_DIR = "data/processed/"
os.makedirs(OUT_DIR, exist_ok=True)
DOC_TYPES = ["Report", "ApplicationForm"]
FONTS = ["Helvetica", "Courier", "Times-Roman"]
MIN_PAGES, MAX_PAGES = 8, 20
BODY_FONT_SIZE = 11

# --- Style Helpers ---
def make_font_name(base, bold=False, italic=False):
    b = base.lower()
    if "times" in b:
        if bold and italic: return "Times-BoldItalic"
        if bold: return "Times-Bold"
        if italic: return "Times-Italic"
        return "Times-Roman"
    if "courier" in b:
        if bold and italic: return "Courier-BoldOblique"
        if bold: return "Courier-Bold"
        if italic: return "Courier-Oblique"
        return "Courier"
    if bold and italic: return "Helvetica-BoldOblique"
    if bold: return "Helvetica-Bold"
    if italic: return "Helvetica-Oblique"
    return "Helvetica"


def make_style(name, parent, family, size, alignment,
               bold=False, italic=False, space_after=12, left_indent=0):
    return ParagraphStyle(
        name=name, parent=parent,
        fontName=make_font_name(family, bold, italic),
        fontSize=size, leading=size * 1.2,
        spaceAfter=space_after, leftIndent=left_indent,
        alignment=alignment
    )

# --- Feature Computations ---
def add_lexical_features(text):
    clean = text.strip()
    words = clean.split()
    return {
        "text": text,
        "word_count": len(words),
        "char_count": len(clean),
        "is_all_caps": clean.isupper() and len(clean) > 1,
        "is_title_case": clean.istitle() and len(clean) > 1,
        "starts_with_numbering": bool(re.match(r'^\d+\.', clean)),
        "ends_with_punctuation": clean.endswith(('.', ':', ';')) if clean else False,
    }


def add_line_spacing_feature(lines):
    if not lines:
        return []
    for i in range(len(lines) - 1):
        curr = lines[i]
        nxt = lines[i+1]
        gap = nxt["y_position"] - (curr["y_position"] + curr["line_height"])
        curr["line_spacing_ratio"] = round(gap / curr["line_height"], 2) \
            if curr["line_height"] > 0 and gap > 0 else 0.0
    lines[-1]["line_spacing_ratio"] = 0.0
    return lines


def extract_line_by_line_features(pdf_path, paragraph_records):
    doc = fitz.open(pdf_path)
    height = doc[0].rect.height if doc else 842
    results = []
    for page_idx, page in enumerate(doc):
        data = page.get_text("dict", flags=1)
        for block in sorted(data['blocks'], key=lambda b: b['bbox'][1]):
            if block['type'] != 0: continue
            for line in sorted(block['lines'], key=lambda l: l['bbox'][1]):
                text = ''.join(span['text'] for span in line['spans']).strip()
                if not text: continue
                for p in paragraph_records:
                    if text in ' '.join(p['text'].split()):
                        x0, y0, x1, y1 = line['bbox']
                        rec = {
                            'label': p['label'],
                            'font_size': p['font_size'],
                            'relative_font_size': p['relative_font_size'],
                            'page': page_idx,
                            'x_position': round(x0, 2),
                            'y_position': round(y0, 2),
                            'line_height': round(y1 - y0, 2),
                            'page_position_ratio': round(y0 / height, 3),
                            'is_centered': abs(((x1 - x0)/2 + x0) - (page.rect.width/2)) < 20
                        }
                        rec.update(add_lexical_features(text))
                        results.append(rec)
                        break
    return results

# --- Enhanced Report Generator ---
def generate_report(doc_id):
    styles = getSampleStyleSheet()
    fam = random.choice(FONTS)
    st_h1 = make_style("H1", styles['Heading1'], fam, 18, random.choice([TA_LEFT, TA_CENTER]), bold=True)
    st_h2 = make_style("H2", styles['Heading2'], fam, 14, TA_LEFT, bold=True, left_indent=inch*0.25)
    st_h3 = make_style("H3", styles['Heading3'], fam, 12, TA_LEFT, bold=True, italic=True, left_indent=inch*0.5)
    st_body = make_style("Body", styles['BodyText'], fam, BODY_FONT_SIZE, TA_LEFT)

    story, paras = [], []

    def record(text, label, style):
        paras.append({
            'text': text,
            'label': label,
            'font_size': style.fontSize,
            'relative_font_size': round(style.fontSize / BODY_FONT_SIZE, 2)
        })
        story.append(Paragraph(text, style))

    pages = random.randint(MIN_PAGES, MAX_PAGES)
    for i in range(pages):
        blocks = []
        # Each page: random mix of headings and bodies
        for _ in range(random.randint(4, 8)):
            # Decide element type
            if random.random() < 0.3:
                # Insert a heading (random level)
                level = random.choices(['H1','H2','H3'], weights=[0.2,0.5,0.3])[0]
                text = (f"Section {i+1}: {fake.catch_phrase()}" if level=='H1'
                        else fake.bs().capitalize())
                style = {'H1':st_h1,'H2':st_h2,'H3':st_h3}[level]
                blocks.append((text, level, style))
            else:
                # Insert body
                blocks.append((fake.paragraph(nb_sentences=random.randint(3,6)), 'BODY', st_body))

        for text, label, style in blocks:
            record(text, label, style)
        if i < pages-1:
            story.append(PageBreak())

    base = f"report_{doc_id}_label"
    pdf_file = os.path.join(OUT_DIR, base + ".pdf")
    json_file = base + ".labels.json"
    SimpleDocTemplate(pdf_file, pagesize=A4).build(story)

    lines = extract_line_by_line_features(pdf_file, paras)
    final = add_line_spacing_feature(lines)
    with open(os.path.join(OUT_DIR, json_file), 'w', encoding='utf-8') as jf:
        json.dump(final, jf, ensure_ascii=False, indent=2)
    print(f"-> Report: {pdf_file}, {json_file}")

# --- ApplicationForm Generator (unchanged) ---
def generate_application_form(doc_id):
    styles = getSampleStyleSheet()
    fam = random.choice(FONTS)
    st_title = make_style("H1", styles['Title'], fam, 16, TA_CENTER, bold=True)
    st_body  = make_style("Body", styles['BodyText'], fam, BODY_FONT_SIZE, TA_LEFT)

    fields = [
        "Full Name", "Date of Birth", "Email Address",
        "Phone Number", "Address", "Position Applied For", "Date"
    ]

    story, paras = [], []
    paras.append({'text':"Application Form", 'label':'H1', 'font_size':st_title.fontSize, 'relative_font_size':round(st_title.fontSize/BODY_FONT_SIZE,2)})
    story.append(Paragraph("Application Form", st_title))
    story.append(Spacer(1, 0.3*inch))

    for fld in fields:
        text = fld + ":"
        paras.append({'text':text, 'label':'BODY', 'font_size':st_body.fontSize, 'relative_font_size':round(st_body.fontSize/BODY_FONT_SIZE,2)})
        story.append(Paragraph(text, st_body))
        tbl = Table([['']], colWidths=[4.5*inch], rowHeights=[0.3*inch])
        tbl.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.black)]))
        story.append(tbl)

    sig = "Signature"
    paras.append({'text':sig, 'label':'BODY', 'font_size':st_body.fontSize, 'relative_font_size':round(st_body.fontSize/BODY_FONT_SIZE,2)})
    story.append(Spacer(1, 0.5*inch))
    sig_tbl = Table([[sig]], colWidths=[2*inch], rowHeights=[0.3*inch])
    sig_tbl.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.black),
                                  ('ALIGN',(0,0),(-1,-1),'LEFT')]))
    story.append(sig_tbl)

    base = f"application_{doc_id}_label"
    pdf_file  = os.path.join(OUT_DIR, base + ".pdf")
    json_file = base + ".labels.json"
    SimpleDocTemplate(pdf_file, pagesize=letter,
                     rightMargin=72,leftMargin=72,
                     topMargin=72,bottomMargin=72).build(story)

    lines = extract_line_by_line_features(pdf_file, paras)
    final = add_line_spacing_feature(lines)
    with open(os.path.join(OUT_DIR, json_file), 'w', encoding='utf-8') as jf:
        json.dump(final, jf, ensure_ascii=False, indent=2)
    print(f"-> ApplicationForm: {pdf_file}, {json_file}")

# --- Main Entry ---
def generate_document(doc_id, doc_type=None):
    if doc_type == "ApplicationForm":
        generate_application_form(doc_id)
    else:
        generate_report(doc_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PDF documents with diverse headings.")
    parser.add_argument("--num", type=int, default=50, help="Number of documents to generate.")
    parser.add_argument("--type", type=str, choices=DOC_TYPES, default=None,
                        help="Document type: Report or ApplicationForm.")
    args = parser.parse_args()

    for i in range(args.num):
        generate_document(i, args.type)
    print(f"✅ Generated {args.num} documents in '{OUT_DIR}'")
