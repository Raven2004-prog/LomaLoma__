import argparse
import os
import random
import json
import re
import fitz  # PyMuPDF for PDF parsing
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

# --- Configuration ---
OUT_DIR = "generated_docs"
os.makedirs(OUT_DIR, exist_ok=True)
DOC_TYPES = ["Report", "ApplicationForm", "Invoice", "Resume"]
FONTS = ["Helvetica", "Courier", "Times-Roman"]
MIN_PAGES, MAX_PAGES = 2, 5  # adjust as desired
BODY_FONT_SIZE = 11

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

# --- Style Helpers ---
def make_font_name(base, bold=False, italic=False):
    name = base.lower()
    if "times" in name:
        if bold and italic: return "Times-BoldItalic"
        if bold: return "Times-Bold"
        if italic: return "Times-Italic"
        return "Times-Roman"
    if "courier" in name:
        if bold and italic: return "Courier-BoldOblique"
        if bold: return "Courier-Bold"
        if italic: return "Courier-Oblique"
        return "Courier"
    if bold and italic: return "Helvetica-BoldOblique"
    if bold: return "Helvetica-Bold"
    if italic: return "Helvetica-Oblique"
    return "Helvetica"

def make_style(name, parent, family, size, alignment,
               bold=False, italic=False, space_after=6, left_indent=0):
    return ParagraphStyle(
        name=name,
        parent=parent,
        fontName=make_font_name(family, bold, italic),
        fontSize=size,
        leading=size * 1.2,
        spaceAfter=space_after,
        leftIndent=left_indent,
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
        nxt = lines[i + 1]
        gap = nxt["y_position"] - (curr["y_position"] + curr["line_height"])
        curr["line_spacing_ratio"] = (
            round(gap / curr["line_height"], 2)
            if curr["line_height"] > 0 and gap > 0 else 0.0
        )
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
                    if text in p['text']:
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

# --- Generators ---

def generate_report(doc_id):
    styles = getSampleStyleSheet()
    fam = random.choice(FONTS)
    st_h1 = make_style("H1", styles['Heading1'], fam, 18, TA_CENTER, bold=True)
    st_h2 = make_style("H2", styles['Heading2'], fam, 14, TA_LEFT, bold=True, left_indent=inch*0.25)
    st_body = make_style("Body", styles['BodyText'], fam, BODY_FONT_SIZE, TA_LEFT)

    story, paras = [], []
    def record(text, label, style):
        paras.append({
            'text': text, 'label': label,
            'font_size': style.fontSize,
            'relative_font_size': round(style.fontSize/BODY_FONT_SIZE, 2)
        })
        story.append(Paragraph(text, style))

    pages = random.randint(MIN_PAGES, MAX_PAGES)
    for i in range(pages):
        for _ in range(random.randint(3, 6)):
            if random.random() < 0.3:
                lvl = random.choice(['H1', 'H2'])
                text = fake.catch_phrase() if lvl=='H1' else fake.bs().capitalize()
                style = st_h1 if lvl=='H1' else st_h2
                record(text, lvl, style)
            record(fake.paragraph(), 'BODY', st_body)
        if i < pages-1:
            story.append(PageBreak())

    base = f"report_{doc_id}_label_{doc_id}"
    pdf_path = os.path.join(OUT_DIR, base + "_kus.pdf")
    json_path = os.path.join(OUT_DIR, base + "_kus.json")
    SimpleDocTemplate(pdf_path, pagesize=A4).build(story)

    lines = extract_line_by_line_features(pdf_path, paras)
    final = add_line_spacing_feature(lines)
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(final, jf, ensure_ascii=False, indent=2)
    print(f"-> Generated Report: {pdf_path}")

def generate_application_form(doc_id):
    styles = getSampleStyleSheet()
    fam = random.choice(FONTS)
    st_title = make_style("Title", styles['Title'], fam, 16, TA_CENTER, bold=True)
    st_body = make_style("Body", styles['BodyText'], fam, BODY_FONT_SIZE, TA_LEFT)

    fields = ["Full Name", "Email", "Phone Number", "Address", "DOB", "Nationality"]
    story, paras = [], []
    paras.append({'text': "Application Form", 'label': 'H1',
                  'font_size': st_title.fontSize, 'relative_font_size': round(st_title.fontSize/BODY_FONT_SIZE,2)})
    story.append(Paragraph("Application Form", st_title))
    story.append(Spacer(1, 0.2*inch))

    for fld in fields:
        txt = fld + ":"
        paras.append({'text': txt, 'label': 'FIELD',
                      'font_size': st_body.fontSize, 'relative_font_size': round(st_body.fontSize/BODY_FONT_SIZE,2)})
        story.append(Paragraph(txt, st_body))
        tbl = Table([['']], colWidths=[4*inch], rowHeights=[0.3*inch])
        tbl.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.black)]))
        story.append(tbl)

    story.append(Spacer(1, 0.3*inch))
    paras.append({'text': "Signature", 'label': 'FIELD',
                  'font_size': st_body.fontSize, 'relative_font_size': round(st_body.fontSize/BODY_FONT_SIZE,2)})
    sig_tbl = Table([['']], colWidths=[2*inch], rowHeights=[0.3*inch])
    sig_tbl.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.black)]))
    story.append(sig_tbl)

    base = f"application_{doc_id}_label_{doc_id}"
    pdf_path = os.path.join(OUT_DIR, base + "_kus.pdf")
    json_path = os.path.join(OUT_DIR, base + "_kus.json")
    SimpleDocTemplate(pdf_path, pagesize=letter,
                     rightMargin=72,leftMargin=72,
                     topMargin=72,bottomMargin=72).build(story)

    lines = extract_line_by_line_features(pdf_path, paras)
    final = add_line_spacing_feature(lines)
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(final, jf, ensure_ascii=False, indent=2)
    print(f"-> Generated ApplicationForm: {pdf_path}")

def generate_invoice(doc_id):
    styles = getSampleStyleSheet()
    fam = random.choice(FONTS)
    st_h1 = make_style("H1", styles['Heading1'], fam, 18, TA_CENTER, bold=True)
    st_body = make_style("Body", styles['BodyText'], fam, BODY_FONT_SIZE, TA_LEFT)

    story, paras = [], []
    def record(txt, lbl, sty):
        paras.append({'text':txt,'label':lbl,'font_size':sty.fontSize,'relative_font_size':round(sty.fontSize/BODY_FONT_SIZE,2)})
        story.append(Paragraph(txt, sty))

    record("Invoice", 'H1', st_h1)
    for i in range(1,6):
        item = f"Item {i}: {fake.catch_phrase()} - ${random.randint(10,500)}"
        record(item, 'BODY', st_body)
    record(f"Total: ${random.randint(500,2000)}", 'BODY', st_body)

    base = f"invoice_{doc_id}_label_{doc_id}"
    pdf_path = os.path.join(OUT_DIR, base + "_kus.pdf")
    json_path = os.path.join(OUT_DIR, base + "_kus.json")
    SimpleDocTemplate(pdf_path, pagesize=A4).build(story)

    lines = extract_line_by_line_features(pdf_path, paras)
    final = add_line_spacing_feature(lines)
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(final, jf, ensure_ascii=False, indent=2)
    print(f"-> Generated Invoice: {pdf_path}")

def generate_resume(doc_id):
    styles = getSampleStyleSheet()
    fam = random.choice(FONTS)
    st_h1 = make_style("H1", styles['Heading1'], fam, 18, TA_CENTER, bold=True)
    st_h2 = make_style("H2", styles['Heading2'], fam, 14, TA_LEFT, bold=True)
    st_body = make_style("Body", styles['BodyText'], fam, BODY_FONT_SIZE, TA_LEFT)

    story, paras = [], []
    name = fake.name()
    paras.append({'text':name,'label':'H1','font_size':st_h1.fontSize,'relative_font_size':round(st_h1.fontSize/BODY_FONT_SIZE,2)})
    story.append(Paragraph(name, st_h1))

    record = lambda txt, lbl, sty: (
        paras.append({'text':txt,'label':lbl,'font_size':sty.fontSize,'relative_font_size':round(sty.fontSize/BODY_FONT_SIZE,2)}),
        story.append(Paragraph(txt, sty))
    )
    record("Professional Summary", 'H2', st_h2)
    record(fake.paragraph(nb_sentences=3), 'BODY', st_body)
    record("Skills", 'H2', st_h2)
    record(", ".join(fake.words(nb=6)), 'BODY', st_body)

    base = f"resume_{doc_id}_label_{doc_id}"
    pdf_path = os.path.join(OUT_DIR, base + "_kus.pdf")
    json_path = os.path.join(OUT_DIR, base + "_kus.json")
    SimpleDocTemplate(pdf_path, pagesize=A4).build(story)

    lines = extract_line_by_line_features(pdf_path, paras)
    final = add_line_spacing_feature(lines)
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(final, jf, ensure_ascii=False, indent=2)
    print(f"-> Generated Resume: {pdf_path}")

# --- Main Entry ---
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PDF documents with feature extraction.")
    parser.add_argument("--num", type=int, default=60, help="Number of documents to generate.")
    parser.add_argument("--type", type=str, choices=DOC_TYPES, default=None, help="Document type to generate.")
    args = parser.parse_args()

    for i in range(args.num):
        dtype = args.type if args.type else random.choice(DOC_TYPES)
        if dtype == "Report":
            generate_report(i)
        elif dtype == "ApplicationForm":
            generate_application_form(i)
        elif dtype == "Invoice":
            generate_invoice(i)
        elif dtype == "Resume":
            generate_resume(i)

    print(f"✅ Generated {args.num} documents in '{OUT_DIR}'")

if __name__ == "__main__":
    main()
