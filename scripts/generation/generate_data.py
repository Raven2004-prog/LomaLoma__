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
    base = base.lower()
    if "times" in base:
        if bold and italic: return "Times-BoldItalic"
        if bold: return "Times-Bold"
        if italic: return "Times-Italic"
        return "Times-Roman"
    if "courier" in base:
        if bold and italic: return "Courier-BoldOblique"
        if bold: return "Courier-Bold"
        if italic: return "Courier-Oblique"
        return "Courier"
    if bold and italic: return "Helvetica-BoldOblique"
    if bold: return "Helvetica-Bold"
    if italic: return "Helvetica-Oblique"
    return "Helvetica"


def make_style(name, parent, family, size, alignment, bold=False, italic=False, space_after=12, left_indent=0):
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

# --- Feature Functions ---
def add_lexical_features(text):
    clean_text = text.strip()
    words = clean_text.split()
    return {
        "text": text,
        "word_count": len(words),
        "char_count": len(clean_text),
        "is_all_caps": clean_text.isupper() and len(clean_text) > 1,
        "is_title_case": clean_text.istitle() and len(clean_text) > 1,
        "starts_with_numbering": bool(re.match(r'^\d+\.', clean_text)),
        "ends_with_punctuation": clean_text.endswith(('.', ':', ';')) if clean_text else False,
    }

def add_line_spacing_feature(line_records):
    if not line_records:
        return []
    for i in range(len(line_records)-1):
        curr = line_records[i]
        nxt = line_records[i+1]
        gap = nxt['y_position'] - (curr['y_position'] + curr['line_height'])
        curr['line_spacing_ratio'] = round(gap/curr['line_height'], 2) if curr['line_height']>0 and gap>0 else 0.0
    line_records[-1]['line_spacing_ratio'] = 0.0
    return line_records


def extract_line_by_line_features(pdf_path, paragraph_records):
    doc = fitz.open(pdf_path)
    page_height = doc[0].rect.height if doc else 842
    records = []
    for page_num, page in enumerate(doc):
        data = page.get_text("dict", flags=1)
        for block in sorted(data['blocks'], key=lambda b: b['bbox'][1]):
            if block['type'] != 0: continue
            for line in sorted(block['lines'], key=lambda l: l['bbox'][1]):
                text = ''.join(span['text'] for span in line['spans']).strip()
                if not text: continue
                for p in paragraph_records:
                    if text in ' '.join(p['text'].split()):
                        if p['label'] not in ['H1','H2','H3','BODY','FIELD','SIGNATURE']:
                            continue
                        x0,y0,x1,y1 = line['bbox']
                        rec = {
                            'label': p['label'],
                            'font_size': p['font_size'],
                            'relative_font_size': p['relative_font_size'],
                            'page': page_num,
                            'x_position': round(x0,2),
                            'y_position': round(y0,2),
                            'line_height': round(y1-y0,2),
                            'page_position_ratio': round(y0/page_height,3),
                            'is_centered': abs(((x1-x0)/2 + x0) - (page.rect.width/2))<20
                        }
                        rec.update(add_lexical_features(text))
                        records.append(rec)
                        break
    return records

# --- Report Generator ---
def generate_report(doc_id):
    styles = getSampleStyleSheet()
    family = random.choice(FONTS)
    h1 = make_style("H1", styles['Heading1'], family, 18, TA_LEFT, bold=True)
    h2 = make_style("H2", styles['Heading2'], family, 14, TA_LEFT, bold=True, left_indent=inch*0.25)
    h3 = make_style("H3", styles['Heading3'], family, 12, TA_LEFT, bold=True, italic=True, left_indent=inch*0.5)
    body = make_style("Body", styles['BodyText'], family, BODY_FONT_SIZE, TA_LEFT)

    story = []
    paragraph_records = []
    def add_rec(text, label, st):
        paragraph_records.append({
            'text': text, 'label': label,
            'font_size': st.fontSize,
            'relative_font_size': round(st.fontSize/BODY_FONT_SIZE,2)
        })
        story.append(Paragraph(text, st))

    pages = random.randint(MIN_PAGES, MAX_PAGES)
    for i in range(pages):
        # H1
        h1_txt = f"Section {i+1}: {fake.catch_phrase()}"
        add_rec(h1_txt, 'H1', h1)
        # body
        for _ in range(random.randint(2,3)):
            txt = fake.paragraph(nb_sentences=random.randint(4,8))
            add_rec(txt, 'BODY', body)
        # H2
        if random.random()>0.4:
            h2_txt = fake.catch_phrase()
            add_rec(h2_txt, 'H2', h2)
            if random.random()>0.5:
                for j in range(random.randint(1,2)):
                    h3_txt = f"Subsection {i+1}.{j+1}: {fake.bs()}"
                    add_rec(h3_txt, 'H3', h3)
                    p_txt = fake.paragraph(nb_sentences=2)
                    add_rec(p_txt, 'BODY', body)
        if i<pages-1:
            story.append(PageBreak())

    pdf = os.path.join(OUT_DIR, f"doc_{doc_id}_0_label.pdf")
    jsonf = pdf.replace('.pdf','.labels.json')
    doc = SimpleDocTemplate(pdf, pagesize=A4)
    doc.build(story)
    lines = extract_line_by_line_features(pdf, paragraph_records)
    final = add_line_spacing_feature(lines)
    with open(jsonf,'w',encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"-> Report written {pdf}, {jsonf}")
    return pdf, jsonf

# --- Application Form Generator ---
def generate_application_form(doc_id):
    base_name = f"doc_{doc_id}_1_label"
    pdf = os.path.join(OUT_DIR, base_name + ".pdf")
    jsonf = os.path.join(OUT_DIR, base_name + ".labels.json")
    fields = ["Full Name","Date of Birth","Email","Phone","Address","Position","Date"]
    styles = getSampleStyleSheet()
    label_st = make_style("Label", styles['Normal'], random.choice(FONTS), 12, TA_LEFT, bold=True)
    paragraph_records = []
    for fld in fields:
        paragraph_records.append({'text': fld+':','label':'FIELD', 'font_size':label_st.fontSize,'relative_font_size':round(label_st.fontSize/BODY_FONT_SIZE,2)})
    paragraph_records.append({'text':'Signature','label':'SIGNATURE','font_size':label_st.fontSize,'relative_font_size':round(label_st.fontSize/BODY_FONT_SIZE,2)})

    doc = SimpleDocTemplate(pdf, pagesize=letter,rightMargin=72,leftMargin=72,topMargin=72,bottomMargin=72)
    elems=[Paragraph("Application Form", make_style("Title",styles['Title'],random.choice(FONTS),16,TA_CENTER,bold=True)),Spacer(1,0.3*inch)]
    for fld in fields:
        elems.append(Paragraph(fld+":", label_st))
        tbl=Table([['']],colWidths=[4.5*inch],rowHeights=[0.3*inch])
        tbl.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.black)]))
        elems.append(tbl)
    elems.append(Spacer(1,0.5*inch))
    sig=Table([['Signature']],colWidths=[2*inch],rowHeights=[0.3*inch])
    sig.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.black),('ALIGN',(0,0),(-1,-1),'LEFT')]))
    elems.append(sig)
    doc.build(elems)
    print(f"-> Form written {pdf}")

    lines = extract_line_by_line_features(pdf, paragraph_records)
    final = add_line_spacing_feature(lines)
    with open(jsonf,'w',encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"-> JSON written {jsonf}")
    return pdf, jsonf

# --- Main ---
def generate_document(doc_id, doc_type=None):
    doc_type = doc_type or random.choice(DOC_TYPES)
    if doc_type=="Report": return generate_report(doc_id)
    return generate_application_form(doc_id)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--num",type=int,default=50)
    parser.add_argument("--type",type=str,choices=DOC_TYPES,default=None)
    args=parser.parse_args()
    for i in range(args.num):
        generate_document(i,args.type)
    print(f"✅ Generated {args.num} docs in '{OUT_DIR}'")
