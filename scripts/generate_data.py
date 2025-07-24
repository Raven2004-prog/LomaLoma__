import argparse
import os
import random
import json
import re
import fitz  # PyMuPDF
from faker import Faker
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# --- Basic Setup ---
fake = Faker()
Faker.seed(42)

# --- Configuration ---
OUT_DIR = "data/processed/"
os.makedirs(OUT_DIR, exist_ok=True)

DOC_TYPES = ["Report", "Analysis", "Study", "Review", "Proposal", "Manual", "Briefing"]
FONTS = ["Helvetica", "Courier", "Times-Roman"]
MIN_PAGES, MAX_PAGES = 8, 20
BODY_FONT_SIZE = 11

# --- Style & Font Helpers ---
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

# --- Post-Processing to Add Contextual Features ---
def add_contextual_features(records):
    """Adds layout and sequential features after PDF generation."""
    if not records:
        return []

    # Add Line Spacing Ratio
    for i in range(len(records) - 1):
        current_rec = records[i]
        next_rec = records[i+1]
        if "y_position" in current_rec and "y_position" in next_rec and "line_height" in current_rec:
            vertical_gap = next_rec["y_position"] - (current_rec["y_position"] + current_rec["line_height"])
            if current_rec["line_height"] > 0 and vertical_gap > 0:
                records[i]["line_spacing_ratio"] = round(vertical_gap / current_rec["line_height"], 2)
            else:
                records[i]["line_spacing_ratio"] = 0.0
        else:
             records[i]["line_spacing_ratio"] = 0.0
    if records: records[-1]["line_spacing_ratio"] = 0.0

    # Add BOS/EOS flags
    if records:
        records[0]['BOS'] = True
        records[-1]['EOS'] = True
    return records

# --- Feature Extraction from Generated PDF ---
def extract_and_merge_layout_features(pdf_path):
    doc = fitz.open(pdf_path)
    page_height = doc[0].rect.height if len(doc) > 0 else 842
    
    # Extract lines from PDF
    extracted_lines = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_width = page.rect.width
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(span["text"] for span in line["spans"])
                if not line_text.strip():
                    continue
                
                # Get first span for font info
                first_span = line["spans"][0] if line["spans"] else {}
                bbox = line["bbox"]
                # Calculate centered status using page width
                line_center_x = (bbox[0] + bbox[2]) / 2
                page_center_x = page_width / 2
                is_centered = abs(line_center_x - page_center_x) < 20
                
                extracted_lines.append({
                    "text": line_text,
                    "normalized_text": re.sub(r'\s+', ' ', line_text).strip(),
                    "bbox": bbox,
                    "page": page_num,
                    "font_size": first_span.get("size", 11),
                    "font_name": first_span.get("font", "Helvetica"),
                    "is_centered": is_centered
                })

    # Create line-wise records
    line_records = []
    for line_data in extracted_lines:
        # Determine label based on style properties
        font_size = line_data["font_size"]
        is_bold = "bold" in line_data["font_name"].lower()
        is_italic = "italic" in line_data["font_name"].lower()
        is_centered = line_data["is_centered"]
        
        # Label determination logic
        if line_data["page"] == 0 and font_size > 20 and is_centered:
            label = "TITLE"
        elif font_size > 16 and is_bold and not is_centered:
            label = "H1"
        elif 14 <= font_size <= 16 and is_bold and not is_centered:
            label = "H2"
        elif 12 <= font_size < 14 and is_bold and is_italic and not is_centered:
            label = "H3"
        else:
            label = "BODY"
        
        # Add layout features
        bbox = line_data["bbox"]
        rec = {
            "text": line_data["text"],
            "label": label,
            "font_size": line_data["font_size"],
            "relative_font_size": round(line_data["font_size"] / BODY_FONT_SIZE, 2),
            "is_bold": is_bold,
            "is_italic": is_italic,
            "page": line_data["page"],
            "x_position": bbox[0],
            "y_position": bbox[1],
            "line_height": bbox[3] - bbox[1],
            "page_position_ratio": bbox[1] / page_height,
            "is_centered": is_centered
        }
        
        # Add lexical features
        clean_text = rec["text"].strip()
        words = clean_text.split()
        rec.update({
            "word_count": len(words),
            "char_count": len(clean_text),
            "is_all_caps": clean_text.isupper() and len(clean_text) > 1,
            "is_title_case": clean_text.istitle() and len(clean_text) > 1,
            "starts_with_numbering": bool(re.match(r'^\d+\.', clean_text)),
            "ends_with_punctuation": clean_text.endswith(('.', ':', ';')) if clean_text else False,
        })
        
        line_records.append(rec)
    
    doc.close()
    return line_records

# --- Main Document Generation Logic ---
def generate_document(doc_id):
    print(f"Generating document {doc_id}...")
    base_styles = getSampleStyleSheet()
    family = random.choice(FONTS)

    title_st = make_style("Title", base_styles["h1"], family, 24, TA_CENTER, bold=True, space_after=20)
    h1_st = make_style("H1", base_styles["h1"], family, 18, TA_LEFT, bold=True, space_after=14)
    h2_st = make_style("H2", base_styles["h2"], family, 14, TA_LEFT, bold=True, left_indent=inch*0.25)
    h3_st = make_style("H3", base_styles["h3"], family, 12, TA_LEFT, bold=True, italic=True, left_indent=inch*0.5)
    body_st = make_style("Body", base_styles["BodyText"], family, BODY_FONT_SIZE, TA_LEFT)
    list_st = make_style("List", base_styles["BodyText"], family, BODY_FONT_SIZE, TA_LEFT, left_indent=inch*0.25)
    
    story = []

    # Title
    title_txt = f"{random.choice(DOC_TYPES)}: {fake.bs().title()}"
    story.append(Paragraph(title_txt, title_st))
    story.append(PageBreak())

    # Main Content
    num_pages = random.randint(MIN_PAGES, MAX_PAGES)
    for i in range(num_pages):
        h1_txt = f"Section {i + 1}: {fake.catch_phrase()}"
        story.append(Paragraph(h1_txt, h1_st))

        for _ in range(random.randint(2, 3)):
            p_txt = fake.paragraph(nb_sentences=random.randint(4, 8))
            story.append(Paragraph(p_txt, body_st))
        
        if random.random() > 0.4:
            h2_txt = fake.catch_phrase()
            story.append(Paragraph(h2_txt, h2_st))

            if random.random() > 0.5:
                for j in range(random.randint(1, 2)):
                    h3_txt = f"Subsection {i+1}.{j+1} - {fake.bs()}"
                    story.append(Paragraph(h3_txt, h3_st))
                    p_txt = fake.paragraph(nb_sentences=2)
                    story.append(Paragraph(p_txt, body_st))
            else:
                list_paragraphs = [Paragraph(fake.sentence(nb_words=8), list_st) for _ in range(random.randint(3, 5))]
                list_items = [ListItem(p) for p in list_paragraphs]
                story.append(ListFlowable(list_items, bulletType='bullet', start='bulletchar'))

        if i < num_pages - 1:
            story.append(PageBreak())

    # --- PDF and JSON Creation ---
    pdf_path = os.path.join(OUT_DIR, f"doc_{doc_id}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(story)
    print(f"  -> Saved PDF: {pdf_path}")

    # Extract line-wise features
    line_records = extract_and_merge_layout_features(pdf_path)
    
    # Add contextual features
    final_records = add_contextual_features(line_records)

    json_path = pdf_path.replace('.pdf', '.labels.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)
    
    print(f"  -> Saved labels: {json_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PDF documents for model training.")
    parser.add_argument("--num", type=int, default=50, help="Number of documents to generate.")
    parser.add_argument("--out", type=str, default=OUT_DIR, help="Output directory for PDFs and JSON files.")
    args = parser.parse_args()

    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)

    for i in range(args.num):
        generate_document(i)

    print(f"✅ Generated {args.num} PDF documents with labeled JSON in '{OUT_DIR}'")