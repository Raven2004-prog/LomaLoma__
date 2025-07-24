import argparse
import os
import random
import json
import re
import fitz  # PyMuPDF
from faker import Faker
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT

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

def add_lexical_features(text):
    """Calculates lexical features for a given line of text."""
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


# --- Post-Processing to Add Contextual Features ---
def add_line_spacing_feature(line_records):
    """Calculates the spacing ratio between each line and the next."""
    if not line_records:
        return []

    for i in range(len(line_records) - 1):
        current_line = line_records[i]
        next_line = line_records[i+1]
        
        # Calculate vertical gap from the bottom of the current line to the top of the next
        vertical_gap = next_line["y_position"] - (current_line["y_position"] + current_line["line_height"])
        
        # Normalize by current line height. Avoid division by zero.
        if current_line["line_height"] > 0 and vertical_gap > 0:
            line_records[i]["line_spacing_ratio"] = round(vertical_gap / current_line["line_height"], 2)
        else:
            line_records[i]["line_spacing_ratio"] = 0.0
            
    # Last record has no line after it
    if line_records: line_records[-1]["line_spacing_ratio"] = 0.0
    return line_records


# --- Core Logic: True Line-by-Line Feature Extraction ---
def extract_line_by_line_features(pdf_path, paragraph_records):
    """
    Opens the generated PDF and creates a feature set for EACH visual line,
    mapping it back to its original paragraph's label and style.
    """
    doc = fitz.open(pdf_path)
    page_height = doc[0].rect.height if len(doc) > 0 else 842  # A4 height fallback
    final_line_records = []

    for page_num, page in enumerate(doc):
        # CORRECTED: Use integer flag 1 instead of named attribute
        page_dict = page.get_text("dict", flags=1) 
        
        for block in sorted(page_dict["blocks"], key=lambda b: b['bbox'][1]): # Sort blocks by y-pos
            if block['type'] != 0: continue # Skip image blocks

            for line in sorted(block["lines"], key=lambda l: l['bbox'][1]): # Sort lines by y-pos
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text: continue

                # Find the original paragraph record that this line belongs to
                matched_paragraph = None
                for p_rec in paragraph_records:
                    # Clean paragraph text for robust matching
                    p_text_cleaned = " ".join(p_rec["text"].strip().split())
                    if line_text in p_text_cleaned:
                        matched_paragraph = p_rec
                        break
                
                if matched_paragraph:
                    # Only include H1, H2, H3, and BODY
                    if matched_paragraph["label"] not in ["H1", "H2", "H3", "BODY"]:
                        continue
                        
                    x0, y0, x1, y1 = line["bbox"]
                    
                    # Create a new record for this specific line
                    line_record = {
                        # Inherit label and style from the parent paragraph
                        "label": matched_paragraph["label"],
                        "font_size": matched_paragraph["font_size"],
                        "relative_font_size": matched_paragraph["relative_font_size"],
                        
                        # Add line-specific layout features (0-indexed page)
                        "page": page_num,  # 0-indexed
                        "x_position": round(x0, 2),
                        "y_position": round(y0, 2),
                        "line_height": round(y1 - y0, 2),
                        "page_position_ratio": round(y0 / page_height, 3),
                        "is_centered": abs(((x1 - x0) / 2 + x0) - (page.rect.width / 2)) < 20
                    }
                    
                    # Add lexical features for the line's text
                    line_record.update(add_lexical_features(line_text))
                    
                    final_line_records.append(line_record)

    return final_line_records


# --- Main Document Generation Logic ---
def generate_document(doc_id):
    print(f"Generating document {doc_id}...")
    base_styles = getSampleStyleSheet()
    family = random.choice(FONTS)

    # Only include styles for H1, H2, H3, BODY
    h1_st = make_style("H1", base_styles["h1"], family, 18, TA_LEFT, bold=True, space_after=14)
    h2_st = make_style("H2", base_styles["h2"], family, 14, TA_LEFT, bold=True, left_indent=inch*0.25)
    h3_st = make_style("H3", base_styles["h3"], family, 12, TA_LEFT, bold=True, italic=True, left_indent=inch*0.5)
    body_st = make_style("Body", base_styles["BodyText"], family, BODY_FONT_SIZE, TA_LEFT)
    
    story = []
    paragraph_records = []

    # Helper to create records
    def add_para_record(text, label, style):
        paragraph_records.append({
            "text": text, "label": label, "font_size": style.fontSize,
            "relative_font_size": round(style.fontSize / BODY_FONT_SIZE, 2)
        })

    # Start with content directly (no title)
    num_pages = random.randint(MIN_PAGES, MAX_PAGES)
    for i in range(num_pages):
        # H1 header
        h1_txt = f"Section {i + 1}: {fake.catch_phrase()}"
        story.append(Paragraph(h1_txt, h1_st))
        add_para_record(h1_txt, "H1", h1_st)

        # Body paragraphs
        for _ in range(random.randint(2, 3)):
            p_txt = fake.paragraph(nb_sentences=random.randint(4, 8))
            story.append(Paragraph(p_txt, body_st))
            add_para_record(p_txt, "BODY", body_st)
        
        # Optional H2 section
        if random.random() > 0.4:
            h2_txt = fake.catch_phrase()
            story.append(Paragraph(h2_txt, h2_st))
            add_para_record(h2_txt, "H2", h2_st)

            # Optional H3 subsections
            if random.random() > 0.5:
                for j in range(random.randint(1, 2)):
                    h3_txt = f"Subsection {i+1}.{j+1} - {fake.bs()}"
                    story.append(Paragraph(h3_txt, h3_st))
                    add_para_record(h3_txt, "H3", h3_st)
                    p_txt = fake.paragraph(nb_sentences=2)
                    story.append(Paragraph(p_txt, body_st))
                    add_para_record(p_txt, "BODY", body_st)

        if i < num_pages - 1:
            story.append(PageBreak())

    # --- PDF and JSON Creation ---
    pdf_path = os.path.join(OUT_DIR, f"doc_{doc_id}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(story)

    # Extract line-by-line features
    line_records = extract_line_by_line_features(pdf_path, paragraph_records)
    
    # Add line spacing feature
    final_records = add_line_spacing_feature(line_records)

    json_path = pdf_path.replace('.pdf', '.labels.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)
    
    print(f"  -> Wrote {pdf_path} and {json_path} ({len(final_records)} lines)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PDF documents for model training.")
    parser.add_argument("--num", type=int, default=50, help="Number of documents to generate.")
    args = parser.parse_args()

    for i in range(args.num):
        generate_document(i)

    print(f"✅ Generated {args.num} PDF documents with labeled JSON in '{OUT_DIR}'")