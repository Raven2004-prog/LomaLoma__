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
from reportlab.lib.pagesizes import A4, letter, legal
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors

# --- Basic Setup ---
fake = Faker()
Faker.seed(42)

# --- Configuration ---
OUT_DIR = "data/processed/"
os.makedirs(OUT_DIR, exist_ok=True)

DOC_TYPES = ["Report", "Analysis", "Study", "Review", "Proposal", "Manual", "Briefing", 
             "Memo", "Guideline", "Assessment", "Plan", "Summary"]
FONTS = ["Helvetica", "Courier", "Times-Roman", "Helvetica-Bold", "Courier-Bold", "Times-Bold"]
MIN_PAGES, MAX_PAGES = 5, 25
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

# FIXED: Added space_before parameter
def make_style(name, parent, family, size, alignment, bold=False, italic=False, 
               space_after=12, space_before=0, left_indent=0, right_indent=0, text_color=None,
               bulletIndent=0, first_line_indent=0):
    style = ParagraphStyle(
        name=name,
        parent=parent,
        fontName=make_font_name(family, bold, italic),
        fontSize=size,
        leading=size * 1.2,
        spaceAfter=space_after,
        spaceBefore=space_before,  # Added spaceBefore parameter
        leftIndent=left_indent,
        rightIndent=right_indent,
        alignment=alignment,
        firstLineIndent=first_line_indent,
        bulletIndent=bulletIndent
    )
    if text_color:
        style.textColor = text_color
    return style

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
        is_italic = "italic" in line_data["font_name"].lower() or "oblique" in line_data["font_name"].lower()
        is_centered = line_data["is_centered"]
        
        # Improved label determination logic
        label = "BODY"  # Default label
        
        # Title detection (first page, centered, large font)
        if line_data["page"] == 0 and is_centered and font_size > 20:
            label = "H1"
        # Header detection
        elif is_bold and not is_centered:
            if font_size >= 18:
                label = "H1"
            elif 16 <= font_size < 18:
                label = "H2"
            elif 12 <= font_size < 16:
                label = "H3"
        # Numbered/bulleted list detection
        elif re.match(r'^(\d+\.|•|[-*+])\s', line_data["text"].strip()):
            label = "BODY"  # Keep as body but mark as list item
            is_list_item = True
        
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
    
    # Page size variation
    page_size = random.choice([A4, letter, legal])
    
    # Create diverse styles
    title_st = make_style("Title", base_styles["h1"], family, random.randint(22, 28), 
                         TA_CENTER, bold=True, space_after=20)
    
    h1_st = make_style("H1", base_styles["h1"], family, random.randint(16, 20), 
                      TA_LEFT, bold=True, space_after=12)
    
    h2_st = make_style("H2", base_styles["h2"], family, random.randint(14, 16), 
                      TA_LEFT, bold=True, space_after=10, 
                      left_indent=random.choice([0, 0.2*inch, 0.4*inch]))
    
    h3_st = make_style("H3", base_styles["h3"], family, random.randint(12, 14), 
                      TA_LEFT, bold=True, italic=random.choice([True, False]), 
                      space_after=8, left_indent=random.choice([0.4*inch, 0.6*inch]))
    
    body_st = make_style("Body", base_styles["BodyText"], family, BODY_FONT_SIZE, 
                        random.choice([TA_LEFT, TA_JUSTIFY]), space_after=8)
    
    list_st = make_style("List", base_styles["BodyText"], family, BODY_FONT_SIZE, 
                        TA_LEFT, left_indent=0.5*inch, bulletIndent=0.25*inch)
    
    # Special styles for diversity
    quote_st = make_style("Quote", base_styles["BodyText"], family, BODY_FONT_SIZE, 
                         TA_JUSTIFY, italic=True, left_indent=0.5*inch, 
                         right_indent=0.5*inch, space_after=10)
    
    note_st = make_style("Note", base_styles["Italic"], family, BODY_FONT_SIZE-1, 
                        TA_LEFT, italic=True, text_color=colors.grey)
    
    important_st = make_style("Important", base_styles["BodyText"], family, BODY_FONT_SIZE, 
                             TA_LEFT, bold=True, text_color=colors.darkred)
    
    story = []

    # Title Page
    title_txt = f"{random.choice(DOC_TYPES)}: {fake.bs().title()}"
    story.append(Paragraph(title_txt, title_st))
    
    # Random subtitle
    if random.random() > 0.3:
        subtitle_st = make_style("Subtitle", base_styles["h2"], family, 16, 
                               TA_CENTER, italic=True, space_after=30)
        story.append(Paragraph(fake.catch_phrase(), subtitle_st))
    
    # Author/date line
    if random.random() > 0.2:
        author_st = make_style("Author", base_styles["BodyText"], family, 12, 
                             TA_CENTER, space_after=40)
        story.append(Paragraph(f"Prepared by: {fake.name()}", author_st))
        story.append(Paragraph(f"Date: {fake.date_this_year()}", author_st))
    
    story.append(PageBreak())

    # Table of Contents (randomly included)
    if random.random() > 0.6:
        toc_st = make_style("TOC", base_styles["h2"], family, 14, TA_CENTER, bold=True)
        story.append(Paragraph("Table of Contents", toc_st))
        story.append(Spacer(1, 0.2*inch))
        
        for i in range(random.randint(4, 8)):
            story.append(Paragraph(f"Section {i+1}: {fake.bs().title()}", body_st))
        
        story.append(PageBreak())

    # Main Content
    num_pages = random.randint(MIN_PAGES, MAX_PAGES)
    for page_num in range(num_pages):
        # Page header (optional)
        if random.random() > 0.7:
            header_st = make_style("Header", base_styles["BodyText"], family, 9, 
                                 TA_RIGHT, space_after=0)
            story.append(Paragraph(f"{title_txt} | Page {page_num+1}", header_st))
            story.append(Spacer(1, 0.1*inch))
        
        # Section header
        if page_num > 0 or random.random() > 0.2:  # 80% chance of section header
            h1_txt = f"Section {page_num + 1}: {fake.catch_phrase().title()}"
            story.append(Paragraph(h1_txt, h1_st))
        
        # Paragraphs with varied structure
        num_paragraphs = random.randint(1, 5)
        for para_num in range(num_paragraphs):
            # Random paragraph type
            p_type = random.choices(
                ["normal", "list", "quote", "note", "important"],
                weights=[6, 2, 1, 1, 1],
                k=1
            )[0]
            
            if p_type == "normal":
                # Standard paragraph
                p_txt = fake.paragraph(nb_sentences=random.randint(2, 8))
                story.append(Paragraph(p_txt, body_st))
                
            elif p_type == "list":
                # Bullet point list
                list_type = random.choice(["bullet", "number"])
                num_items = random.randint(3, 7)
                items = []
                
                for _ in range(num_items):
                    item = fake.sentence(nb_words=random.randint(4, 12))
                    items.append(item)
                
                if list_type == "bullet":
                    story.append(ListFlowable(
                        [Paragraph(item, list_st) for item in items],
                        bulletType='bullet',
                        start='bulletchar'
                    ))
                else:
                    story.append(ListFlowable(
                        [Paragraph(item, list_st) for item in items],
                        bulletType='1',
                        start=1
                    ))
                
            elif p_type == "quote":
                # Block quote
                quote_txt = fake.paragraph(nb_sentences=random.randint(1, 3))
                story.append(Paragraph(quote_txt, quote_st))
                
            elif p_type == "note":
                # Side note
                note_txt = f"NOTE: {fake.sentence(nb_words=random.randint(8, 15))}"
                story.append(Paragraph(note_txt, note_st))
                
            elif p_type == "important":
                # Important notice
                imp_txt = f"IMPORTANT: {fake.sentence(nb_words=random.randint(6, 12))}"
                story.append(Paragraph(imp_txt, important_st))
        
        # Subsection (randomly included)
        if random.random() > 0.4:
            h2_txt = fake.bs().title()
            story.append(Paragraph(h2_txt, h2_st))
            
            # Subsection content
            for _ in range(random.randint(1, 3)):
                if random.random() > 0.6:  # 40% chance of H3
                    h3_txt = f"{fake.bs().title()}:"
                    story.append(Paragraph(h3_txt, h3_st))
                
                # Mixed content under subsection
                content_type = random.choices(
                    ["paragraph", "list", "mixed"],
                    weights=[5, 3, 2],
                    k=1
                )[0]
                
                if content_type == "paragraph":
                    p_txt = fake.paragraph(nb_sentences=random.randint(2, 5))
                    story.append(Paragraph(p_txt, body_st))
                elif content_type == "list":
                    num_items = random.randint(2, 5)
                    items = [fake.sentence(nb_words=6) for _ in range(num_items)]
                    story.append(ListFlowable(
                        [Paragraph(item, list_st) for item in items],
                        bulletType='bullet'
                    ))
                else:  # mixed
                    p_txt = fake.paragraph(nb_sentences=2)
                    story.append(Paragraph(p_txt, body_st))
                    num_items = random.randint(2, 4)
                    items = [fake.sentence(nb_words=8) for _ in range(num_items)]
                    story.append(ListFlowable(
                        [Paragraph(item, list_st) for item in items],
                        bulletType='bullet'
                    ))
        
        # Page footer (optional)
        if random.random() > 0.6:
            story.append(Spacer(1, 0.3*inch))
            # FIXED: Changed space_before to space_before in make_style call
            footer_st = make_style("Footer", base_styles["BodyText"], family, 8, 
                                 TA_CENTER, space_before=0.2*inch)
            story.append(Paragraph(f"Confidential - {fake.company()}", footer_st))
        
        if page_num < num_pages - 1:
            story.append(PageBreak())

    # --- PDF and JSON Creation ---
    pdf_path = os.path.join(OUT_DIR, f"doc_{doc_id}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=page_size, 
                           leftMargin=random.choice([0.75*inch, 1*inch, 1.25*inch]),
                           rightMargin=random.choice([0.75*inch, 1*inch, 1.25*inch]),
                           topMargin=random.choice([0.5*inch, 0.75*inch, 1*inch]),
                           bottomMargin=random.choice([0.5*inch, 0.75*inch, 1*inch]))
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