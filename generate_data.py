import argparse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib import colors
import os
import random
import json
import re
from datetime import datetime
import fitz  # PyMuPDF for PDF parsing

# Setup
styles = getSampleStyleSheet()
out_dir = "data/synthetic"
os.makedirs(out_dir, exist_ok=True)

# Constants
FONT_NAMES = ["Helvetica", "Courier", "Times-Roman"]
ALIGNMENTS = [TA_LEFT, TA_CENTER]
SECTION_LEVELS = ["H1", "H2", "H3"]  # Only up to H3
LONG_SECTION_TITLES = [
    "Financial Performance Analysis Q3 2023",
    "Market Expansion Strategy Implementation",
    "Customer Satisfaction Metrics and Improvement Plans",
    "Operational Efficiency Optimization Framework",
    "Risk Management Assessment and Mitigation Strategies"
]

# Utility to select valid font variants
def make_font_name(base, bold, italic):
    base = base.lower()
    if base == "times-roman":
        if bold and italic:
            return "Times-BoldItalic"
        elif bold:
            return "Times-Bold"
        elif italic:
            return "Times-Italic"
        else:
            return "Times-Roman"
    elif base == "courier":
        if bold and italic:
            return "Courier-BoldOblique"
        elif bold:
            return "Courier-Bold"
        elif italic:
            return "Courier-Oblique"
        else:
            return "Courier"
    else:  # default Helvetica
        if bold and italic:
            return "Helvetica-BoldOblique"
        elif bold:
            return "Helvetica-Bold"
        elif italic:
            return "Helvetica-Oblique"
        else:
            return "Helvetica"

# Create paragraph style with variability
def make_style(base, font_size, font_family, alignment, bold=True, italic=False):
    font_name = make_font_name(font_family, bold, italic)
    return ParagraphStyle(
        name=f"{font_name}_{font_size}_{bold}_{italic}",
        parent=base,
        fontName=font_name,
        fontSize=font_size,
        spaceAfter=random.randint(4, 12),
        leftIndent=random.choice([0, 10, 20, 40]),
        alignment=alignment
    )

# Generate table of contents
def generate_toc(sections):
    toc_data = [["<b>Table of Contents</b>", "<b>Page</b>"]]
    page_counter = 3  # Start after title and TOC pages
    
    for section in sections:
        # Add main section
        toc_data.append([section['title'], str(page_counter)])
        page_counter += 1
        
        # Add subsections
        for sub in section['subsections']:
            toc_data.append([f"&nbsp;&nbsp;&nbsp;&nbsp;{sub}", str(page_counter)])
            page_counter += 1
            
        # Add false TOC entry randomly
        if random.random() < 0.3:
            false_entry = random.choice([
                f"Appendix {chr(random.randint(65, 70))}",
                "Acknowledgements",
                "List of Tables",
                "Bibliography"
            ])
            toc_data.append([false_entry, str(page_counter)])
            page_counter += 1
    
    # Create table
    toc_table = Table(toc_data, colWidths=[400, 50])
    toc_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 12),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEADING', (0, 0), (-1, -1), 14),
    ]))
    
    return toc_table

# Enhanced feature extraction with all requested features
def extract_features(text, style):
    """Extract rich features from text and style"""
    # Text-based features
    length = len(text)
    words = text.split()
    word_count = len(words)
    has_digit = any(char.isdigit() for char in text)
    has_special = any(not char.isalnum() and not char.isspace() for char in text)
    is_capitalized = text.strip()[0].isupper() if text.strip() else False
    is_upper = text.isupper()
    is_title = text.istitle()
    ends_with_colon = text.strip().endswith(':')
    starts_with_bullet = text.strip().startswith(('•', '-', '*', '→'))
    
    # Style-based features
    font_size = style.fontSize
    font_name = style.fontName
    alignment = style.alignment
    left_indent = style.leftIndent
    space_after = style.spaceAfter
    is_bold = "Bold" in font_name
    is_italic = "Italic" in font_name or "Oblique" in font_name
    
    # Content-based features
    is_section_number = bool(re.match(r'^(?:(?:\d+\.)+\s?|\d+\)\s?|[IVX]+\.\s?)', text.strip()))
    is_short = length < 30
    is_very_short = length < 5
    is_empty = length == 0
    starts_with_number = text.strip()[0].isdigit() if text.strip() else False
    contains_colon = ':' in text
    
    # Year detection (4-digit years between 1900-2099)
    contains_year = bool(re.search(r'\b(19|20)\d{2}\b', text))
    
    # Word statistics
    avg_word_len = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Named entity estimation (simplified)
    capital_words = sum(1 for word in words if word[0].isupper() and len(word) > 1)
    named_entity_ratio = capital_words / word_count if word_count > 0 else 0
    
    # Return flat dictionary of features
    return {
        "char_count": length,
        "word_count": word_count,
        "is_all_caps": is_upper,
        "is_title_case": is_title,
        "starts_with_number": starts_with_number,
        "contains_colon": contains_colon,
        "contains_year": contains_year,
        "avg_word_len": round(avg_word_len, 2),
        "named_entity_ratio": round(named_entity_ratio, 4),
        "font_size": font_size,
        "font_name": font_name,
        "alignment": alignment,
        "left_indent": left_indent,
        "space_after": space_after,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "has_digit": has_digit,
        "has_special_char": has_special,
        "is_capitalized": is_capitalized,
        "ends_with_colon": ends_with_colon,
        "starts_with_bullet": starts_with_bullet,
        "is_section_number": is_section_number,
        "is_short_text": is_short,
        "is_very_short_text": is_very_short,
        "is_empty_line": is_empty,
        # Layout features (to be populated later)
        "line_width": None,
        "line_height": None,
        "page": None,
        "y_position": None
    }

# Extract layout features from PDF
def extract_layout_features(pdf_path, labeled_data):
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            if block_type == 0 and text.strip():  # Text block with content
                all_blocks.append({
                    "text": text.strip(),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "page": page_num + 1
                })
    
    # Match extracted blocks with labeled data
    for item in labeled_data:
        item_text = item['text'].strip()
        for block in all_blocks:
            # Simple text matching
            if item_text and block['text'] and item_text in block['text']:
                # Update layout features directly in the item dictionary
                item['line_width'] = block['x1'] - block['x0']
                item['line_height'] = block['y1'] - block['y0']
                item['page'] = block['page']
                item['y_position'] = block['y0']
                break
    
    return labeled_data

# Generate one synthetic document
def generate_document(doc_id):
    story = []
    labeled_data = []
    sections = []

    # Random document-level features
    font = random.choice(FONT_NAMES)
    align = random.choice(ALIGNMENTS)
    italic_flag = random.choice([False, True])

    # Define styles (only up to H3)
    styles_map = {
        "Title": make_style(styles['Heading1'], 24, font, TA_CENTER, bold=True, italic=italic_flag),
        "TOC_Heading": make_style(styles['Heading1'], 18, font, TA_CENTER, bold=True, italic=italic_flag),
        "H1":    make_style(styles['Heading2'], 20, font, align, bold=True, italic=italic_flag),
        "H2":    make_style(styles['Heading3'], 16, font, align, bold=False, italic=italic_flag),
        "H3":    make_style(styles['Heading4'], 14, font, align, bold=False, italic=italic_flag),
        "BODY":  make_style(styles['BodyText'], 11, font, TA_LEFT, bold=random.choice([False, True]), italic=italic_flag),
        "HEADER": make_style(styles['BodyText'], 9, font, TA_RIGHT, bold=False, italic=False),
        "FOOTER": make_style(styles['BodyText'], 8, font, TA_CENTER, bold=False, italic=True)
    }

    def add(text, level):
        p = Paragraph(text, styles_map[level])
        story.append(p)
        # Create the data entry with text, label, and all features
        entry = {
            "text": text,
            "label": level
        }
        # Merge in the features from extract_features
        entry.update(extract_features(text, styles_map[level]))
        labeled_data.append(entry)
        # random spacing to simulate variability
        story.append(Spacer(1, random.uniform(0.02, 0.15) * inch))

    # Title Page
    title_text = "Synthetic Document " + str(doc_id + 1)
    add(title_text, "Title")
    
    # Add false heading on title page
    if random.random() < 0.4:
        add(f"Document ID: {random.randint(1000, 9999)}-{random.randint(100, 999)}", "BODY")
    
    # Page break to TOC
    story.append(PageBreak())
    
    # Generate sections for TOC (only up to H3)
    num_sections = random.randint(3, 6)
    for i in range(1, num_sections + 1):
        # Create section with long title resembling section name
        section_title = f"{i}. {random.choice(LONG_SECTION_TITLES)}"
        subsections = []
        
        # Add subsections (H2 level)
        num_subsections = random.randint(2, 4)
        for j in range(1, num_subsections + 1):
            # Create subsection with detailed numbering
            subsection_title = f"{i}.{j} {random.choice(LONG_SECTION_TITLES)}"
            subsections.append(subsection_title)
        
        sections.append({
            'title': section_title,
            'subsections': subsections
        })
    
    # Add TOC heading
    add("Table of Contents", "TOC_Heading")
    
    # Generate and add TOC
    toc = generate_toc(sections)
    story.append(toc)
    
    # Add false TOC entry
    if random.random() < 0.3:
        add(f"* Unnumbered Section {random.randint(1, 10)}", "BODY")
    
    # Page break to content
    story.append(PageBreak())
    
    # Add header/footer
    if random.random() < 0.5:
        add(f"Confidential - Document {doc_id+1}", "HEADER")
    if random.random() < 0.5:
        add(f"Page {random.randint(1, 10)} of {random.randint(10, 20)}", "FOOTER")
    
    # Add false headings before content
    if random.random() < 0.4:
        false_heading = random.choice([
            str(random.randint(1, 10)),
            f"{random.randint(1, 5)}.{random.randint(1, 9)}",
            "DRAFT",
            datetime.now().strftime("%Y-%m-%d")
        ])
        add(false_heading, "BODY")
    
    # Generate document content (only up to H3)
    for section in sections:
        # Add main section (H1)
        add(section['title'], "H1")
        
        # Add false heading
        if random.random() < 0.3:
            false_heading = random.choice([
                f"{random.randint(1, 10)}",
                "Interim Report",
                "Preliminary Findings",
                "Supplementary Materials"
            ])
            add(false_heading, "BODY")
        
        # Add very short text
        if random.random() < 0.4:
            short_text = random.choice([
                f"{random.randint(1, 10)}.",
                f"{random.randint(1, 5)}.{random.randint(1, 9)}",
                "•",
                "→",
                "*"
            ])
            add(short_text, "BODY")
        
        # Add body text
        add("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.", "BODY")
        
        # Add subsections (H2 level)
        for sub in section['subsections']:
            add(sub, "H2")
            
            # Add H3 level content
            if random.random() < 0.5:
                short_sub = f"{random.randint(1, 5)}.{random.randint(1, 9)}.{random.randint(1, 9)}"
                add(short_sub, "H3")
                
                # Add very short text
                if random.random() < 0.6:
                    add(f"{random.choice(['a', 'b', 'c'])})", "BODY")
            
            # Add body text
            add("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.", "BODY")
    
    # Add appendix with false headings
    if random.random() < 0.7:
        add("Appendix", "H1")
        
        # Add numbered appendix sections (H2 level)
        for i in range(1, random.randint(2, 4)):
            add(f"Appendix {chr(64+i)}: Supplementary Data", "H2")
            
            # Add false heading
            if random.random() < 0.5:
                add(f"Attachment {i}", "BODY")
            
            # Add very short text
            if random.random() < 0.5:
                add(f"{i}.)", "BODY")
            
            add("Additional supporting materials and data references are available upon request.", "BODY")

    # Save PDF
    pdf_path = os.path.join(out_dir, f"doc_{doc_id}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(story)
    
    # Extract layout features from PDF
    labeled_data = extract_layout_features(pdf_path, labeled_data)
    
    # Save JSON with all features
    json_path = os.path.join(out_dir, f"doc_{doc_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, ensure_ascii=False, indent=2)

# CLI entrypoint
if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=60, help="Number of docs to generate")
    args = parser.parse_args()
    
    for n in range(args.num):
        generate_document(n)
    print(f"Generated {args.num} docs in {out_dir}")