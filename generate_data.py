# generate_data.py
import os
import json
import random
import argparse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from utils.pdf_util import extract_lines_from_pdf
from utils.feature_utils import document_to_feature_sequence

output_dir = "data/synthetic"
os.makedirs(output_dir, exist_ok=True)

FONT_NAMES = ["Helvetica", "Courier", "Times-Roman"]
ALIGNMENTS = [TA_LEFT, TA_CENTER]

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
    else:
        if bold and italic:
            return "Helvetica-BoldOblique"
        elif bold:
            return "Helvetica-Bold"
        elif italic:
            return "Helvetica-Oblique"
        else:
            return "Helvetica"

def make_heading_style(base, font_size, font_family, alignment, bold=True, italic=False):
    font_name = make_font_name(font_family, bold, italic)
    return ParagraphStyle(
        name=f"{font_name}_{font_size}_{bold}_{italic}",
        parent=base,
        fontName=font_name,
        fontSize=font_size,
        spaceAfter=random.randint(6, 14),
        leftIndent=random.choice([0, 20]),
        alignment=alignment
    )

def generate_document(doc_id):
    story = []
    labels = []

    font = random.choice(FONT_NAMES)
    align = random.choice(ALIGNMENTS)
    use_italic = random.choice([True, False])

    ambiguous_styles = random.random() < 0.3
    base_size = 11 if ambiguous_styles else None

    styles = getSampleStyleSheet()
    heading_styles = {
        "Title": make_heading_style(styles['Heading1'], base_size or 24, font, align, bold=True, italic=use_italic),
        "H1": make_heading_style(styles['Heading2'], base_size or 20, font, align, bold=True, italic=use_italic),
        "H2": make_heading_style(styles['Heading3'], base_size or 16, font, align, bold=False, italic=use_italic),
        "H3": make_heading_style(styles['Heading4'], base_size or 14, font, align, bold=False, italic=use_italic),
        "BODY": make_heading_style(styles['BodyText'], 11, font, TA_LEFT, bold=False, italic=use_italic)
    }

    story.append(Paragraph("Sample Document Title", heading_styles["Title"]))
    labels.append("Title")
    story.append(Spacer(1, 0.2 * inch))

    for i in range(3):
        story.append(Paragraph(f"{i+1}. Section {i+1}", heading_styles["H1"]))
        labels.append("H1")
        for j in range(2):
            story.append(Paragraph(f"{i+1}.{j+1} Subsection", heading_styles["H2"]))
            labels.append("H2")
            for k in range(2):
                story.append(Paragraph(f"{i+1}.{j+1}.{k+1} Detail Header", heading_styles["H3"]))
                labels.append("H3")
                for _ in range(3):
                    story.append(Paragraph(
                        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", heading_styles["BODY"]
                    ))
                    labels.append("BODY")

    pdf_path = os.path.join(output_dir, f"doc_{doc_id}.pdf")
    json_path = os.path.join(output_dir, f"doc_{doc_id}.json")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(story)

    # Now extract lines and generate feature-rich json
    lines = extract_lines_from_pdf(pdf_path)
    features = document_to_feature_sequence(lines)
    structured = []
    for i in range(min(len(features), len(labels))):
        structured.append({
            "text": lines[i]["text"],
            "features": features[i],
            "label": labels[i]
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="Number of synthetic documents to generate")
    args = parser.parse_args()

    for i in range(args.num):
        generate_document(i)

    print(f"{args.num} synthetic documents generated in {output_dir}.")