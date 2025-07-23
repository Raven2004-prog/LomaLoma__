import argparse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
import random
import json

styles = getSampleStyleSheet()
out_dir = "data/synthetic"
os.makedirs(out_dir, exist_ok=True)

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
    else:  # default Helvetica
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
        leftIndent=random.choice([0, 10, 20, 40]),
        alignment=alignment
    )

def generate_document(doc_id):
    story = []
    labeled_lines = []

    font = random.choice(FONT_NAMES)
    align = random.choice(ALIGNMENTS)
    use_italic = random.choice([True, False])

    ambiguous_styles = random.random() < 0.3
    base_size = 11 if ambiguous_styles else None

    heading_styles = {
        "Title": make_heading_style(styles['Heading1'], base_size or 24, font, align, bold=True, italic=use_italic),
        "H1": make_heading_style(styles['Heading2'], base_size or 20, font, align, bold=True, italic=use_italic),
        "H2": make_heading_style(styles['Heading3'], base_size or 16, font, align, bold=False, italic=use_italic),
        "H3": make_heading_style(styles['Heading4'], base_size or 14, font, align, bold=False, italic=use_italic),
        "BODY": make_heading_style(styles['BodyText'], 11, font, TA_LEFT, bold=random.choice([False, True]), italic=use_italic)
    }

    def add_line(text, style, label):
        story.append(Paragraph(text, style))
        labeled_lines.append({"text": text, "label": label})
        story.append(Spacer(1, random.uniform(0.03, 0.12) * inch))

    # Title candidates
    if random.random() < 0.3:
        add_line("Special Report", heading_styles["Title"], "BODY")  # misleading title
    if random.random() < 0.3:
        add_line("TABLE OF CONTENTS", heading_styles["H1"], "BODY")  # always body
    add_line("Synthetic Document Title", heading_styles["Title"], "Title")

    bullet_items = [
        "• High-level overview",
        "• Deep insights provided",
        "• Case studies included",
        "• Relevant examples",
        "• Practical applications"
    ]

    for i in range(3):
        h1_text = random.choice([f"{i+1}. Overview Section", f"CHAPTER {i+1}", f"Part {i+1}"])
        add_line(h1_text, heading_styles["H1"], "H1")

        for j in range(2):
            h2_text = random.choice([f"{i+1}.{j+1} Insights", f"{i+1}.{j+1} Discussion Points", f"Section {j+1}"])
            add_line(h2_text, heading_styles["H2"], "H2")

            for bullet in random.sample(bullet_items, 2):
                bullet_style = heading_styles["H2"] if random.random() < 0.2 else heading_styles["BODY"]
                bullet_label = "BODY"
                add_line(bullet, bullet_style, bullet_label)

            for k in range(2):
                h3_text = random.choice([
                    f"{i+1}.{j+1}.{k+1} Detail Topic.",
                    f"Highlight {k+1}: Key Element",
                    f"• Emphasis Point {k+1}"
                ])
                add_line(h3_text, heading_styles["H3"], "H3")
                for _ in range(2):
                    body = random.choice([
                        "This is a descriptive paragraph for the section.",
                        "Additional information is shared in this part.",
                        "Details are expanded here with context."
                    ])
                    add_line(body, heading_styles["BODY"], "BODY")

    pdf_path = os.path.join(out_dir, f"doc_{doc_id}.pdf")
    label_path = os.path.join(out_dir, f"doc_{doc_id}.labels.json")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(story)

    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(labeled_lines, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="Number of synthetic documents to generate")
    args = parser.parse_args()

    for i in range(args.num):
        generate_document(i)

    print(f"{args.num} synthetic documents generated in {out_dir}.")