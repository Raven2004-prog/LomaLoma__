import fitz  # PyMuPDF

def extract_lines_from_pdf(pdf_path):
    """
    Parses a PDF and returns a list of dictionaries,
    where each dictionary represents a text line with its properties.
    """
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        page_height = page.rect.height
        blocks = page.get_text("dict", flags=11)["blocks"]

        prev_bottom_y = None

        for block in blocks:
            if block['type'] == 0:  # Text block
                for line in block['lines']:
                    spans = line['spans']
                    if not spans:
                        continue

                    line_text = ''.join(span['text'] for span in spans).strip()
                    if not line_text:
                        continue

                    # Use first span for font and style (approximation)
                    first_span = spans[0]

                    x0, y0, x1, y1 = line['bbox']
                    line_width = x1 - x0

                    line_info = {
                        'text': line_text,
                        'font': first_span['font'],
                        'font_size': first_span['size'],
                        'flags': first_span['flags'],  # for bold/italic
                        'bbox': line['bbox'],
                        'x0': x0,
                        'y0': y0,
                        'x1': x1,
                        'y1': y1,
                        'width': line_width,
                        'page_num': page_num,
                        'page_width': page_width,
                        'page_height': page_height,
                        'space_before': (y0 - prev_bottom_y) if prev_bottom_y is not None else None
                    }

                    prev_bottom_y = y1
                    all_lines.append(line_info)

    return all_lines


if __name__ == "__main__":
    from pprint import pprint
    lines = extract_lines_from_pdf("data/synthetic/doc_0.pdf")
    pprint(lines[:5])
