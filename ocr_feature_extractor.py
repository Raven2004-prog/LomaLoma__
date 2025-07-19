import pathlib
import pymupdf  # modern import instead of fitz
import io
import os
import pytesseract
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json

def is_page_empty(text):
    return not text.strip() or len(text.strip()) < 10

def ocr_page_with_features(pdf_path, page_num, dpi=150):
    import pymupdf
    from PIL import Image
    import pytesseract
    import io
    import os
    import numpy as np

    doc = pymupdf.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n = len(data["text"])

    lines = {}
    for i in range(n):
        txt = data["text"][i].strip()
        if not txt:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        left, top = data["left"][i], data["top"][i]
        width, height = data["width"][i], data["height"][i]

        if key not in lines:
            lines[key] = {
                "words": [], "lefts": [], "tops": [], "rights": [], "bottoms": []
            }

        grp = lines[key]
        grp["words"].append(txt)
        grp["lefts"].append(left)
        grp["tops"].append(top)
        grp["rights"].append(left + width)
        grp["bottoms"].append(top + height)

    enriched_lines = []
    for grp in lines.values():
        text = " ".join(grp["words"])
        x0, y0 = min(grp["lefts"]), min(grp["tops"])
        x1, y1 = max(grp["rights"]), max(grp["bottoms"])

        line_width = x1 - x0
        char_count = len(text)
        word_count = len(text.split())
        line_height = y1 - y0
        avg_char_width = line_width / char_count if char_count else 0

        enriched_lines.append({
            "text": text,
            "bbox": [x0, y0, x1, y1],
            "page_num": page_num + 1,
            "char_count": char_count,
            "word_count": word_count,
            "line_width": line_width,
            "line_height": line_height,
            "avg_char_width": avg_char_width,
            "all_caps": text.isupper(),
            "starts_with_digit": text.strip()[0].isdigit() if text.strip() else False
        })

    return (os.path.basename(pdf_path), page_num + 1, enriched_lines)

def process_pdf_extract_features(pdf_path, ocr_executor, scheduled_tasks):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        text = page.get_text("text")
        page_num = page.number
        if is_page_empty(text):
            future = ocr_executor.submit(ocr_page_with_features, str(pdf_path), page_num)
            scheduled_tasks.append(future)
        else:
            print(f"[{pdf_path.name} Page {page_num+1}] Parsed (text mode):\n{text}\n")
    doc.close()

def main():
    start_time = time.time()
    folder = pathlib.Path("input")
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDFs found in input folder.")
        return

    scheduled_tasks = []

    with ProcessPoolExecutor() as ocr_executor:
        for pdf_file in pdf_files:
            process_pdf_extract_features(pdf_file, ocr_executor, scheduled_tasks)

        for future in as_completed(scheduled_tasks):
            try:
                pdf_name, page_num, lines = future.result()
                for line in lines:
                    print(f"[{pdf_name} Page {page_num}] Line: {line['text']}")
                    print(f"  Features: {json.dumps(line, indent=2)}\n")
            except Exception as e:
                print(f"❌ OCR failed: {e}")

    elapsed = time.time() - start_time
    print(f"⏱️ Script completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
