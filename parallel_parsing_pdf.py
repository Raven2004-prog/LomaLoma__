import pathlib
import pymupdf
import os
import json
import numpy as np
import io
import pytesseract
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_next_available_filename(folder, base_name="label", extension=".json"):
    i = 1
    while True:
        filename = f"{base_name}{i}{extension}"
        full_path = folder / filename
        if not full_path.exists():
            return full_path
        i += 1


def extract_text_features(pdf_path):
    doc = pymupdf.open(pdf_path)
    features = []
    ocr_tasks = []

    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        page_num = page.number
        has_text = any("lines" in block for block in blocks)

        if not has_text:
            ocr_tasks.append((str(pdf_path), page_num))
            continue

        for block in blocks:
            for line in block.get("lines", []):
                line_text = []
                font_sizes = []
                x0s, x1s = [], []
                y0s, y1s = [], []

                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    line_text.append(text)
                    font_sizes.append(span.get("size", 0))
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    x0s.append(bbox[0])
                    x1s.append(bbox[2])
                    y0s.append(bbox[1])
                    y1s.append(bbox[3])

                if not line_text:
                    continue

                text_combined = " ".join(line_text)
                font_size = np.median(font_sizes) if font_sizes else 0
                line_width = max(x1s) - min(x0s) if x0s and x1s else 0
                char_count = len(text_combined)
                line_height = max(y1s) - min(y0s) if y0s and y1s else 0
                y_position = min(y0s) if y0s else 0

                features.append({
                    "text": text_combined,
                    "font_size": font_size,
                    "line_width": line_width,
                    "line_height": line_height,
                    "char_count": char_count,
                    "page": page_num,
                    "y_position": y_position
                })

    doc.close()
    return features, ocr_tasks


def ocr_page(pdf_path, page_num, dpi=150):
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
            lines[key] = {"words": [], "lefts": [], "tops": [], "rights": [], "bottoms": []}
        grp = lines[key]
        grp["words"].append(txt)
        grp["lefts"].append(left)
        grp["tops"].append(top)
        grp["rights"].append(left + width)
        grp["bottoms"].append(top + height)

    enriched_lines = []
    for grp in lines.values():
        text = " ".join(grp["words"])
        x0 = min(grp["lefts"])
        y0 = min(grp["tops"])
        x1 = max(grp["rights"])
        y1 = max(grp["bottoms"])
        line_width = x1 - x0
        line_height = y1 - y0
        char_count = len(text)

        enriched_lines.append({
            "text": text,
            "font_size": 0,
            "line_width": line_width,
            "line_height": line_height,
            "char_count": char_count,
            "page": page_num + 1,
            "y_position": y0
        })

    doc.close()
    return enriched_lines


def main():
    input_folder = pathlib.Path("input")
    output_folder = pathlib.Path("output")
    output_folder.mkdir(exist_ok=True)

    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found in 'input/' folder.")
        return

    all_features = []
    all_ocr_tasks = []

    for pdf_path in pdf_files:
        print(f"📄 Extracting from: {pdf_path.name}")
        text_features, ocr_tasks = extract_text_features(pdf_path)
        all_features.extend(text_features)

        for task in ocr_tasks:
            all_ocr_tasks.append({
                "pdf_path": task[0],
                "page_num": task[1]
            })

    if all_ocr_tasks:
        print(f"🔍 Running OCR fallback for {len(all_ocr_tasks)} pages...")
        future_to_task = {}

        with ProcessPoolExecutor() as executor:
            for task in all_ocr_tasks:
                future = executor.submit(
                    ocr_page,
                    task["pdf_path"],
                    task["page_num"]
                )
                future_to_task[future] = task

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    lines = future.result()
                    all_features.extend(lines)
                    print(f"  ✓ OCR completed for {task['pdf_path']} page {task['page_num']+1}")
                except Exception as e:
                    print(f"❌ OCR failed for {task['pdf_path']} page {task['page_num']+1}: {e}")

    output_path = get_next_available_filename(output_folder)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_features, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(all_features)} text features to {output_path}")


if __name__ == "__main__":
    main()