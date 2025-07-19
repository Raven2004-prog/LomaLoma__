import pathlib
import pymupdf  # modern import
import io
import os
import pytesseract
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def is_page_empty(text):
    return not text.strip() or len(text.strip()) < 10


def ocr_page(pdf_path, page_num, dpi=150):
    """
    OCR fallback for a specific page.
    NOTE: This function runs in a subprocess, so imports must be inside.
    """
    import pymupdf
    from PIL import Image
    import pytesseract
    import io

    doc = pymupdf.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(img)
    return (os.path.basename(pdf_path), page_num + 1, text)



def process_pdf(pdf_path, ocr_executor, scheduled_tasks):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        text = page.get_text("text")
        page_num = page.number
        if is_page_empty(text):
            future = ocr_executor.submit(ocr_page, str(pdf_path), page_num)
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
            process_pdf(pdf_file, ocr_executor, scheduled_tasks)

        for future in as_completed(scheduled_tasks):
            try:
                pdf_name, page_num, text = future.result()
                print(f"[{pdf_name} Page {page_num}] OCR fallback:\n{text}\n")
            except Exception as e:
                print(f"❌ OCR failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n⏱️ Script completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
