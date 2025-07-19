import pymupdf
import pathlib
import time  # <-- Import the time module

start_time = time.time()  # <-- Start timing

folder = pathlib.Path("input")
pdf_files = list(folder.glob("*.pdf"))

for pdf_file in pdf_files:
    doc = pymupdf.open(pdf_file)
    for page in doc:
        text = page.get_text("text")
        if not text.strip():
            print("OCR FALLBACK")
            tp = page.get_textpage_ocr()
            text = page.get_text(textpage=tp)
            print(f"Text from {pdf_file.name}, page {page.number + 1}:\n{text}\n")
        else:
            print(f"Text from {pdf_file.name}, page {page.number + 1}:\n{text}\n")
    doc.close()

end_time = time.time()  # <-- End timing
elapsed = end_time - start_time
print(f"\n⏱️ Script completed in {elapsed:.2f} seconds")
