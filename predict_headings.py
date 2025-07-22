import json
import joblib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from parallel_parsing_pdf import extract_text_features, ocr_page
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat"])

# Features expected by the model
features_to_use = [
    "font_size", "line_width", "line_height", "char_count", "y_position",
    "is_all_caps", "is_title_case", "starts_with_number", "contains_colon",
    "contains_year", "word_count", "avg_word_len", "named_entity_ratio"
]

def run_parser_pipeline():
    input_folder = Path("input")
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)

    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found in input/")
        return None

    all_features = []
    all_ocr_tasks = []

    for pdf_path in pdf_files:
        print(f"📄 Parsing: {pdf_path.name}")
        text_features, ocr_tasks = extract_text_features(pdf_path)
        for feat in text_features:
            feat["pdf_name"] = pdf_path.name
        all_features.extend(text_features)

        for task in ocr_tasks:
            all_ocr_tasks.append({"pdf_path": task[0], "page_num": task[1], "pdf_name": pdf_path.name})

    if all_ocr_tasks:
        print(f"🔍 Running OCR on {len(all_ocr_tasks)} pages...")
        with ProcessPoolExecutor() as executor:
            future_to_task = {
                executor.submit(ocr_page, task["pdf_path"], task["page_num"]): task
                for task in all_ocr_tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    lines = future.result()
                    for line in lines:
                        line["pdf_name"] = task["pdf_name"]
                    all_features.extend(lines)
                except Exception as e:
                    print(f"❌ OCR failed for {task['pdf_path']} page {task['page_num'] + 1}: {e}")

    output_path = output_folder / "features.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_features, f, indent=2, ensure_ascii=False)
    print(f"✅ Features written to {output_path}")
    return output_path

def run_inference():
    features_path = run_parser_pipeline()
    if features_path is None:
        return

    # Load features
    with open(features_path, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    # Fast batch NLP feature enrichment
    texts = [entry.get("text", "") for entry in parsed_data]
    docs = list(nlp.pipe(texts, batch_size=64))

    enriched_data = []
    for entry, doc in zip(parsed_data, docs):
        text = entry.get("text", "")
        is_all_caps = text.isupper()
        is_title_case = text.istitle()
        starts_with_number = text[:2].strip().split(" ")[0].isdigit() if text else False
        contains_colon = ":" in text
        contains_year = any(str(y) in text for y in range(1990, 2031))

        words = [token.text for token in doc if token.is_alpha]
        word_count = len(words)
        avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        ner_count = len([ent for ent in doc.ents])
        named_entity_ratio = ner_count / word_count if word_count > 0 else 0

        entry.update({
            "is_all_caps": is_all_caps,
            "is_title_case": is_title_case,
            "starts_with_number": starts_with_number,
            "contains_colon": contains_colon,
            "contains_year": contains_year,
            "word_count": word_count,
            "avg_word_len": avg_word_len,
            "named_entity_ratio": named_entity_ratio
        })
        enriched_data.append(entry)

    # Load model & label encoder
    clf = joblib.load("models/heading_classifier.joblib")
    le = joblib.load("models/label_encoder.joblib")

    outline = []
    for entry in enriched_data:
        try:
            vec = [
                float(entry.get(f, 0)) if isinstance(entry.get(f), (int, float))
                else int(entry.get(f, False))
                for f in features_to_use
            ]
            label_idx = clf.predict([vec])[0]
            label = le.inverse_transform([label_idx])[0]

            if label in ["H1", "H2", "H3"]:
                outline.append({
                    "level": label,
                    "text": entry["text"],
                    "page": entry["page"]
                })
        except Exception as e:
            print(f"⚠️ Skipping line due to error: {e}")

    final_output = {
        "title": enriched_data[0].get("pdf_name", "Untitled Document"),
        "outline": outline
    }

    output_path = Path("output") / "output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"✅ Final predictions saved to {output_path}")

if __name__ == "__main__":
    run_inference()
