import json
import pandas as pd
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat"])

def enrich_entry_with_nlp(entry):
    text = entry.get("text", "").strip()
    doc = nlp(text)

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

    return {
        "is_all_caps": is_all_caps,
        "is_title_case": is_title_case,
        "starts_with_number": starts_with_number,
        "contains_colon": contains_colon,
        "contains_year": contains_year,
        "word_count": word_count,
        "avg_word_len": avg_word_len,
        "named_entity_ratio": named_entity_ratio
    }
def get_nlp_features(text):
    text = text.strip()
    doc = nlp(text)

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

    return {
        "is_all_caps": is_all_caps,
        "is_title_case": is_title_case,
        "starts_with_number": starts_with_number,
        "contains_colon": contains_colon,
        "contains_year": contains_year,
        "word_count": word_count,
        "avg_word_len": avg_word_len,
        "named_entity_ratio": named_entity_ratio
    }


def rebuild_features_from_labeled_json(input_json: str, output_file: str):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [entry.get("text", "") for entry in data]
    docs = list(nlp.pipe(texts, batch_size=64))

    updated_records = []
    for entry, doc in zip(data, docs):
        text = entry.get("text", "")

        # Layout-based features
        base = {
            "text": text,
            "font_size": entry.get("font_size", 0),
            "line_width": entry.get("line_width", 0),
            "line_height": entry.get("line_height", 0),
            "char_count": entry.get("char_count", len(text)),
            "page": entry.get("page", 0),
            "y_position": entry.get("y_position", 0),
            "label": entry.get("label")
        }

        # NLP-based features from pre-parsed doc
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

        semantic_feats = {
            "is_all_caps": is_all_caps,
            "is_title_case": is_title_case,
            "starts_with_number": starts_with_number,
            "contains_colon": contains_colon,
            "contains_year": contains_year,
            "word_count": word_count,
            "avg_word_len": avg_word_len,
            "named_entity_ratio": named_entity_ratio
        }

        combined = {**base, **semantic_feats}
        updated_records.append(combined)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(updated_records, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved enriched labeled dataset to {output_file}")


if __name__ == "__main__":
    input_path = "labeled_data.json"
    output_path = "labeled_data_with_features.json"
    rebuild_features_from_labeled_json(input_path, output_path)
