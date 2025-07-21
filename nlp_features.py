import re
import string
import numpy as np
import spacy

# Load the small English NLP model from spaCy (must be installed via `python -m spacy download en_core_web_sm`)
nlp = spacy.load("en_core_web_sm")

def get_nlp_features(text):
    doc = nlp(text)

    is_all_caps = text.isupper()
    is_title_case = text.istitle()
    starts_with_number = bool(re.match(r"^\d+\.?", text.strip()))
    contains_colon = ":" in text
    contains_year = bool(re.search(r"\b(19|20)\d{2}\b", text))

    words = [token.text for token in doc if token.is_alpha]
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    pos_tags = [token.pos_ for token in doc]
    pos_pattern = " ".join(pos_tags)

    named_entities = list(doc.ents)
    named_entity_ratio = len(named_entities) / word_count if word_count else 0

    return {
        "is_all_caps": is_all_caps,
        "is_title_case": is_title_case,
        "starts_with_number": starts_with_number,
        "contains_colon": contains_colon,
        "contains_year": contains_year,
        "word_count": word_count,
        "avg_word_len": avg_word_len,
        "pos_pattern": pos_pattern,
        "named_entity_ratio": named_entity_ratio,
    }
