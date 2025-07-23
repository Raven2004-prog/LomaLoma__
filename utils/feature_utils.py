import re
import numpy as np

def compute_median_font_size(lines):
    font_sizes = [line["font_size"] for line in lines]
    return np.median(font_sizes)

def line_to_features(lines, i, median_font_size):
    line = lines[i]
    prev_line = lines[i - 1] if i > 0 else None

    width_ratio = line["width"] / line.get("page_width", 595.2) # default A4 width
    center_x = line.get("x0", 0) + line.get("line_width", 0) / 2
    page_center = line.get("page_width", 595.2) / 2
    center_deviation = abs(center_x - page_center) / page_center

    features = {
        "y_position": round(line["y_position"], 1),
        "width_ratio": round(width_ratio, 2),
        "vertical_gap": round(line.get("space_before", 0.0), 1),
        "font_size_ratio": round(line["font_size"] / median_font_size, 2),
        "has_numeric_prefix": bool(re.match(r"^\d+(\.\d+)*", line["text"].strip())),
        "word_count": len(line["text"].split()),
        "center_deviation": round(center_deviation, 2),
        "size_vs_prev": round(line["font_size"] / prev_line["font_size"], 2) if prev_line else 1.0,
        "page_top_distance": round(line["y_position"] / line.get("page_height", 842), 2),
    }

    return features

def document_to_feature_sequence(lines):
    median_font_size = compute_median_font_size(lines)
    return [line_to_features(lines, i, median_font_size) for i in range(len(lines))]
