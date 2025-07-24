import re

def is_bold(flags):
    return (flags & (1 << 4)) > 0

def is_italic(flags):
    return (flags & (1 << 1)) > 0

def line_to_features(lines, i):
    line = lines[i]
    features = {
        'bias': 1.0,
        'font_size': round(line['font_size'], 1),
        'is_bold': is_bold(line['flags']),
        'is_italic': is_italic(line['flags']),
        'is_all_caps': line['text'].isupper(),
        'is_title_case': line['text'].istitle(),
        'ends_with_colon': line['text'].strip().endswith(':'),
        'line_length': len(line['text']),
        'word_count': len(line['text'].split()),
        'is_centered': abs(line['x0'] - (line['page_width'] - line['x1'])) < 5.0,
        'indentation': round(line['x0'], 1),
        'line_width_ratio': round(line['width'] / line['page_width'], 2),
        'numeric_prefix': bool(re.match(r'^[0-9]+(\.[0-9]+)*', line['text'])),
    }

    if line['space_before'] is not None:
        features['space_before'] = round(line['space_before'], 1)

    # Contextual features: previous line
    if i > 0:
        prev = lines[i - 1]
        features.update({
            '-1:is_bold': is_bold(prev['flags']),
            '-1:font_size': round(prev['font_size'], 1),
            '-1:line_length': len(prev['text'])
        })
    else:
        features['BOS'] = True  # Beginning of sequence

    # Contextual features: next line
    if i < len(lines) - 1:
        nxt = lines[i + 1]
        features.update({
            '+1:is_bold': is_bold(nxt['flags']),
            '+1:font_size': round(nxt['font_size'], 1),
            '+1:line_length': len(nxt['text'])
        })
    else:
        features['EOS'] = True  # End of sequence

    return features

def document_to_feature_sequence(lines):
    return [line_to_features(lines, i) for i in range(len(lines))]
