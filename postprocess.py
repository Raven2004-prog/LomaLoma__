def build_hierarchy(lines, labels):
    result = {"title": None, "children": []}
    stack = [result]  # stack[1] = H1, stack[2] = H2, etc.

    title_candidates = []
    heading_blocks = []

    for line, label in zip(lines, labels):
        text = line['text'].strip()
        if not text:
            continue

        if label == "Title":
            title_candidates.append((line['font_size'], line['y0'], text))

        elif label.startswith("H"):
            level = int(label[1])
            if text == "•":
                continue  # skip bullet headings

            node = {"heading": text, "children": [], "content": []}
            while len(stack) > level:
                stack.pop()
            stack[-1]["children"].append(node)
            stack.append(node)
            heading_blocks.append((text, line['font_size'], line['y0']))

        elif label == "BODY":
            if len(stack) > 1:
                stack[-1]["content"].append(text)
            else:
                result.setdefault("content", []).append(text)

    # Improved title logic: check first 2 heading blocks
    if not result["title"] and heading_blocks:
        for text, size, y in heading_blocks[:2]:
            if len(text.strip().split()) > 3 and text != "•":
                result["title"] = text
                break

    return result


if __name__ == "__main__":
    import sys, json, os
    from predict import predict_labels

    if len(sys.argv) != 2:
        print("Usage: python postprocess.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    lines, labels = predict_labels(pdf_path)
    hierarchy = build_hierarchy(lines, labels)

    if not hierarchy["title"]:
        hierarchy["title"] = os.path.splitext(os.path.basename(pdf_path))[0]

    print(json.dumps(hierarchy, indent=2, ensure_ascii=False))