# prepare_dataset.py
import json
import glob

with open("dataset.jsonl", "w", encoding="utf-8") as out:
    for path in glob.glob("data/synthetic/*.json") + glob.glob("data/manual/*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            json.dump(data, out)
            out.write("\n")
