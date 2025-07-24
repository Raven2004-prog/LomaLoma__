import os
import glob
import json

INPUT_DIR  = "data/synthetic"
OUTPUT_FILE = "data/synthetic_combined.json"

def stitch_json(input_dir, output_file):
    """Read all .json in input_dir, merge into one list, and write to output_file."""
    all_docs = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_docs.extend(data)
                else:
                    print(f"⚠️ Skipping {path}: not a JSON list")
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(all_docs, out, ensure_ascii=False, indent=2)
    print(f"✅ Stitched {len(all_docs)} entries into {output_file}")

if __name__ == "__main__":
    stitch_json(INPUT_DIR, OUTPUT_FILE)
