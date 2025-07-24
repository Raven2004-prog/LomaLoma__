#!/usr/bin/env python3
import os
import glob
import json
import argparse
import sys

def stitch_json(input_dir: str, output_file: str):
    """
    Read all .json files in `input_dir`, merge their top‑level lists into one list,
    and write the combined list to `output_file`.
    """
    if not os.path.isdir(input_dir):
        print(f"❌ Input directory does not exist: {input_dir!r}")
        sys.exit(1)

    json_paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not json_paths:
        print(f"⚠️  No JSON files found in {input_dir!r}")
        sys.exit(0)

    all_docs = []
    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_docs.extend(data)
            else:
                print(f"⚠️  Skipping {os.path.basename(path)}: top‑level is not a list")
        except Exception as e:
            print(f"⚠️  Failed to load {os.path.basename(path)}: {e}")

    out_dir = os.path.dirname(output_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(all_docs, out_f, ensure_ascii=False, indent=2)
        print(f"✅ Stitched {len(all_docs)} total entries into {output_file!r}")
    except Exception as e:
        print(f"❌ Failed to write output file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Stitch all JSON files in a directory into one combined JSON list."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/Synthetic_data",
        help="Directory containing input .json files (default: data/synthetic)"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="data/synthetic_kusu.json",
        help="Path for the combined output JSON file (default: data/synthetic_kusu.json)"
    )
    args = parser.parse_args()
    stitch_json(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
