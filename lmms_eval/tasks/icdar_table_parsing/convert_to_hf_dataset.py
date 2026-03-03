"""Convert ICDAR 2021 Table Parsing local dataset to HuggingFace Dataset format.

Input format:
    data_dir/
    ├── final_eval.json     # dict: {filename: {"html": str, "type": "simple"|"complex"}}
    └── final_eval/         # directory of PNG images matching JSON keys

Output:
    data_dir/hf_dataset/    # HuggingFace DatasetDict with 'test' split
        columns: image (Image), html (str), type (str), filename (str)

Usage:
    python -m lmms_eval.tasks.icdar_table_parsing.convert_to_hf_dataset
    python -m lmms_eval.tasks.icdar_table_parsing.convert_to_hf_dataset --data_dir /path/to/data
    python -m lmms_eval.tasks.icdar_table_parsing.convert_to_hf_dataset --skip_files file1.png file2.png
"""

from __future__ import annotations

import argparse
import json
import os

import datasets

DEFAULT_DATA_DIR = "/data/workspace/datasets/vdu/icdar_2021_table_parsing"

SKIP_FILES = {
    "a0797c6d5a5003694e665466b9b5aa1277dc8b8de8300178e2d9348aedb910c2.png",
}


def convert(data_dir: str, skip_files: set[str] | None = None) -> datasets.DatasetDict:
    if skip_files is None:
        skip_files = SKIP_FILES

    json_path = os.path.join(data_dir, "final_eval.json")
    img_dir = os.path.join(data_dir, "final_eval")
    out_dir = os.path.join(data_dir, "hf_dataset")

    with open(json_path) as f:
        data = json.load(f)

    filenames = []
    htmls = []
    types = []
    images = []
    skipped = []

    for filename, info in data.items():
        if filename in skip_files:
            skipped.append(filename)
            continue

        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            skipped.append(filename)
            continue

        filenames.append(filename)
        htmls.append(info["html"])
        types.append(info["type"])
        images.append(img_path)

    print(f"Total samples: {len(filenames)}")
    print(f"Skipped: {len(skipped)} ({skipped})")
    print(f"Types: simple={types.count('simple')}, complex={types.count('complex')}")

    ds = datasets.Dataset.from_dict(
        {
            "image": images,
            "html": htmls,
            "type": types,
            "filename": filenames,
        }
    )
    ds = ds.cast_column("image", datasets.Image())

    ds_dict = datasets.DatasetDict({"test": ds})
    ds_dict.save_to_disk(out_dir)
    print(f"Saved HF dataset to {out_dir}")
    print(ds_dict)
    return ds_dict


def main():
    parser = argparse.ArgumentParser(description="Convert ICDAR 2021 Table Parsing dataset to HuggingFace format")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Path to dataset root directory")
    parser.add_argument("--skip_files", nargs="*", default=None, help="Filenames to skip (default: known oversized files)")
    args = parser.parse_args()

    skip = set(args.skip_files) if args.skip_files is not None else SKIP_FILES
    convert(args.data_dir, skip_files=skip)


if __name__ == "__main__":
    main()
