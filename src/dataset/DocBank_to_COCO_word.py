import os
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import random

TEXT_LABELS = {"abstract", "paragraph", "title", "author"}

def convert_label(label):
    return 1 if label.lower() in TEXT_LABELS else 2

import pickle

def load_split_ids(split_json):
    cache_path = Path(split_json).with_suffix(".split_ids.pkl")

    if cache_path.exists():
        print(f"✅ Loading cached split IDs from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"📦 Loading split JSON: {split_json}")
    with open(split_json, "r") as f:
        coco = json.load(f)

    split_ids = set(
        img["file_name"].replace(".jpg", "")
        for img in coco["images"]
    )
    print(f"💾 Caching split IDs to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(split_ids, f)


    return split_ids


def parse_txt_file(txt_path):
    try:
        df = pd.read_table(
            txt_path,
            header=None,
            names=["token", "x0", "y0", "x1", "y1", "R", "G", "B", "font", "label"],
            on_bad_lines="skip",
            engine="python"
        )
        return df
    except Exception as e:
        print(f"❌ Failed to parse {txt_path.name}: {e}")
        return pd.DataFrame()

PDF_WIDTH = 612  # constant in PDF points
def main(args):
    
    txt_paths = list(Path(args.src_dir).rglob("*.txt"))
    random.seed(42)  # Optional: makes results reproducible
    random.shuffle(txt_paths)

    split_ids = load_split_ids(args.split_json) if args.split_json else None
    
    images = []
    annotations = []
    ann_id = 0
    img_id = 0

    for txt_path in tqdm(txt_paths, desc="Processing"):
        base_name = txt_path.stem  # e.g. 22.tar_1402.4604.gz_cordero-revised_22
        image_base = base_name + "_ori"  # match image name and split ID

        if split_ids and image_base not in split_ids:
            #print(f"Skipping {image_base} as it's not in the split JSON")
            continue

        img_path = Path(args.image_root) / f"{image_base}.jpg"
        if not img_path.is_file():
            #print(f"Skipping {image_base}, missing image: {img_path}")
            continue

        if args.max_images is not None and img_id >= args.max_images:
            break

        img = Image.open(img_path)
        img_width, img_height = img.size

        df = parse_txt_file(txt_path)
        if df.empty:
            continue

        images.append({
            "id": img_id,
            "file_name": f"{image_base}.jpg",
            "height": None,
            "width": None,
            "date_captured": "",
            "license": "",
        })

        for _, row in df.iterrows():
            x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
            x0 = int(row["x0"] / 1000 * img_width)
            x1 = int(row["x1"] / 1000 * img_width)

            width = x1 - x0
            height = y1 - y0



            if width <= 0 or height <= 0:
                #print(f"⚠️ Invalid box dimensions for {base_name}: ({x0}, {y0}, {x1}, {y1})")
                continue  # Skip invalid boxes

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": convert_label(row["label"]),
                "bbox": [x0, y0, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1

        img_id += 1

    output = {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 0, "name": "text", "supercategory": ""},
            {"id": 1, "name": "other", "supercategory": ""}
        ],
        "images": images,
        "annotations": annotations,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=4)

    print(f"✅ Done. Saved {len(images)} images and {len(annotations)} annotations to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DocBank .txt files into a single COCO JSON with word-level boxes.")
    parser.add_argument("--src_dir", type=str, required=True, help="Path to folder containing DocBank .txt files")
    parser.add_argument("--image_root", type=str, required=True, help="Path to folder containing corresponding .jpg images")
    parser.add_argument("--split_json", type=str, default=None, help="Optional: COCO-format split JSON (e.g. 500K_train.json)")
    parser.add_argument("--output_json", type=str, required=True, help="Path to output COCO-style JSON file")
    parser.add_argument("--max_images", type=int, default=None, help="Optional: Maximum number of images to include")

    args = parser.parse_args()
    main(args)
