import json
import argparse
from pathlib import Path
from tqdm import tqdm

def load_json(path):
    with open(path, "r") as f:
        print(f"Loading JSON from {path}")
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        print(f"Saving JSON to {path}")
        json.dump(obj, f, indent=4)

def build_image_map(images):
    return {img["id"]: img for img in images}

def build_ann_map(annotations):
    anns_by_image = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return anns_by_image

def convert_categories(cat_name):
    return 0 if cat_name.lower() in {"abstract", "paragraph", "title", "author"} else 1

def main(args):
    input_json = load_json(args.input_json)
    image_root = Path(args.image_root)

    image_map = build_image_map(input_json["images"])
    ann_map = build_ann_map(input_json["annotations"])

    output = {
        "info": input_json.get("info", {}),
        "licenses": input_json.get("licenses", []),
        "categories": [
            {"id": 0, "name": "text", "supercategory": ""},
            {"id": 1, "name": "other", "supercategory": ""}
        ],
        "images": [],
        "annotations": []
    }

    # Filter categories into new ID map
    cat_id_to_name = {c["id"]: c["name"] for c in input_json["categories"]}

    img_count = 0
    ann_id = 0

    for image in tqdm(input_json["images"], desc="Filtering images"):
        img_path = image_root / image["file_name"]
        if not img_path.is_file():
            continue

        if args.max_images is not None and img_count >= args.max_images:
            break

        new_img_id = img_count  # Renumber image IDs
        old_img_id = image["id"]

        image["id"] = new_img_id
        output["images"].append(image)
        img_count += 1

        if old_img_id in ann_map:
            for ann in ann_map[old_img_id]:
                old_cat_id = ann["category_id"]
                cat_name = cat_id_to_name[old_cat_id]
                new_cat_id = convert_categories(cat_name)

                new_ann = ann.copy()
                new_ann["id"] = ann_id
                new_ann["image_id"] = new_img_id
                new_ann["category_id"] = new_cat_id
                output["annotations"].append(new_ann)
                ann_id += 1

    save_json(output, args.output_json)
    print(f"✅ Done! {img_count} valid images written to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DocBank COCO JSON into 2-category filtered COCO format.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to original DocBank COCO JSON (e.g. 500K_train.json)")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the filtered COCO JSON")
    parser.add_argument("--image_root", type=str, required=True, help="Path to image root folder (e.g. docbank_500K)")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to keep")

    args = parser.parse_args()
    main(args)
