import os
import json
import argparse
from PIL import Image

def convert_funsd_to_coco(funsd_dir, output_json, max_images=None):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text"}]
    }

    ann_id = 1
    image_id = 1

    ann_dir = os.path.join(funsd_dir, "annotations")
    img_dir = os.path.join(funsd_dir, "images")

    ann_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
    ann_files.sort()  # optional, for deterministic behavior
    if max_images is not None:
        ann_files = ann_files[:max_images]

    for ann_file in ann_files:
        with open(os.path.join(ann_dir, ann_file)) as f:
            data = json.load(f)

        img_name = ann_file.replace(".json", ".png")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for annotation {ann_file}")
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": os.path.join(funsd_dir.split('/')[-1], "images", img_name),
            "width": width,
            "height": height
        })

        for form in data["form"]:
            x1, y1, x2, y2 = form["box"]
            bbox = [x1, y1, x2 - x1, y2 - y1]

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "iscrowd": 0,
                "area": bbox[2] * bbox[3]
            })
            ann_id += 1

        image_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"✅ COCO annotations saved to {output_json} ({len(coco['images'])} images used)")

def main():
    parser = argparse.ArgumentParser(description="Convert FUNSD format to COCO format")
    parser.add_argument("--funsd_dir", type=str, help="Path to FUNSD data (should contain annotations/ and images/)")
    parser.add_argument("--output_json", type=str, help="Path to output COCO-format JSON")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to convert (for quick testing)")

    args = parser.parse_args()
    convert_funsd_to_coco(args.funsd_dir, args.output_json, args.max_images)

if __name__ == "__main__":
    main()
