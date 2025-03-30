import os
import json
import argparse
from PIL import Image
from collections import defaultdict


def crop_textocr(json_path, image_root, output_dir, max_imgs=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    imgs = data["imgs"]
    anns = list(data["anns"].values())

    image_to_anns = defaultdict(list)
    for ann in anns:
        if "utf8_string" not in ann or "bbox" not in ann:
            continue
        image_to_anns[ann["image_id"]].append(ann)

    used = 0
    for img_id, ann_list in image_to_anns.items():
        if max_imgs is not None and used >= max_imgs:
            break

        img_info = imgs[str(img_id)] if str(img_id) in imgs else imgs[img_id]
        img_path = os.path.join(image_root, img_info["file_name"].split("/")[-1])
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        folder_name = os.path.splitext(os.path.basename(img_info["file_name"]))[0]
        image_output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(image_output_dir, exist_ok=True)

        for i, ann in enumerate(ann_list):
            x, y, w, h = ann["bbox"]
            cropped = image.crop((x, y, x + w, y + h))
            text = ann["utf8_string"].strip().replace(" ", "_") or f"box{i}"
            save_name = f"{i:03d}_{text}.png"
            cropped.save(os.path.join(image_output_dir, save_name))

        print(f"Cropped {len(ann_list)} boxes → {image_output_dir}")
        used += 1

def crop_single_image(json_path, image_root, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    imgs = data["imgs"]
    anns = list(data["anns"].values())

    # Find image by file_name match
    img_id = None
    for k, v in imgs.items():
        if v["file_name"].split("/")[-1] == image_name:
            img_id = k
            img_info = v
            break

    if img_id is None:
        print(f"Image name '{image_name}' not found in JSON.")
        return

    img_path = os.path.join(image_root, image_name)
    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    image_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])
    os.makedirs(image_output_dir, exist_ok=True)

    count = 0
    for ann in anns:
        if str(ann["image_id"]) != str(img_id):
            continue
        if "utf8_string" not in ann or "bbox" not in ann:
            continue

        x, y, w, h = ann["bbox"]
        cropped = image.crop((x, y, x + w, y + h))
        text = ann["utf8_string"].strip().replace(" ", "_") or f"box{count}"
        save_name = f"{count:03d}_{text}.png"
        cropped.save(os.path.join(image_output_dir, save_name))
        count += 1

    print(f"Cropped {count} boxes from '{image_name}' → {image_output_dir}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to TextOCR format JSON")
    parser.add_argument("--image_root", required=True, help="Path to full images")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--max_imgs", type=int, default=None)
    parser.add_argument("--image_name", type=str, help="If provided, only crop boxes for this image")

    args = parser.parse_args()

    if args.image_name:
        crop_single_image(args.json, args.image_root, args.output_dir, args.image_name)
    else:
        crop_textocr(args.json, args.image_root, args.output_dir, args.max_imgs)



if __name__ == "__main__":
    main()
