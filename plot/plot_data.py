import os
import json
import argparse
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from collections import defaultdict

def visualize_gt(json_path, image_root, output_dir, max_imgs=5):
    os.makedirs(output_dir, exist_ok=True)
    coco = COCO(json_path)
    img_ids = coco.getImgIds()

    for i, img_id in enumerate(img_ids[:max_imgs]):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_root, os.path.basename(img_info['file_name']))
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            x, y, w, h = ann['bbox']
            draw.rectangle([x, y, x + w, y + h], outline='red', width=2)

        image.save(os.path.join(output_dir, f"gt_{i}_{os.path.basename(img_path)}"))

def visualize_eval(pred_json_path, gt_json_path, image_root, output_dir, max_imgs=5):
    os.makedirs(output_dir, exist_ok=True)

    # Load GT for file_name lookup
    coco_gt = COCO(gt_json_path)

    with open(pred_json_path, 'r') as f:
        preds = json.load(f)
        print(f'loaded {pred_json_path}')

    pred_dict = defaultdict(list)
    for ann in preds:
        pred_dict[ann['image_id']].append(ann)

    for i, (img_id, anns) in enumerate(list(pred_dict.items())[:max_imgs]):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info['file_name'].split('/')[-1]

        img_path = os.path.join(image_root, os.path.basename(file_name))
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for ann in anns:
            x, y, w, h = ann['bbox']
            draw.rectangle([x, y, x + w, y + h], outline='blue', width=2)

        image.save(os.path.join(output_dir, f"eval_{i}_{os.path.basename(img_path)}"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to COCO JSON (GT or prediction)")
    parser.add_argument("--gt_json", help="(Only for eval mode) Path to GT COCO JSON")
    parser.add_argument("--image_root", required=True, help="Root directory of images")
    parser.add_argument("--output_dir", required=True, help="Directory to save visualized outputs")
    parser.add_argument("--mode", choices=["gt", "eval"], required=True, help="Type of JSON format")
    parser.add_argument("--max_imgs", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "gt":
        visualize_gt(args.json, args.image_root, args.output_dir, args.max_imgs)
    else:
        if not args.gt_json:
            raise ValueError("In eval mode, --gt_json must be provided to get file names.")
        visualize_eval(args.json, args.gt_json, args.image_root, args.output_dir, args.max_imgs)

if __name__ == "__main__":
    main()
