import json
import argparse
from pathlib import Path

def limit_textocr_json(input_path, output_path, num_images):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Step 1: Get the first N image IDs
    all_image_ids = list(data['imgs'].keys())
    selected_image_ids = set(all_image_ids[:num_images])

    # Step 2: Filter imgs
    limited_imgs = {img_id: data['imgs'][img_id] for img_id in selected_image_ids}

    # Step 3: Filter img2Anns
    limited_img2anns = {
        img_id: data['imgToAnns'][img_id]
        for img_id in selected_image_ids if img_id in data['imgToAnns']
    }

    # Step 4: Filter anns
    selected_ann_ids = set(ann_id for ann_list in limited_img2anns.values() for ann_id in ann_list)
    limited_anns = {
        ann_id: data['anns'][ann_id]
        for ann_id in selected_ann_ids
    }

    # Step 5: Assemble and write output
    limited_data = {
        "imgs": limited_imgs,
        "anns": limited_anns,
        "imgToAnns": limited_img2anns
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(limited_data, f, indent=2)
    print(f"Saved {len(limited_imgs)} images and {len(limited_anns)} annotations to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limit number of images in a TextOCR-format JSON file.")
    parser.add_argument("--input_json", type=str, help="Path to input JSON file")
    parser.add_argument("--output_json", type=str, help="Path to output limited JSON file")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to keep")

    args = parser.parse_args()
    limit_textocr_json(args.input_json, args.output_json, args.num_images)
