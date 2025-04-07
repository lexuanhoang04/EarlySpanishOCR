import argparse
import json
from pathlib import Path
from PIL import Image

def crop_and_save(image_path, bbox, output_path):
    image = Image.open(image_path).convert("RGB")
    x, y, w, h = map(int, bbox)
    cropped = image.crop((x, y, x + w, y + h))
    cropped.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Find and crop word from TextOCR-style JSON.")
    parser.add_argument("--json_path", type=Path, help="Path to TextOCR JSON file")
    parser.add_argument("--images_dir", type=Path, help="Directory containing source images")
    parser.add_argument("--output_dir", type=Path, help="Directory to save cropped word images")
    parser.add_argument("--target_word", type=str, help="Word to find (case sensitive)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Map image_id → file_name
    id_to_filename = {img["id"]: img["file_name"] for img in data["imgs"]}

    count = 0
    for ann in data["anns"]:
        if ann["utf8_string"] == args.target_word:
            image_id = ann["image_id"]
            file_name = id_to_filename.get(image_id)
            if not file_name:
                print(f"Skipping annotation with unknown image_id {image_id}")
                continue
            image_path = args.images_dir / file_name
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue
            output_path = args.output_dir / f"{args.target_word}_{count}.png"
            crop_and_save(image_path, ann["bbox"], output_path)
            count += 1

    print(f"Found and saved {count} occurrences of '{args.target_word}' to {args.output_dir}")

if __name__ == "__main__":
    main()
