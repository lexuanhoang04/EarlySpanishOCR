import json
import argparse
from pathlib import Path
import os

def parse_dbnet_polygons(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    polygons = []
    for line in lines:
        parts = list(map(float, line.strip().split(',')))
        if len(parts) < 9:
            continue
        coords = parts[:-1]
        polygons.append(coords)
    return polygons

def polygon_to_bbox(coords):
    xs = coords[::2]
    ys = coords[1::2]
    xyxy = [min(xs), min(ys), max(xs), max(ys)]
    xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
    return xywh

def convert_to_textocr_format(image_id, image_filename, width, height, polygons, image_set="test"):
    anns = {}
    img2Anns = {}
    for i, coords in enumerate(polygons):
        ann_id = f"{image_id}_{i+1}"
        bbox = polygon_to_bbox(coords)
        anns[ann_id] = {
            "id": ann_id,
            "image_id": image_id,
            "bbox": bbox,
            "points": coords,
            "utf8_string": "",
            "area": bbox[2] * bbox[3]
        }
        img2Anns.setdefault(image_id, []).append(ann_id)

    return {
        "imgs": {
            image_id: {
                "id": image_id,
                "width": width,
                "height": height,
                "set": image_set,
                "file_name": image_filename
            }
        },
        "anns": anns,
        "imgToAnns": img2Anns
    }

def main():
    parser = argparse.ArgumentParser(description="Batch convert DBNet outputs to TextOCR format.")
    parser.add_argument("--txt_dir", required=True, help="Directory of DBNet polygon .txt files")
    parser.add_argument("--image_dir", required=True, help="Directory of corresponding images")
    parser.add_argument("--output_dir", required=True, help="Directory to save JSONs")
    parser.add_argument("--image_set", default="test", help="Dataset split: train|val|test")
    parser.add_argument("--merge_into_one", action='store_true', help="If set, merge all into one JSON file")
    parser.add_argument("--height", type=int, default=720, help="Height of the images")
    parser.add_argument("--width", type=int, default=1280, help="Width of the images")

    args = parser.parse_args()

    txt_dir = Path(args.txt_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = {
        "imgs": {},
        "anns": {},
        "imgToAnns": {}
    }

    for txt_file in sorted(txt_dir.glob("*.txt")):
        image_id = txt_file.stem.replace("res_", "")
        image_path = image_dir / (image_id + ".jpg")  # you can support .png too
        if not image_path.exists():
            image_path = image_dir / (image_id + ".png")
        if not image_path.exists():
            print(f"Image not found for {txt_file.name}")
            continue

        height = args.height
        width = args.width
        polygons = parse_dbnet_polygons(txt_file)
        json_data = convert_to_textocr_format(
            image_id=image_id,
            image_filename=image_id + ".jpg",  # or .png
            width=width,
            height=height,
            polygons=polygons,
            image_set=args.image_set
        )

        if args.merge_into_one:
            merged["imgs"].update(json_data["imgs"])
            merged["anns"].update(json_data["anns"])
            merged["imgToAnns"].update(json_data["imgToAnns"])
        else:
            out_file = output_dir / f"{image_id}.json"
            with open(out_file, "w") as f:
                json.dump(json_data, f, indent=2)
            print(f"✅ Saved {out_file}")

    if args.merge_into_one:
        merged_path = output_dir / "merged_textocr.json"
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"✅ Merged JSON saved to {merged_path}")

if __name__ == "__main__":
    main()
