import os
import argparse
import json
from lxml import etree
from PIL import Image

def extract_line_boxes_from_pagexml(xml_path):
    ns = {'ns': 'https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    tree = etree.parse(xml_path)

    boxes = []
    for textline in tree.xpath('//ns:TextLine', namespaces=ns):
        coords = textline.find('.//ns:Coords', namespaces=ns)
        if coords is not None:
            points = coords.attrib.get('points', '').strip()
            if not points:
                continue  # Skip if empty

            try:
                coords_list = [tuple(map(int, pt.split(','))) for pt in points.split()]
            except Exception as e:
                print(f"⚠️ Skipping malformed points in {xml_path}: {points}")
                continue

            if not coords_list:
                continue  # Skip if still empty

            xs = [x for x, y in coords_list]
            ys = [y for x, y in coords_list]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            boxes.append([x0, y0, x1, y1])
    return boxes


def cbad_to_coco(input_dir, output_json, limit=None):
    xml_dir = os.path.join(input_dir, "page")
    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith(".xml")])

    if limit:
        xml_files = xml_files[:limit]

    images = []
    annotations = []
    annotation_id = 1
    category_id = 1
    image_id = 1

    for xml_filename in xml_files:
        xml_path = os.path.join(xml_dir, xml_filename)
        img_filename = os.path.splitext(xml_filename)[0] + ".jpg"
        img_path = os.path.join(input_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"⚠️ Skipping missing image: {img_filename}")
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        boxes = extract_line_boxes_from_pagexml(xml_path)

        split_prefix = os.path.basename(os.path.normpath(input_dir))  # e.g., "train" or "eval"

        images.append({
            "id": image_id,
            "file_name": f"{split_prefix}/{img_filename}",
            "width": width,
            "height": height
        })

        for box in boxes:
            x0, y0, x1, y1 = box
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x0, y0, x1 - x0, y1 - y0],  # xywh for COCO
                "iscrowd": 0,
                "area": (x1 - x0) * (y1 - y0)
            })
            annotation_id += 1

        image_id += 1

    if limit is not None:
        assert len(images) == limit, f"Expected {limit} images, but got {len(images)}. Check for missing image or XML files."

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": category_id,
                "name": "text",
                "supercategory": "text"
            }
        ]
    }

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ COCO annotations saved to {output_json} with {len(images)} images and {len(annotations)} annotations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert cBAD (PAGE XML) to COCO format with xyxy boxes.")
    parser.add_argument("--input_dir", required=True, help="Path to cBAD split folder (e.g., train/, test/)")
    parser.add_argument("--output", required=True, help="Output path for COCO JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")

    args = parser.parse_args()
    cbad_to_coco(args.input_dir, args.output, args.limit)
