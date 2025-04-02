# convert_to_coco.py
import json
import argparse
import datetime

def convert_textocr_to_coco(json_path, output_path, max_samples=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    coco = {
        "info": {
            "description": "TextOCR to COCO Format",
            "version": "1.0",
            "year": datetime.datetime.now().year
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text"}]
    }

    imgs = data['imgs']
    anns = list(data['anns'].values())
    img_ids_seen = set()
    ann_id = 1
    img_index = 0

    for ann in anns:
        if max_samples and len(coco["annotations"]) >= max_samples:
            break
        img_id = ann['image_id']
        bbox = ann['bbox']
        if img_id not in img_ids_seen:
            img_info = imgs[img_id]
            coco['images'].append({
                "id": img_index,
                "width": int(img_info['width']),
                "height": int(img_info['height']),
                "file_name": img_info['file_name']
            })
            img_ids_seen.add(img_id)
            img_index += 1

        coco['annotations'].append({
            "id": ann_id,
            "image_id": img_index - 1,
            "bbox": bbox,
            "category_id": 1,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        ann_id += 1

    with open(output_path, 'w') as f:
        json.dump(coco, f)
    print(f"COCO-style annotation saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', required=True, help='Path to TextOCR json')
    parser.add_argument('--output_path', required=True, help='Where to save converted json')
    parser.add_argument('--max_samples', type=int, default=10000000000, help='Limit number of annotations')
    args = parser.parse_args()

    
    convert_textocr_to_coco(args.json_path, args.output_path, args.max_samples)
