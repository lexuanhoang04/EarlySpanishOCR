import os
import json
from tqdm import tqdm
from PIL import Image


def convert_funsd_to_exact_textocr_format(annotations_dir, images_dir, output_json_path):
    imgs = {}
    anns = {}
    ann_id = 0

    for fname in tqdm(os.listdir(annotations_dir)):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(annotations_dir, fname), "r") as f:
            anno = json.load(f)

        image_id = fname.replace(".json", ".png")
        image_path = os.path.join(images_dir, image_id)

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        # Save image metadata
        # Split path and take last 3 components
        path_components = image_path.split(os.sep)
        file_name = os.path.join(*path_components[-3:]) if len(path_components) >= 3 else image_path

        imgs[image_id] = {
            "file_name": file_name,
            "width": width,
            "height": height,
            "id": image_id
        }

        for entry in anno.get("form", []):
            words = entry.get("words", [])
            if not words:
                continue

            text = " ".join(w["text"] for w in words).strip()
            if not text:
                continue

            x1 = min(w["box"][0] for w in words)
            y1 = min(w["box"][1] for w in words)
            x2 = max(w["box"][2] for w in words)
            y2 = max(w["box"][3] for w in words)

            width_box = x2 - x1
            height_box = y2 - y1
            if len(text) > 80 or len(text.split()) > 12:
                continue  # likely paragraph
            if height_box > width_box * 1.5:
                continue  # vertical
            if width_box < 20:
                continue  # too narrow

            anns[str(ann_id)] = {
                "id": ann_id,
                "image_id": image_id,
                "utf8_string": text,         # ✅ required by TextOCRDataset             # optional / for completeness
                "bbox": [x1, y1, x2, y2]
            }
            ann_id += 1

    output = {
        "imgs": imgs,
        "anns": anns
    }

    with open(output_json_path, "w") as out_f:
        json.dump(output, out_f, indent=2)

    print(f"Saved {len(anns)} annotations across {len(imgs)} images to {output_json_path}")


if __name__ == "__main__":
    convert_funsd_to_exact_textocr_format(
        annotations_dir="dataset/FUNSD/training_data/annotations",
        images_dir="dataset/FUNSD/training_data/images",
        output_json_path="json/FUNSD/funsd_train_textocr_style_filtered.json"
    )

    convert_funsd_to_exact_textocr_format(
        annotations_dir="dataset/FUNSD/testing_data/annotations",
        images_dir="dataset/FUNSD/testing_data/images",
        output_json_path="json/FUNSD/funsd_val_textocr_style_filtered.json"
    )
