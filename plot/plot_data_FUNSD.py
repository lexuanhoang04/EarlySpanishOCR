import os
import json
from PIL import Image
from tqdm import tqdm
import random


def save_overfit_samples(textocr_json_path, image_root, output_dir, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)

    with open(textocr_json_path, "r") as f:
        data = json.load(f)

    imgs = data["imgs"]
    anns = list(data["anns"].values())

    # Filter out long lines (paragraphs)
    filtered = [
        ann for ann in anns
        if len(ann["utf8_string"].split()) <= 12 and len(ann["utf8_string"]) <= 80
    ]

    # Randomly select samples
    samples = random.sample(filtered, min(num_samples, len(filtered)))
    saved_entries = []

    for i, ann in enumerate(samples):
        image_id = ann["image_id"]
        box = ann["bbox"]
        text = ann["utf8_string"]

        img_filename = imgs[image_id]["file_name"]
        img_path = os.path.join(image_root, img_filename)
        img = Image.open(img_path).convert("RGB")

        x1, y1, x2, y2 = box
        crop = img.crop((x1, y1, x2, y2))

        crop_filename = f"overfit_sample_{i}.png"
        crop_path = os.path.join(output_dir, crop_filename)
        crop.save(crop_path)

        # Save ground truth text
        txt_path = os.path.join(output_dir, f"overfit_sample_{i}.txt")
        with open(txt_path, "w") as ftxt:
            ftxt.write(text)

        saved_entries.append({
            "filename": crop_filename,
            "text": text
        })

    # Save summary
    with open(os.path.join(output_dir, "samples.json"), "w") as f:
        json.dump(saved_entries, f, indent=2)

    print(f"✅ Saved {len(saved_entries)} cropped samples to {output_dir}")


if __name__ == "__main__":
    save_overfit_samples(
        textocr_json_path="json/FUNSD/funsd_val_textocr_style_filtered.json",
        image_root="dataset/FUNSD",
        output_dir="vis/FUNSD/overfit_samples_textocr",
        num_samples=5
    )
