# visualize_coco_boxes.py
import json
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

def visualize_boxes(coco_json, image_root, output_dir=None, num_images=5):
    with open(coco_json, 'r') as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    images = coco['images']
    anns = coco['annotations']
    img_to_anns = {}
    for ann in anns:
        img_id = ann['image_id']
        img_to_anns.setdefault(img_id, []).append(ann)

    for img in images[:num_images]:
        img_path = os.path.join(image_root, os.path.basename(img['file_name']))
        image = Image.open(img_path).convert("RGB")

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for ann in img_to_anns[img['id']]:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        if output_dir:
            save_path = os.path.join(output_dir, f"{os.path.basename(img['file_name'])}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved to {save_path}")
        else:
            plt.title(img['file_name'])
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_json', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--num_images', type=int, default=5)
    args = parser.parse_args()

    visualize_boxes(args.coco_json, args.image_root, args.output_dir, args.num_images)
