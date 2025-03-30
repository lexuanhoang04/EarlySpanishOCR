import os, sys
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.getcwd())  # Add current working directory to import models
from src.text_detection_FasterRCNN import get_text_detector  # adjust import path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference pipeline for a single image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    return transform(image), image  # return both tensor and PIL image

def run_inference(model, image_path, output_image_path):
    model.eval()
    image_tensor, image_pil = load_image(image_path)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    draw = ImageDraw.Draw(image_pil)
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score >= 0.5:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{score:.2f}", fill="red")

    image_pil.save(output_image_path)
    print(f"Saved visualization to {output_image_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_image', type=str, default='inference_viz.jpg')
    args = parser.parse_args()

    model = get_text_detector(num_classes=2)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE)

    run_inference(model, args.image_path, args.output_image)
