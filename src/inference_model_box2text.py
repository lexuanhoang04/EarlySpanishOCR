from PIL import Image
import torch
from torchvision import transforms
import sys
import os
import argparse

# Add current working directory to import models
sys.path.insert(0, os.getcwd())
from models import CRNN_ResNet18  # or CRNN / TransformerOCR depending on your model

# -------------------- Global Configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
from src.box_to_text import greedy_decoder, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_WORKERS, EPOCHS, LEARNING_RATE, MAX_LABEL_LEN, NUM_CLASSES



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to the image to be predicted")
    args = parser.parse_args()

    # Load the model
    model = CRNN_ResNet18(NUM_CLASSES).to(DEVICE)
    state_dict = torch.load("checkpoints/crnn_resnet18_full.pth", map_location=DEVICE)

    # Remove "module." prefix if trained with DDP
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Load the image (replace with your actual cropped image path)
    img_path = args.image_path
    image = Image.open(img_path).convert("RGB")

    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Run prediction
    with torch.no_grad():
        logits = model(image_tensor)  # [T, B, C]
        print("logits.shape:", logits.shape)
        decoded = greedy_decoder(logits.permute(1, 0, 2))  # [B, T, C]

    print("Prediction:", decoded[0])


if __name__ == "__main__":
    main()
