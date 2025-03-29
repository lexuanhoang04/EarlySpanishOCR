import argparse
import sys, os

from PIL import Image
import torch
from torch import nn
from torchvision import transforms
os.sys.path.insert(0, os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
from models import CRNN, TransformerOCR, CRNN_ResNet18
import torch.nn.functional as F

# ---------------- Configuration ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT = 32
IMG_WIDTH = 128
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
CHAR2IDX[""] = 0  # CTC blank
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
NUM_CLASSES = len(CHAR2IDX)

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# ---------------- Greedy Decoder ----------------
def greedy_decoder(preds):
    preds = preds.argmax(2)  # (T, B)
    decoded_texts = []
    for seq in preds.permute(1, 0):  # (B, T)
        prev = 0
        s = ""
        for p in seq:
            p = p.item()
            if p != prev and p != 0:
                s += IDX2CHAR.get(p, "")
            prev = p
        decoded_texts.append(s)
    return decoded_texts

# ---------------- Main Test Script ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--gt_text", type=str, required=True, help="Ground truth text label")
    parser.add_argument("--model", type=str, choices=["crnn", "transformer", "crnn_resnet18"], default="crnn")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    label_indices = [CHAR2IDX[c] for c in args.gt_text]
    label = torch.LongTensor(label_indices)

    target_len = torch.tensor([len(label)], dtype=torch.long)

    image = Image.open(args.img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    print("Image tensor shape:", image_tensor.shape)
    label = label.to(DEVICE)

    if args.model == "crnn":
        model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    elif args.model == "transformer":
        model = TransformerOCR(num_classes=NUM_CLASSES, d_model=2048, nhead=8, num_layers=3).to(DEVICE)
    elif args.model == "crnn_resnet18":
        model = CRNN_ResNet18(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("start training")
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits = model(image_tensor)  # (B=1, T, C)
        log_probs = logits

        batch_size = image_tensor.size(0)
        input_len = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long, device=DEVICE)
        target_len = torch.full(size=(batch_size,), fill_value=label.size(0), dtype=torch.long, device=DEVICE)
    
        # print("log_probs.shape:", log_probs.shape)  # should be (T=32, B=1, C)
        # print("label.shape:", label.shape)          # should be (4,)
        # print("input_len:", input_len)              # should be tensor([32])
        # print("target_len:", target_len)            # should be tensor([4])

        loss = criterion(log_probs, label, input_len, target_len)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            decoded = greedy_decoder(log_probs)
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Pred={decoded[0]}, GT={args.gt_text}")
        if decoded[0] == args.gt_text:
            print("Model overfitted!")
            break
        model.train()

if __name__ == "__main__":
    main()
