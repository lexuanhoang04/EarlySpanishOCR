# This script uses a well-known CTC-based OCR model (Rosetta-style CNN+LSTM)
# and plugs in our custom encoder (CHAR2IDX) and decoder (CTC decoding logic)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import editdistance

# ------------------ Constants ------------------
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for blank
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------ Dataset ------------------
class DummyTextDataset(Dataset):
    def __init__(self, json_path, image_root, max_samples=100):
        with open(json_path) as f:
            data = json.load(f)

        self.imgs = data['imgs']
        self.anns = list(data['anns'].values())
        self.samples = []

        for ann in self.anns:
            text = ann['utf8_string']
            if text != '.' and all(c in CHAR2IDX for c in text):
                self.samples.append((ann['image_id'], ann['bbox'], text))
            if len(self.samples) >= max_samples:
                break

        self.root = image_root
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        img_id, bbox, text = self.samples[idx]
        info = self.imgs[img_id]
        img_path = os.path.join(self.root, os.path.basename(info['file_name']))
        image = Image.open(img_path).convert("RGB")

        x, y, w, h = bbox
        cropped = image.crop((x, y, x + w, y + h))
        cropped = self.transform(cropped)

        label = torch.LongTensor([CHAR2IDX[c] for c in text])
        return cropped, label, text

    def __len__(self):
        return len(self.samples)

# ------------------ Model ------------------
class SimpleCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.rnn = nn.LSTM(128 * 8, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # B x W x C x H
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)

# ------------------ Decoding ------------------
def decode(preds):
    preds = preds.argmax(2)
    preds = preds.permute(1, 0)
    results = []
    for pred in preds:
        text = ''
        prev = 0
        for p in pred:
            p = p.item()
            if p != 0 and p != prev:
                text += IDX2CHAR[p]
            prev = p
        results.append(text)
    return results

# ------------------ Evaluation ------------------
def evaluate(model, loader):
    model.eval()
    total_cer, total_chars = 0, 0
    with torch.no_grad():
        for images, _, texts in tqdm(loader, desc="Evaluating"):
            images = images.to(DEVICE)
            preds = model(images)
            decoded = decode(preds)
            for gt, pr in zip(texts, decoded):
                total_cer += editdistance.eval(gt, pr)
                total_chars += len(gt)
                print(f"GT: {gt} | PR: {pr}")
    print(f"\nCER: {total_cer / total_chars:.4f}")

# ------------------ Main ------------------
def main():
    dataset = DummyTextDataset(
        json_path='dataset/TextOCR/TextOCR_0.1_val.json',
        image_root='dataset/TextOCR/train_val_images/train_images',
        max_samples=100
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = SimpleCRNN(num_classes=len(CHAR2IDX) + 1).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/ocr_crnn.pth", map_location=DEVICE))
    evaluate(model, loader)

if __name__ == '__main__':
    main()
