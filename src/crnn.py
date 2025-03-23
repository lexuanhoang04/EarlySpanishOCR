import json
import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
import random
from tqdm import tqdm
import editdistance

# Function to save the model
def save_model(model, path="checkpoints/ocr_crnn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# ------------------ Config ------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
IMG_HEIGHT = 32
IMG_WIDTH = 128
NUM_WORKERS = 4
MAX_LABEL_LEN = 32
EPOCHS = 10
SAVE_PATH = 'checkpoints/full_textocr_crnn.pth'

# Characters supported (used for label encoding)
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # leave 0 for blank (CTC)
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}

# ------------------ Dataset ------------------
class TextOCRDataset(Dataset):
    def __init__(self, json_path, image_root, max_samples=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.imgs = self.data['imgs']
        self.anns = list(self.data['anns'].values())

        self.samples = []
        for ann in self.anns:
            text = ann['utf8_string']
            if text != '.' and all(c in CHAR2IDX for c in text):
                self.samples.append((ann['image_id'], ann['bbox'], text))
            if max_samples and len(self.samples) >= max_samples:
                break

        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, bbox, text = self.samples[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.image_root, os.path.basename(img_info['file_name']))
        image = Image.open(img_path).convert("RGB")

        x, y, w, h = bbox
        cropped = image.crop((x, y, x + w, y + h))
        cropped = self.transform(cropped)

        label = torch.LongTensor([CHAR2IDX[c] for c in text])
        return cropped, label, len(label), text  # include GT text for eval

# ------------------ Collate ------------------
def collate_fn(batch):
    images, labels, lengths, texts = zip(*batch)
    images = torch.stack(images)
    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor(lengths)
    return images, labels_concat, label_lengths, texts

# ------------------ CRNN ------------------
class CRNN(nn.Module):
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
        self.rnn = nn.LSTM(128 * 8, 256, bidirectional=True, batch_first=True, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # B x W x C x H
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)

# ------------------ Decode ------------------
def decode(preds):
    preds = preds.argmax(2)
    preds = preds.permute(1, 0)
    texts = []
    for pred in preds:
        text = ''
        prev = 0
        for p in pred:
            p = p.item()
            if p != prev and p != 0:
                text += IDX2CHAR[p]
            prev = p
        texts.append(text)
    return texts

# ------------------ Train ------------------

def train(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, labels, label_lengths, _ in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)

            logits = model(images)  # [B, W, C]
            log_probs = logits.permute(1, 0, 2)
            input_lengths = torch.full((logits.size(0),), log_probs.size(0), dtype=torch.long).to(DEVICE)

            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Optional: update tqdm with current loss
            progress_bar.set_postfix(loss=loss.item())

        print(f"✅ Epoch {epoch+1}/{EPOCHS} - Avg Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾 Model saved to {SAVE_PATH}")

# ------------------ Evaluate ------------------

def evaluate(model, test_loader):
    model.eval()
    total_cer = 0
    total_chars = 0

    with torch.no_grad():
        for images, _, _, gt_texts in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = images.to(DEVICE)
            preds = model(images)
            decoded = decode(preds)

            for pred, target in zip(decoded, gt_texts):
                cer = editdistance.eval(pred, target)
                total_cer += cer
                total_chars += len(target)

                # Optional: print sample predictions
                # print(f"GT: {target}\nPR: {pred}\n---")

    print(f"\n📊 CER: {total_cer / total_chars:.4f}")

# ------------------ Main ------------------
def main():
    train_data = TextOCRDataset(
        json_path='dataset/TextOCR/TextOCR_0.1_train.json',
        image_root='dataset/TextOCR/train_val_images/train_images',
        max_samples=1000000000 # Use full train set
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    test_data = TextOCRDataset(
        json_path='dataset/TextOCR/TextOCR_0.1_val.json',
        image_root='dataset/TextOCR/train_val_images/train_images'
    )
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = CRNN(num_classes=len(CHAR2IDX) + 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    train(model, train_loader, criterion, optimizer)
    evaluate(model, test_loader)

if __name__ == '__main__':
    main()