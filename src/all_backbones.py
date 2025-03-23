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
import argparse
import yaml


# ------------------ Load Config ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config YAML file")
parser.add_argument("--json_train", default=None)
parser.add_argument("--json_val", default=None)
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Optionally restrict GPU usage
if "CUDA_VISIBLE_DEVICES" in cfg:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["CUDA_VISIBLE_DEVICES"]

# ------------------ Config ------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = cfg["batch_size"]
IMG_HEIGHT = cfg["img_height"]
IMG_WIDTH = cfg["img_width"]
NUM_WORKERS = cfg["num_workers"]
MAX_LABEL_LEN = 32
EPOCHS = cfg["epochs"]
SAVE_PATH = f"checkpoints/{cfg['name']}_crnn.pth"
OUTPUT_PATH = f"output/{cfg['name']}_cer.txt"

# Characters supported (used for label encoding)
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # leave 0 for blank (CTC)
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}

print("Visible GPUs:", torch.cuda.device_count())

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
        return cropped, label, len(label), text

# ------------------ Collate ------------------
def collate_fn(batch):
    images, labels, lengths, texts = zip(*batch)
    images = torch.stack(images)
    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor(lengths)
    return images, labels_concat, label_lengths, texts

# ------------------ Model ------------------
class CRNN_ResNet18(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        base = models.resnet18(pretrained=True)
        modules = list(base.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("layer4") and not name.startswith("layer3") and not name.startswith("bn1"):
                    param.requires_grad = False

        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)

# ------------------ Decode ------------------
def decode(preds):
    preds = preds.argmax(2).permute(1, 0)
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

def decode_better(preds):
    preds = preds.argmax(2).permute(1, 0)
    texts = []
    for pred in preds:
        text = []
        last = -1
        for p in pred:
            p = p.item()
            if p != 0 and p != last:
                text.append(IDX2CHAR[p])
            last = p
        texts.append("".join(text))
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

            logits = model(images)
            log_probs = logits.permute(1, 0, 2)
            input_lengths = torch.full((logits.size(0),), log_probs.size(0), dtype=torch.long).to(DEVICE)

            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        print(f"✅ Epoch {epoch+1}/{EPOCHS} - Avg Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾 Model saved to {SAVE_PATH}")

# ------------------ Evaluate ------------------
def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        for decoder_fn in [decode, decode_better]:
            total_cer = 0
            total_chars = 0
            for images, _, _, gt_texts in tqdm(test_loader, desc=f"Evaluating {decoder_fn.__name__}", unit="batch"):
                images = images.to(DEVICE)
                preds = model(images)
                decoded = decoder_fn(preds)
                for pred, target in zip(decoded, gt_texts):
                    cer = editdistance.eval(pred, target)
                    total_cer += cer
                    total_chars += len(target)
            cer_value = total_cer / total_chars
            print(f"📊 CER ({decoder_fn.__name__}): {cer_value:.4f}")
            with open(OUTPUT_PATH, "a") as f:
                f.write(f"{SAVE_PATH}, {decoder_fn.__name__}, CER: {cer_value:.4f}\n")

# ------------------ Main ------------------
def main():


    if args.json_train is not None:
        train_data = TextOCRDataset(
            json_path=args.json_train,
            image_root=cfg["image_root"],
            max_samples=cfg.get("max_samples")
        )
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    if args.json_val is not None:
        test_data = TextOCRDataset(
            json_path=args.json_val,
            image_root=cfg["image_root"]
        )
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = nn.DataParallel(CRNN_ResNet18(num_classes=len(CHAR2IDX) + 1)).to(DEVICE)

    if args.json_train is not None:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        train(model, train_loader, criterion, optimizer)

    if args.json_val is not None:
        if args.json_train is None:
            model.load_state_dict(torch.load(SAVE_PATH))
        evaluate(model, test_loader)

if __name__ == '__main__':
    main()
