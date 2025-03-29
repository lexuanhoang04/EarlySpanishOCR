import json
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, sys
import editdistance
import random
import torchvision.transforms.functional as TF

sys.path.insert(0, os.getcwd())
from models import TransformerOCR, CRNN

# -------------------- Global Configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
IMG_HEIGHT = 32
IMG_WIDTH = 128
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 1e-3
MAX_LABEL_LEN = 32

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR2IDX = {c: i+1 for i, c in enumerate(CHARS)}
CHAR2IDX[""] = 0
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
NUM_CLASSES = len(CHAR2IDX)

# -------------------- Dataset --------------------
class TextOCRDataset(Dataset):
    def __init__(self, json_path, image_root, max_images=None, allowed_image_ids=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_root = image_root
        self.imgs = self.data["imgs"]
        self.anns = list(self.data["anns"].values())

        used_ids = set()
        self.samples = []
        for ann in self.anns:
            text = ann.get("utf8_string", "")
            img_id = ann["image_id"]
            if allowed_image_ids and img_id not in allowed_image_ids:
                continue
            if max_images and img_id not in used_ids and len(used_ids) >= max_images:
                continue
            if text != "." and text and all(c in CHAR2IDX for c in text):
                self.samples.append((img_id, ann["bbox"], text))
                used_ids.add(img_id)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, bbox, text = self.samples[idx]
        img_info = self.imgs[str(img_id)] if str(img_id) in self.imgs else self.imgs[img_id]
        img_path = os.path.join(self.image_root, os.path.basename(img_info["file_name"]))
        image = Image.open(img_path).convert("RGB")
        x, y, w, h = bbox
        cropped = image.crop((x, y, x + w, y + h))
        cropped = self.transform(cropped)
        label_indices = [CHAR2IDX[c] for c in text]
        label = torch.LongTensor(label_indices)
        # print(f"[{idx}] Label: {text} | Length: {len(text)}")
        # TF.to_pil_image(cropped).save(f"debug_train_crop_{idx}_{text}.png")
        return cropped, label, len(label), text


def collate_fn(batch):
    images, labels, lengths, texts = zip(*batch)
    images = torch.stack(images, 0)
    labels_concat = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels_concat, lengths, texts


def greedy_decoder(preds):
    preds = preds.argmax(2)
    decoded_texts = []
    for seq in preds:
        prev = 0
        s = ""
        for p in seq:
            p = p.item()
            if p != prev and p != 0:
                s += IDX2CHAR[p]
            prev = p
        decoded_texts.append(s)
    return decoded_texts


def train_model(model, train_loader, criterion, optimizer, epochs, checkpoint=None, log=False):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for step, (images, labels_concat, label_lengths, texts) in enumerate(progress_bar):
            images = images.to(DEVICE)

            logits = model(images)
            log_probs = logits
            T = log_probs.size(0)
            B = log_probs.size(1)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=DEVICE)
            label_lengths = label_lengths[:B].to(DEVICE)
            labels_concat = labels_concat.to(DEVICE)

            loss = criterion(log_probs, labels_concat, input_lengths, label_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            decoded = greedy_decoder(logits.permute(1, 0, 2))

            # === Log shapes ===
            if step == 0 and log:
                print(f"[Epoch {epoch+1}] Shapes:")
                print(f"  - images:          {images.shape}")            # [B, 3, H, W]
                print(f"  - logits:          {logits.shape}")           # [T, B, C]
                print(f"  - labels_concat:   {labels_concat.shape}")    # [total_label_len]
                print(f"  - input_lengths:   {input_lengths.shape} {input_lengths.tolist()}")
                print(f"  - label_lengths:   {label_lengths.shape} {label_lengths.tolist()}")
                print(f"  - Text GT sample:  {texts[0]}")


        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {total_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), f"{checkpoint}")
    print(f"Model saved to {checkpoint}")


def evaluate_model(model, test_loader, args):
    model.eval()
    total_edits, total_chars = 0, 0
    all_preds, all_gt = [], []

    with torch.no_grad():
        for images, labels_concat, label_lengths, texts in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = images.to(DEVICE)
            logits = model(images)
            preds = greedy_decoder(logits)
            all_preds.extend(preds)
            all_gt.extend(texts)
            for pred, gt in zip(preds, texts):
                total_edits += editdistance.eval(pred, gt)
                total_chars += len(gt)

    cer = total_edits / total_chars if total_chars > 0 else float('inf')
    if args.print_results:
        for pred, gt in zip(all_preds[:10], all_gt[:10]):
            print(f"Pred: {pred} | GT: {gt}")
    print(f"Character Error Rate (CER): {cer:.4f}")


def main():
    global BATCH_SIZE, NUM_WORKERS, EPOCHS, IMG_WIDTH
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_train", type=str, required=True)
    parser.add_argument("--json_val", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max_train_images", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/transformer_ocr.pth")
    parser.add_argument("--print_results", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--model", choices=["transformer", "crnn"], default="crnn")
    parser.add_argument("--img_width", type=int, default=IMG_WIDTH)
    parser.add_argument("--train_same_test", action="store_true")
    parser.add_argument("--log", action="store_true", help="Log shapes of tensors during training")
    
    args = parser.parse_args()

    IMG_WIDTH = args.img_width
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EPOCHS = args.epochs

    with open(args.json_train) as f:
        train_data = json.load(f)
    train_ids = list(train_data["imgs"].keys())
    if args.max_train_images:
        train_ids = train_ids[:args.max_train_images]
    train_ids = set(train_ids)

    with open(args.json_val) as f:
        val_data = json.load(f)
    val_ids = list(val_data["imgs"].keys())
    if args.max_val_images:
        val_ids = val_ids[:args.max_val_images]
    val_ids = set(val_ids)

    train_dataset = TextOCRDataset(args.json_train, args.image_root, allowed_image_ids=train_ids)
    val_dataset = TextOCRDataset(args.json_val, args.image_root, allowed_image_ids=train_ids if args.train_same_test else val_ids)

    if args.visualize:
        os.makedirs("vis/TransformerOCR", exist_ok=True)
        for i in range(5):
            img_tensor, _, _, label = train_dataset[random.randint(0, len(train_dataset)-1)]
            img_pil = TF.to_pil_image(img_tensor)
            img_pil.save(f"vis/TransformerOCR/train_sample_{i+1}_{label}.png")
        print("Saved 5 sample cropped training images.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = TransformerOCR(NUM_CLASSES).to(DEVICE) if args.model == "transformer" else CRNN(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS, args.checkpoint, args.log)

    print("Starting evaluation...")
    evaluate_model(model, test_loader, args)


if __name__ == '__main__':
    main()
