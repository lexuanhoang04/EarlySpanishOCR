import json
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import os
import editdistance
import random
import torchvision.transforms.functional as TF

# -------------------- Global Configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (defaults, can be overridden via argparse)
BATCH_SIZE = 32
IMG_HEIGHT = 32    # recognition network input size; adjust as needed
IMG_WIDTH = 128
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 1e-3

# Maximum length for text labels (for CTC)
MAX_LABEL_LEN = 32

# Character set & mapping (adjust as needed)
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Reserve index 0 for CTC blank
CHAR2IDX = {c: i+1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
NUM_CLASSES = len(CHAR2IDX) + 1  # including blank

# -------------------- Dataset --------------------
# Assumes the OCR JSON is in a COCO-like format:
# - "imgs": dict mapping image_id to image info (with "file_name")
# - "anns": dict of annotations; each annotation should contain "image_id", "bbox", and "utf8_string"
class TextOCRDataset(Dataset):
    def __init__(self, json_path, image_root, max_images=None, allowed_image_ids=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_root = image_root
        self.imgs = self.data["imgs"]
        self.anns = list(self.data["anns"].values())
        
        # Limit by unique image IDs
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
        # Some COCO JSON files use strings for keys; adjust accordingly.
        img_info = self.imgs[str(img_id)] if str(img_id) in self.imgs else self.imgs[img_id]
        img_path = os.path.join(self.image_root, os.path.basename(img_info["file_name"]))
        image = Image.open(img_path).convert("RGB")
        # bbox in COCO format: [x, y, w, h]
        x, y, w, h = bbox
        # Crop the text region from the image.
        cropped = image.crop((x, y, x + w, y + h))
        cropped = self.transform(cropped)
        # Encode text as list of indices.
        label_indices = [CHAR2IDX[c] for c in text]
        label = torch.LongTensor(label_indices)
        return cropped, label, len(label), text

# Collate function to handle variable-length targets (for CTC loss)
def collate_fn(batch):
    images, labels, lengths, texts = zip(*batch)
    images = torch.stack(images, 0)
    labels_concat = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels_concat, lengths, texts

# -------------------- Model: Transformer-based OCR --------------------
# We build a model that uses a CNN backbone (ResNet50) and a Transformer encoder for sequence modeling.
class TransformerOCR(nn.Module):
    def __init__(self, num_classes, d_model=2048, nhead=8, num_layers=3):
        super(TransformerOCR, self).__init__()
        # Use ResNet50 backbone, remove avgpool and fc layers.
        backbone = models.resnet50(pretrained=True)
        modules = list(backbone.children())[:-2]  # output: (B, 2048, H, W)
        self.cnn = nn.Sequential(*modules)
        # Adaptive pooling to force height=1 (keeping the width dimension variable)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, 2048, 1, W)
        # Transformer encoder: treat the width dimension as time steps.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Fully connected layer to map each time step to character logits.
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.cnn(x)  # (B, 2048, H', W')
        pooled = self.adaptive_pool(features)  # (B, 2048, 1, W')
        B, C, H, W = pooled.size()  # H should be 1
        # Squeeze the height dimension: (B, 2048, W)
        squeezed = pooled.squeeze(2)
        # Permute to shape (W, B, 2048) for the transformer (time steps, batch, features)
        seq = squeezed.permute(2, 0, 1)
        # Pass through transformer encoder
        encoded = self.transformer_encoder(seq)  # (W, B, 2048)
        # Permute back to (B, W, 2048)
        encoded = encoded.permute(1, 0, 2)
        # Linear classifier: (B, W, num_classes)
        logits = self.fc(encoded)
        # Return log probabilities for CTC loss; shape: (B, W, num_classes)
        return logits.log_softmax(2)

# -------------------- Decoder --------------------
# Greedy decoder for CTC output.
def greedy_decoder(preds):
    # preds: (B, T, num_classes)
    preds = preds.argmax(2)  # (B, T)
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

# -------------------- Training & Evaluation --------------------
def train_model(model, train_loader, criterion, optimizer, epochs, checkpoint=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for images, labels_concat, label_lengths, texts in progress_bar:
            images = images.to(DEVICE)
            labels_concat = labels_concat.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)
            
            logits = model(images)  # (B, T, num_classes)
            # For CTC, we need (T, B, num_classes)
            log_probs = logits.permute(1, 0, 2)
            # Assume output sequence length is constant for each image:
            input_lengths = torch.full((logits.size(0),), log_probs.size(0), dtype=torch.long).to(DEVICE)
            
            loss = criterion(log_probs, labels_concat, input_lengths, label_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {total_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), f"{checkpoint}")
    print(f"Model saved to {checkpoint}")

def evaluate_model(model, test_loader, args):
    model.eval()
    all_preds = []
    all_gt = []
    total_edits = 0
    total_chars = 0

    with torch.no_grad():
        for images, labels_concat, label_lengths, texts in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = images.to(DEVICE)
            logits = model(images)  # (B, T, num_classes)
            preds = greedy_decoder(logits)
            all_preds.extend(preds)
            all_gt.extend(texts)
            # Compute edit distance for each pair
            for pred, gt in zip(preds, texts):
                total_edits += editdistance.eval(pred, gt)
                total_chars += len(gt)

    # Compute Character Error Rate (CER)
    cer = total_edits / total_chars if total_chars > 0 else float('inf')

    if args.print_results:
        # Print out about 10 predictions and overall CER
        for pred, gt in zip(all_preds[:10], all_gt[:10]):
            print(f"Pred: {pred} | GT: {gt}")
    print(f"Character Error Rate (CER): {cer:.4f}")

# -------------------- Main --------------------
def main():
    global BATCH_SIZE, NUM_WORKERS, EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_train", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--json_val", type=str, required=True, help="Path to validation JSON file")
    parser.add_argument("--image_root", type=str, required=True, help="Path to images directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="Number of DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for training and testing")
    parser.add_argument("--print_results", action="store_true", help="Print sample predictions")
    parser.add_argument("--max_train_images", type=int, default=None, help="Limit number of unique training images")
    parser.add_argument("--max_val_images", type=int, default=None, help="Limit number of unique validation images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/transformer_ocr.pth", help="Path to save model checkpoint")
    parser.add_argument("--visualize", action="store_true", help="Save sample cropped training images for inspection")
    parser.add_argument("--check_pred", action="store_true", help="Check model predictions for some validation samples")

    args = parser.parse_args()


    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EPOCHS = args.epochs

    with open(args.json_train, 'r') as f:
        train_data = json.load(f)
    train_ids = list(train_data["imgs"].keys())
    if args.max_train_images:
        train_ids = train_ids[:args.max_train_images]
    train_ids = set(train_ids)

    with open(args.json_val, 'r') as f:
        val_data = json.load(f)
    val_ids = list(val_data["imgs"].keys())
    if args.max_val_images:
        val_ids = val_ids[:args.max_val_images]
    val_ids = set(val_ids)

    train_dataset = TextOCRDataset(args.json_train, args.image_root, allowed_image_ids=train_ids)

    if args.visualize:
        os.makedirs("vis/TransformerOCR", exist_ok=True)
        for i in range(5):  # Save 5 random samples
            img_tensor, _, _, label = train_dataset[random.randint(0, len(train_dataset)-1)]
            img_pil = TF.to_pil_image(img_tensor)
            img_pil.save(f"vis/TransformerOCR/train_sample_{i+1}_{label}.png")
        print("Saved 5 sample cropped training images to `visualization/`")

    val_dataset = TextOCRDataset(args.json_val, args.image_root, allowed_image_ids=val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    model = TransformerOCR(num_classes=NUM_CLASSES, d_model=2048, nhead=8, num_layers=3).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS, args.checkpoint)
    
    print("Starting evaluation...")
    evaluate_model(model, test_loader, args)

if __name__ == '__main__':
    main()