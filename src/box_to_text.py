import json
import argparse
from PIL import Image
from tqdm import tqdm
import yaml

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, sys
import editdistance
import random
import torchvision.transforms.functional as TF
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.getcwd())
from models import TransformerOCR, CRNN, CRNN_ResNet18, TransformerOCR_Line
from src.utils import greedy_decoder, setup_chars

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

CHAR2IDX = {}
IDX2CHAR = {}
NUM_CLASSES = 0


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup character set
    global CHAR2IDX, IDX2CHAR, NUM_CLASSES
    chars, CHAR2IDX, IDX2CHAR, NUM_CLASSES = setup_chars(config)

    return config

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
        # Text OCR structure
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_root, img_info["file_name"]) # for FUNSD
            
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

            # === Show a few predictions for debugging ===
            if step == 0:
                decoded_preds = greedy_decoder(log_probs, model.idx2char if hasattr(model, 'idx2char') else {})
                print("\nSample predictions (first 5):")
                for i in range(min(5, len(decoded_preds))):
                    print(f"Pred: {decoded_preds[i]} | GT: {texts[i]}")

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

    os.makedirs(args.vis_path, exist_ok=True)
    
    with torch.no_grad():
        for images, labels_concat, label_lengths, texts in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = images.to(DEVICE)
            logits = model(images)
            preds = greedy_decoder(logits.permute(1, 0, 2), IDX2CHAR) # [T, B, C] -> [B, T, C]

            all_preds.extend(preds)
            all_gt.extend(texts)
            for pred, gt in zip(preds, texts):
                total_edits += editdistance.eval(pred, gt)
                total_chars += len(gt)
            if args.debug:
                for i in range(min(len(images), 3)):
                    TF.to_pil_image(images[i].cpu()).save(f"{args.vis_path}/debug_eval_{i}_{texts[i]}.png")
                    print(f"Saved: debug_eval_{i}_{texts[i]}.png")
            # in evaluate_model() or your test loop

    cer = total_edits / total_chars if total_chars > 0 else float('inf')
    if args.print_results:
        for pred, gt in zip(all_preds[:10], all_gt[:10]):
            print(f"Pred: {pred} | GT: {gt}")
    print(f"Character Error Rate (CER): {cer:.4f}")


def main(rank=0, world_size=1, is_ddp=False, args=None, config=None):

    global BATCH_SIZE, NUM_WORKERS, EPOCHS, IMG_WIDTH, LEARNING_RATE
    # === DDP setup ===
    if is_ddp:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = DEVICE

    IMG_WIDTH = args.img_width
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    IMAGE_HEIGHT = args.img_height

    with open(args.json_train) as f:
        train_data = json.load(f)
    train_ids = list(train_data["imgs"].keys())
    if args.max_train_images and args.max_train_images != "None":
        train_ids = train_ids[:args.max_train_images]
    train_ids = set(train_ids)

    # print("Training file:", list(train_ids)[:5])
    with open(args.json_val) as f:
        val_data = json.load(f)
    val_ids = list(val_data["imgs"].keys())
    if args.max_val_images and args.max_val_images != "None":
        val_ids = val_ids[:args.max_val_images]
    val_ids = set(val_ids)

    train_dataset = TextOCRDataset(args.json_train, args.image_root, allowed_image_ids=train_ids)
    val_dataset = TextOCRDataset(args.json_val, args.image_root, allowed_image_ids=val_ids)
    
    print("Training file:", args.json_train)

    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    if args.model == "transformer":
        model = TransformerOCR(NUM_CLASSES).to(device)
    elif args.model == "crnn":
        model = CRNN(NUM_CLASSES).to(device)
    elif args.model == "crnn_resnet18":
        model = CRNN_ResNet18(NUM_CLASSES).to(device)
    elif args.model == "transformer_line":
        model = TransformerOCR_Line(NUM_CLASSES).to(device)
    if args.ddp:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # === Load checkpoint (always before evaluation) ===
    if args.eval_only or os.path.exists(args.checkpoint):
        if rank == 0 or not args.ddp:
            print(f"Loading model weights from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # === Train ===
    if not args.eval_only:
        if rank == 0 or not args.ddp:
            print("Starting training...")
        train_model(model, train_loader, criterion, optimizer, EPOCHS, args.checkpoint, args.log)

    # === Evaluate ===
    if rank == 0 or not args.ddp:
        print("Starting evaluation...")
        evaluate_model(model.module if args.ddp else model, test_loader, args)

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--print_results", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Load from YAML config
    config = load_config(args.config)

    # Merge YAML into args namespace
    for k, v in config.items():
        setattr(args, k, v)

    # DDP or single-GPU entry point
    if args.ddp:
        world_size = min(args.gpus, torch.cuda.device_count())
        mp.spawn(main, args=(world_size, True, args, config), nprocs=world_size)
    else:
        main(args=args, config=config)
