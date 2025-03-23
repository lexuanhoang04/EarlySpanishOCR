import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
import torchvision.models as models
import torch.distributed as dist
import torch.multiprocessing as mp

# Constants
BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 256
NUM_WORKERS = 4
GRID_SIZE = 4
BOXES_PER_CELL = 2
TOP_K = 10  # Number of top predicted boxes to keep per image

# Custom collate function: returns five items
def collate_fn(batch):
    images = []
    boxes = []
    img_ids = []
    widths = []
    heights = []
    
    for img, box, img_id, width, height in batch:
        images.append(img)
        boxes.append(box)  # box is a tensor of shape [N, 4]
        img_ids.append(int(img_id))
        widths.append(float(width))
        heights.append(float(height))
    images = torch.stack(images, dim=0)
    return images, boxes, img_ids, widths, heights

# Dataset
class CocoBoxDataset(Dataset):
    def __init__(self, coco_json, image_root, max_samples=None):
        self.coco = COCO(coco_json)
        self.image_root = image_root
        self.img_ids = list(self.coco.imgs.keys())
        if max_samples:
            self.img_ids = self.img_ids[:max_samples]
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_root, os.path.basename(img_info['file_name']))
        image = Image.open(img_path).convert("RGB")
        width, height = img_info['width'], img_info['height']
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1 = x / width
            y1 = y / height
            x2 = (x + w) / width
            y2 = (y + h) / height
            boxes.append([x1, y1, x2, y2])
        
        # Ensure boxes is a tensor of shape [N, 4] even if empty
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            
        image = self.transform(image)
        return image, boxes, img_id, width, height

# Multi-box detector model: GridBoxDetector
class GridBoxDetector(nn.Module):
    def __init__(self, grid_size=4, boxes_per_cell=2):
        super(GridBoxDetector, self).__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, boxes_per_cell * 4, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)  # Shape: (B, boxes_per_cell*4, grid_size, grid_size)
        x = x.permute(0, 2, 3, 1)  # Shape: (B, grid_size, grid_size, boxes_per_cell*4)
        x = x.reshape(x.size(0), -1, 4)  # Use reshape instead of view
        return torch.sigmoid(x)

# Simple L1 loss computed on the first k boxes (naively, without proper matching)
def simple_l1_loss(preds, targets):
    loss = 0.0
    for pred_boxes, gt_boxes in zip(preds, targets):
        if gt_boxes.numel() == 0:
            continue
        k = min(pred_boxes.size(0), gt_boxes.size(0))
        loss += nn.functional.l1_loss(pred_boxes[:k], gt_boxes[:k])
    return loss / len(preds)

# Training loop: note that distributed samplers require set_epoch() each epoch.
def train(model, dataloader, optimizer, epochs, sampler):
    model.train()
    for epoch in range(epochs):
        # Set epoch for sampler so the shuffling is synchronized.
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for images, targets, img_ids, widths, heights in tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(dist.get_rank() != 0)):
            images = images.to(DEVICE)
            preds = model(images)  # Shape: (B, N_boxes, 4)
            preds = preds.cpu()    # Loss computation on CPU (targets are on CPU)
            loss = simple_l1_loss(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# Evaluation: writes predictions in COCO format (only on rank 0)
def evaluate_coco(model, dataloader, coco_gt_path, output_json, top_k=TOP_K):
    model.eval()
    coco_gt = COCO(coco_gt_path)
    results = []
    with torch.no_grad():
        for images, _, img_ids, widths, heights in tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0)):
            images = images.to(DEVICE)
            preds = model(images).cpu()  # Shape: (B, N, 4)
            for boxes, img_id, width, height in zip(preds, img_ids, widths, heights):
                boxes = boxes[:top_k]  # Keep top_k predictions
                for box in boxes:
                    x1, y1, x2, y2 = box.tolist()
                    x1 *= width
                    y1 *= height
                    x2 *= width
                    y2 *= height
                    w = x2 - x1
                    h = y2 - y1
                    results.append({
                        "image_id": int(img_id),
                        "category_id": 1,
                        "bbox": [x1, y1, w, h],
                        "score": 1.0
                    })
    if dist.get_rank() == 0:
        with open(output_json, 'w') as f:
            json.dump(results, f)
        print(f"Saved predictions to {output_json}")
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_train', required=True)
    parser.add_argument('--coco_val', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--pred_output', default='predictions.json')
    # The following argument is required for distributed training.
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    global DEVICE
    DEVICE = torch.device("cuda", args.local_rank)
    
    # Only rank 0 prints messages
    if dist.get_rank() == 0:
        print(f"Running distributed training on {dist.get_world_size()} GPUs.")
    
    # Create datasets and distributed samplers
    train_dataset = CocoBoxDataset(args.coco_train, args.image_root, args.max_samples)
    val_dataset = CocoBoxDataset(args.coco_val, args.image_root, args.max_samples)
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, sampler=val_sampler, collate_fn=collate_fn)
    
    # Create model and wrap with DistributedDataParallel
    model = GridBoxDetector(grid_size=GRID_SIZE, boxes_per_cell=BOXES_PER_CELL).to(DEVICE)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train and evaluate (only rank 0 will print/evaluate)
    train(model, train_loader, optimizer, epochs=args.epochs, sampler=train_sampler)
    evaluate_coco(model, val_loader, args.coco_val, args.pred_output, top_k=TOP_K)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
