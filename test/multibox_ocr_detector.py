import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

# Global constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 256
NUM_WORKERS = 4
GRID_SIZE = 4
BOXES_PER_CELL = 2
TOP_K = 10  # Number of top predicted boxes to keep during evaluation

# Custom collate function: returns five items
def collate_fn(batch):
    images = []
    boxes = []
    img_ids = []
    widths = []
    heights = []
    for img, box, img_id, width, height in batch:
        images.append(img)
        boxes.append(box)  # box is a tensor of shape [N, 4] (variable-length per image)
        img_ids.append(int(img_id))
        widths.append(float(width))
        heights.append(float(height))
    images = torch.stack(images, dim=0)
    return images, boxes, img_ids, widths, heights

# Dataset: loads image and annotations from a COCO-format JSON.
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
            # Normalize to [0,1] using image dimensions
            x1 = x / width
            y1 = y / height
            x2 = (x + w) / width
            y2 = (y + h) / height
            boxes.append([x1, y1, x2, y2])
        
        # Always return a tensor of shape [N, 4], even if no boxes are present.
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            
        image = self.transform(image)
        return image, boxes, img_id, width, height

# Multi-box detector model: predicts multiple boxes per image.
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
        x = self.head(x)  # shape: (B, boxes_per_cell*4, grid_size, grid_size)
        x = x.permute(0, 2, 3, 1)  # shape: (B, grid_size, grid_size, boxes_per_cell*4)
        # Use reshape so that non-contiguous tensors are handled correctly.
        x = x.reshape(x.size(0), -1, 4)  # Expected shape: (B, grid_size*grid_size*boxes_per_cell, 4)
        return torch.sigmoid(x)

# A naive L1 loss that compares the first k predicted boxes to the ground truth.
def simple_l1_loss(preds, targets):
    loss = 0.0
    for pred_boxes, gt_boxes in zip(preds, targets):
        if gt_boxes.numel() == 0:
            continue
        k = min(pred_boxes.size(0), gt_boxes.size(0))
        loss += nn.functional.l1_loss(pred_boxes[:k], gt_boxes[:k])
    return loss / len(preds)

# Training loop.
def train(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, targets, img_ids, widths, heights in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = images.to(DEVICE)
            preds = model(images)  # Should be (B, 32, 4) for grid_size=4, boxes_per_cell=2
            # Uncomment the next line to debug the output shape:
            # print("Preds shape:", preds.shape)
            preds = preds.cpu()
            loss = simple_l1_loss(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# Evaluation: writes predictions in COCO format.
def evaluate_coco(model, dataloader, coco_gt_path, output_json, top_k=TOP_K):
    model.eval()
    coco_gt = COCO(coco_gt_path)
    results = []
    with torch.no_grad():
        for images, _, img_ids, widths, heights in tqdm(dataloader, desc="Evaluating"):
            images = images.to(DEVICE)
            preds = model(images).cpu()  # shape: (B, N, 4)
            for boxes, img_id, width, height in zip(preds, img_ids, widths, heights):
                boxes = boxes[:top_k]  # Consider only the top_k predictions per image.
                for box in boxes:
                    x1, y1, x2, y2 = box.tolist()
                    # Convert normalized coordinates back to pixel coordinates.
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
    args = parser.parse_args()
    
    train_dataset = CocoBoxDataset(args.coco_train, args.image_root, args.max_samples)
    val_dataset = CocoBoxDataset(args.coco_val, args.image_root, args.max_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    # Instantiate the multi-box detector model.
    model = GridBoxDetector(grid_size=GRID_SIZE, boxes_per_cell=BOXES_PER_CELL).to(DEVICE)
    
    # Debug: print example output shape. Expect (1, 32, 4) with grid_size=4, boxes_per_cell=2.
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        out = model(dummy)
        print("Example model output shape:", out.shape)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train(model, train_loader, optimizer, epochs=args.epochs)
    evaluate_coco(model, val_loader, args.coco_val, args.pred_output)

if __name__ == '__main__':
    main()
