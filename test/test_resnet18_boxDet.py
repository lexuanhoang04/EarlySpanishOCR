import json
import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
IMG_HEIGHT = 64
IMG_WIDTH = 256
NUM_WORKERS = 4

class TextOCRBoxDataset(Dataset):
    def __init__(self, json_path, image_root, max_samples=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.imgs = self.data['imgs']
        self.anns = list(self.data['anns'].values())

        self.samples = []
        for ann in self.anns:
            img_id = ann['image_id']
            bbox = ann['bbox']  # x, y, w, h
            self.samples.append((img_id, bbox))
            if max_samples and len(self.samples) >= max_samples:
                break

        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, bbox = self.samples[idx]
        img_info = self.data['imgs'][img_id]
        img_path = os.path.join(self.image_root, os.path.basename(img_info['file_name']))
        image = Image.open(img_path).convert("RGB")

        width = int(img_info['width'])
        height = int(img_info['height'])

        x, y, w, h = bbox
        x1 = x / width
        y1 = y / height
        x2 = (x + w) / width
        y2 = (y + h) / height

        image = self.transform(image)
        return image, torch.tensor([x1, y1, x2, y2], dtype=torch.float32), img_id, bbox

def convert_to_coco_format(dataset):
    coco = {
        "info": {
            "description": "TextOCR Converted Dataset",
            "version": "1.0",
            "year": datetime.datetime.now().year
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text"}]
    }
    ann_id = 1
    for idx in range(len(dataset)):
        _, _, img_id, bbox = dataset[idx]
        img_info = dataset.data['imgs'][img_id]
        width = int(img_info['width'])
        height = int(img_info['height'])
        coco['images'].append({
            "id": idx,
            "width": width,
            "height": height,
            "file_name": img_info['file_name']
        })
        coco['annotations'].append({
            "id": ann_id,
            "image_id": idx,
            "bbox": bbox,
            "category_id": 1,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        ann_id += 1
    return coco

class BoxRegressionModel(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        base = models.resnet18(pretrained=True)
        if freeze_backbone:
            for name, param in base.named_parameters():
                if not name.startswith("layer4") and not name.startswith("layer3") and not name.startswith("bn1"):
                    param.requires_grad = False
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.fc(x)

def evaluate_coco(model, dataset, dataloader):
    model.eval()
    coco_gt_dict = convert_to_coco_format(dataset)
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    results = []
    with torch.no_grad():
        for i, (images, _, _, _) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(DEVICE)
            preds = model(images).cpu()
            for j, pred in enumerate(preds):
                x1, y1, x2, y2 = pred
                w = x2 - x1
                h = y2 - y1
                results.append({
                    "image_id": i * BATCH_SIZE + j,
                    "category_id": 1,
                    "bbox": [x1.item(), y1.item(), w.item(), h.item()],
                    "score": 1.0
                })

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, targets, _, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            preds = model(images)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()

    dataset = TextOCRBoxDataset(args.json, args.image_root, max_samples=5000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = BoxRegressionModel(freeze_backbone=args.freeze).to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    train(model, dataloader, criterion, optimizer, epochs=5)
    evaluate_coco(model, dataset, dataloader)