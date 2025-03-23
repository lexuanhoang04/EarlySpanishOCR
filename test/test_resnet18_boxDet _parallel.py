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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
BATCH_SIZE = 64
IMG_HEIGHT = 64
IMG_WIDTH = 256
NUM_WORKERS = 6

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

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        bbox = anns[0]['bbox']  # assume one bbox per image

        width, height = img_info['width'], img_info['height']
        x, y, w, h = bbox
        x1 = x / width
        y1 = y / height
        x2 = (x + w) / width
        y2 = (y + h) / height

        image = self.transform(image)
        return image, torch.tensor([x1, y1, x2, y2], dtype=torch.float32), img_id, bbox

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

def evaluate_coco(model, dataloader, coco_gt_path, output_json):
    model.eval()
    coco_gt = COCO(coco_gt_path)

    results = []
    with torch.no_grad():
        for i, (images, _, img_ids, _) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(DEVICE)
            preds = model(images).cpu()
            for img_id, pred in zip(img_ids, preds):
                x1, y1, x2, y2 = pred.tolist()
                w = x2 - x1
                h = y2 - y1
                results.append({
                    "image_id": img_id,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_train', required=True)
    parser.add_argument('--coco_val', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--pred_output', default='predictions.json')

    args = parser.parse_args()

    train_dataset = CocoBoxDataset(args.coco_train, args.image_root, args.max_samples)
    val_dataset = CocoBoxDataset(args.coco_val, args.image_root, args.max_samples)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = BoxRegressionModel(freeze_backbone=args.freeze)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    train(model, train_loader, criterion, optimizer, epochs=args.epochs)
    evaluate_coco(model, val_loader, args.coco_val, args.pred_output)
