import os
import json
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from pycocotools.coco import COCO
from tqdm import tqdm

# -------------------- Global Variables --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4          # Detection models often use small batch sizes
NUM_WORKERS = 4         # Adjust as needed for your CPU
LR = 1e-4
NUM_EPOCHS = 10

# -------------------- Dataset --------------------
# This dataset uses the COCO API to load images and annotations.
# It returns an image tensor and a target dict in the format expected by torchvision's detection models.
class TextDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_root, transforms_=None):
        self.coco = COCO(json_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_root = image_root
        if transforms_ is None:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms_

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_root, img_info['file_name'].split('/')[-1])
        image = Image.open(img_path).convert("RGB")
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            # COCO format: bbox = [x, y, w, h]
            bbox = ann["bbox"]
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue  # Skip invalid boxes
            # Convert to [x1, y1, x2, y2]
            x2 = x + w
            y2 = y + h
            boxes.append([x, y, x2, y2])
            labels.append(1)  # 1 for text
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd
        
        if self.transforms:
            image = self.transforms(image)
        return image, target

# Collate function for detection: returns a tuple of lists.
def collate_fn(batch):
    return tuple(zip(*batch))

# -------------------- Model --------------------
def get_text_detector(num_classes):
    # Load a pre-trained Faster R-CNN model with ResNet50-FPN backbone.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Replace the head (box predictor) with one for our number of classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

# -------------------- Training Loop --------------------
def train_model(model, dataloader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")
        for images, targets in progress_bar:
            # images: list of image tensors; targets: list of dicts
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

def evaluate_model(model, dataloader, coco_gt_path, output_json):
    model.eval()
    all_results = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = [img.to(DEVICE) for img in images]
            predictions = model(images)  # List of dicts
            for target, prediction in zip(targets, predictions):
                # Ensure the target contains an "image_id" (as a tensor)
                image_id = int(target["image_id"].item())
                prediction["image_id"] = image_id
                prediction["boxes"] = prediction["boxes"].cpu()
                prediction["labels"] = prediction["labels"].cpu()
                prediction["scores"] = prediction["scores"].cpu()
                all_results.append(prediction)

    # Convert the predictions into COCO result format.
    coco_results = []
    for pred in all_results:
        image_id = pred["image_id"]
        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            coco_results.append({
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [x1, y1, w, h],
                "score": float(score.item())
            })

    # Save predictions to a JSON file.
    with open(output_json, "w") as f:
        json.dump(coco_results, f)
    print(f"Saved predictions to {output_json}")

    # Load ground truth and predictions, then run COCO evaluation.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(coco_gt_path)
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_train', type=str, required=True, help="Path to COCO JSON for training")
    parser.add_argument('--json_val', type=str, required=True, help="Path to COCO JSON for validation")
    parser.add_argument('--image_root', type=str, required=True, help="Path to the directory containing images")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--output_json', type=str, default="predictions.json", help="Path to save predictions")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/faster_RCNN.pth', help="Path to model checkpoint")

    args = parser.parse_args()
    
    # Create datasets
    train_dataset = TextDetectionDataset(args.json_train, args.image_root)
    val_dataset = TextDetectionDataset(args.json_val, args.image_root)
    
    # Create data loaders with collate function and tqdm
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    # For text detection, we use 2 classes: background and text.
    num_classes = 2
    model = get_text_detector(num_classes)
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Train the detector with tqdm progress bars.
    train_model(model, train_loader, optimizer, args.epochs)
    
    # Evaluate (display one batch of predictions)
    evaluate_model(model, val_loader, args.json_val, args.output_json)

    
    # Optionally, save the model.
    torch.save(model.state_dict(), args.checkpoint)
    print(f"Model saved to {args.checkpoint}")

if __name__ == '__main__':
    main()
