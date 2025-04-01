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
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import functional as TF
from PIL import ImageDraw

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
        img_path = os.path.join(self.image_root, img_info['file_name'].split('/')[-1]) \
            if 'TextOCR' in img_info['file_name'] else os.path.join(self.image_root, img_info['file_name'])
        
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Resize while keeping aspect ratio
        max_size = 1000
        scale = min(max_size / orig_w, max_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)

        # Apply resizing to bounding boxes
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            x2 = x + w
            y2 = y + h
            # Scale box
            boxes.append([x * scale, y * scale, x2 * scale, y2 * scale])
            labels.append(1)
            areas.append(ann.get("area", w * h) * scale * scale)
            iscrowd.append(ann.get("iscrowd", 0))

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

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
    # Reduce number of region proposals (default is 2000 for train, 1000 for test)
    model.roi_heads.detections_per_img = 250  # Reduce final detections (default 100)

    return model

# -------------------- Training Loop --------------------
def train_model(model, dataloader, optimizer, num_epochs):
    model.train()
    scaler = GradScaler()  # For automatic mixed precision

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

        for images, targets in progress_bar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with autocast():  # AMP enabled
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
        torch.cuda.empty_cache()
@torch.no_grad()
def predict(model, dataloader):
    model.eval()
    all_results = []
    for images, targets in tqdm(dataloader, desc="Predicting", unit="batch"):
        images = [img.to(DEVICE) for img in images]
        predictions = model(images)
        for target, pred in zip(targets, predictions):
            image_id = int(target["image_id"].item())
            pred["image_id"] = image_id
            pred["boxes"] = pred["boxes"].cpu()
            pred["labels"] = pred["labels"].cpu()
            pred["scores"] = pred["scores"].cpu()
            all_results.append(pred)
    return all_results

def evaluate(model, dataloader, coco_gt_path, output_json):
    results = predict(model, dataloader)

    coco_results = []
    for pred in results:
        image_id = pred["image_id"]
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            x1, y1, x2, y2 = box.tolist()
            coco_results.append({
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score.item())
            })

    with open(output_json, "w") as f:
        json.dump(coco_results, f)
    print(f"Saved predictions to {output_json}")

    coco_gt = COCO(coco_gt_path)
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def freeze_backbone_but_not_rpn_or_heads(model):
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze RPN
    for param in model.rpn.parameters():
        param.requires_grad = True

    # Unfreeze ROI heads
    for param in model.roi_heads.parameters():
        param.requires_grad = True

    # Optionally unfreeze just the last few layers of the backbone
    for name, param in model.backbone.body.named_parameters():
        if name.startswith('layer3') or name.startswith('layer4'):
            param.requires_grad = True

    # Print how many layers are trainable
    num_trainable = sum(p.requires_grad for p in model.parameters())
    print(f"Number of trainable parameters: {num_trainable}")

def visualize_one_resized_example(dataset, output_path="vis/resized_example_with_boxes.jpg"):
    image, target = dataset[0]
    boxes = target["boxes"]

    # Convert tensor to numpy for plotting
    image_np = TF.to_pil_image(image)

    # Draw boxes
    draw = ImageDraw.Draw(image_np)
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    image_np.save(output_path)
    print(f"Saved resized visualization to {output_path}")

# -------------------- Main --------------------
def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_train', type=str, required=True, help="Path to COCO JSON for training")
    parser.add_argument('--json_val', type=str, required=True, help="Path to COCO JSON for validation")
    parser.add_argument('--image_root', type=str, required=True, help="Path to the directory containing images")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--output_json', type=str, default="predictions.json", help="Path to save predictions")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to model checkpoint")
    parser.add_argument('--checkpoint_trained', type=str, default='checkpoints/faster_RCNN_trained.pth', help="Path to model checkpoint for training")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes (including background)")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument('--visualize', action='store_true', help="Plot image and boxes after resizing")

    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    
    # Create datasets
    train_dataset = TextDetectionDataset(args.json_train, args.image_root)
    val_dataset = TextDetectionDataset(args.json_val, args.image_root)
    
    if args.visualize:
        visualize_one_resized_example(train_dataset)
        return

    # Create data loaders with collate function and tqdm
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    # For text detection, we use 2 classes: background and text.
    num_classes = args.num_classes

    model = get_text_detector(num_classes)
    freeze_backbone_but_not_rpn_or_heads(model)
    if args.checkpoint:
        # Load the model from a checkpoint
        print(f"Loading model from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    # Train the detector with tqdm progress bars.
    train_model(model, train_loader, optimizer, args.epochs)
    
    # Evaluate (display one batch of predictions)
    evaluate(model, val_loader, args.json_val, args.output_json)

    
    # Optionally, save the model.
    torch.save(model.state_dict(), args.checkpoint_trained)
    print(f"Model saved to {args.checkpoint_trained}")

if __name__ == '__main__':
    main()
