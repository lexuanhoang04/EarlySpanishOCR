import yaml
import sys, os
import torch
import argparse
from PIL import Image
from torchvision import transforms
import json
from tqdm import tqdm
import editdistance

sys.path.insert(0, os.getcwd())
from src.utils import greedy_decoder, setup_chars

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_name, num_classes):
    from models import CRNN, CRNN_ResNet18, TransformerOCR, TransformerOCR_Line
    if model_name == "crnn":
        return CRNN(num_classes)
    elif model_name == "crnn_resnet18":
        return CRNN_ResNet18(num_classes)
    elif model_name == "transformer":
        return TransformerOCR(num_classes)
    elif model_name == "transformer_line":
        return TransformerOCR_Line(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def clean_text(text, valid_chars):
    text = text.replace("-", "")
    return "".join(c for c in text if c in valid_chars or c == " ")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="Path to a cropped word/line image (for single mode)")
    parser.add_argument("--json_input", help="Path to TextOCR JSON file (for batch mode)")
    parser.add_argument("--json_output", help="Path to save the updated JSON (for batch mode)")
    parser.add_argument("--image_root", help="Root directory where images are stored (for batch mode)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--gt_txt", help="Path to processed ground truth txt file (full concatenated GT)")
    parser.add_argument("--print_output", action="store_true", help="Print the output instead of saving it")
    parser.add_argument("--gt_dir")

    args = parser.parse_args()

    config = load_config(args.config)
    CHARSET, CHAR2IDX, IDX2CHAR, NUM_CLASSES = setup_chars(config)
    valid_chars = "".join(CHARSET)

    checkpoint = config.get("checkpoint", None)
    model = load_model(config["model"], NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(checkpoint, map_location=DEVICE)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    if args.json_input:
        assert args.image_root and args.json_output, "Both --json_output and --image_root must be provided in batch mode"

        if not args.gt_dir:
            raise ValueError("You must specify --gt_dir to compute CER per image")

        with open(args.json_input, "r") as f:
            data = json.load(f)

        transform = transforms.Compose([
            transforms.Resize((config["img_height"], config["img_width"])),
            transforms.ToTensor(),
        ])

        total_cer = 0.0
        cer_count = 0

        for image_id, ann_ids in tqdm(data["imgToAnns"].items(), desc="Processing images"):
            img_info = data["imgs"][image_id]
            filename = img_info["file_name"]
            img_path = os.path.join(args.image_root, filename)
            image = Image.open(img_path).convert("RGB")

            sorted_anns = sorted(
                [data["anns"][ann_id] for ann_id in ann_ids],
                key=lambda ann: (ann["bbox"][1], ann["bbox"][0])
            )

            pred_texts = []

            for ann in sorted_anns:
                x1, y1, w, h = map(int, ann["bbox"])
                x2 = x1 + w
                y2 = y1 + h
                cropped = image.crop((x1, y1, x2, y2))
                image_tensor = transform(cropped).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(image_tensor)
                    decoded = greedy_decoder(logits, IDX2CHAR)

                pred = clean_text(decoded[0], valid_chars)
                ann["utf8_string"] = pred
                pred_texts.append(pred)

                # if args.print_output:
                #     print(f"{image_id} | {ann['id']} | pred: {pred}")

            # CER for current image
            gt_filename = filename.replace(".jpg", ".txt")
            gt_path = os.path.join(args.gt_dir, gt_filename)
            if os.path.exists(gt_path):
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_text = f.read().strip()
                pred_joined = " ".join(pred_texts)
                if args.print_output:
                    print('pred', pred_joined)
                    print('gt_text', gt_text)
                cer = editdistance.eval(pred_joined, gt_text) / max(len(gt_text), 1)
                total_cer += cer
                cer_count += 1

        avg_cer = total_cer / max(cer_count, 1)
        print(f"\n✅ Average CER over {cer_count} files: {avg_cer:.4f}")

        with open(args.json_output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved predictions to {args.json_output}")


    elif args.image_path:
        transform = transforms.Compose([
            transforms.Resize((config["img_height"], config["img_width"])),
            transforms.ToTensor(),
        ])

        image = Image.open(args.image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(image_tensor)
            decoded = greedy_decoder(logits, IDX2CHAR)

        pred_text = clean_text(decoded[0], valid_chars)
        print("Prediction:", pred_text)

if __name__ == "__main__":
    main()