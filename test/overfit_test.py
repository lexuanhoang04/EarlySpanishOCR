import argparse
import yaml
import sys, os
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

# Adjust sys path
os.sys.path.insert(0, os.getcwd())
from models import CRNN, TransformerOCR, CRNN_ResNet18_Line, TransformerOCR_Line
import torch.nn.functional as F
from src.utils import greedy_decoder, setup_chars

# ---------------- Main Script ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--gt_text_path", type=str, required=True)
    parser.add_argument("--break_on_success", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load label
    with open(args.gt_text_path, "r") as f:
        gt_text = f.read().strip()

    # # Vocab
    # chars = cfg["chars"]
    # char2idx = {c: i + 1 for i, c in enumerate(chars)}  # shift by 1
    # char2idx[""] = 0  # CTC blank
    # idx2char = {i: c for c, i in char2idx.items()}
    # num_classes = len(char2idx) + 1 # ✅ includes blank

    chars, char2idx, idx2char, num_classes = setup_chars(cfg)

    print(f"Char list length: {len(chars)}")
    print(f"Vocab size (num_classes): {num_classes}")
    print(f"Space index: {char2idx.get(' ', '❌ not found')}")

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((cfg.get("img_height", 32), cfg.get("img_width", 512))),
        transforms.ToTensor(),
    ])

    # Load image and label
    image = Image.open(args.img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    label_indices = [char2idx.get(c, 0) for c in gt_text]
    label = torch.LongTensor(label_indices).to(device)
    target_len = torch.tensor([len(label)], dtype=torch.long).to(device)

    # Model selection
    model_type = cfg["model"]
    if model_type == "crnn":
        model = CRNN(num_classes=num_classes).to(device)
    elif model_type == "transformer":
        model = TransformerOCR(num_classes=num_classes).to(device)
    elif model_type == "crnn_resnet18_line":
        model = CRNN_ResNet18_Line(num_classes=num_classes).to(device)
    elif model_type == "transformer_line":
        model = TransformerOCR_Line(num_classes=num_classes).to(device)
    elif model_type == "transformer_attention":
        model = TransformerOCR_Attention(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-4))

    # CTC loss for all models except attention-based
    use_attention = "attention" in model_type
    criterion = nn.CrossEntropyLoss(ignore_index=0) if use_attention else nn.CTCLoss(blank=0, zero_infinity=True)

    print("Start training")
    model.train()
    for epoch in range(1, cfg.get("epochs", 1000) + 1):
        optimizer.zero_grad()

        if use_attention:
            sos_token = char2idx["<sos>"]
            tgt_input = torch.tensor([[sos_token] + label_indices[:-1]], dtype=torch.long).to(device).transpose(0, 1)
            tgt_output = torch.tensor(label_indices, dtype=torch.long).unsqueeze(1).to(device)
            output = model(image_tensor, tgt_input)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            pred_ids = model.greedy_decode(image_tensor)
            pred_text = "".join([idx2char[i] for i in pred_ids])
        else:
            logits = model(image_tensor)
            log_probs = logits
            input_len = torch.full(size=(1,), fill_value=log_probs.size(0), dtype=torch.long).to(device)
            loss = criterion(log_probs, label, input_len, target_len)
            pred_text = greedy_decoder(log_probs, idx2char)[0]

            # # Space probability inspection
            # if " " in char2idx:
            #     space_index = char2idx[" "]
            #     with torch.no_grad():
            #         space_probs = log_probs[:, 0, space_index].exp()
            #         print("Space probs (top 10):", space_probs[:10].tolist())

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Pred={pred_text}, GT={gt_text}")
        if pred_text == gt_text and args.break_on_success:
            print("Model overfitted!")
            break
        model.train()

if __name__ == "__main__":
    main()