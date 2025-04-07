import torch
from torchvision.transforms.functional import resize, pad

def greedy_decoder(preds, idx2char):
    preds = preds.argmax(2)  # (T, B)
    decoded_texts = []
    for seq in preds.permute(1, 0):  # (B, T)
        prev = 0
        s = ""
        for p in seq:
            p = p.item()
            if p != prev and p != 0:
                s += idx2char.get(p, "")
            prev = p
        decoded_texts.append(s)
    return decoded_texts

def beam_search_decoder(probs, idx2char, beam_width=10):
    T, B, C = probs.shape
    results = []
    for b in range(B):
        beam = [(tuple(), 0.0)]  # (sequence, log prob)
        for t in range(T):
            new_beam = []
            for seq, score in beam:
                for c in range(C):
                    new_seq = seq + (c,)
                    new_score = score + torch.log(probs[t, b, c] + 1e-10).item()
                    new_beam.append((new_seq, new_score))
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
        # collapse repeats and remove blanks (0)
        best_seq = beam[0][0]
        decoded = ""
        prev = None
        for c in best_seq:
            if c != prev and c != 0:
                decoded += idx2char.get(c, "")
            prev = c
        results.append(decoded)
    return results
    
def setup_chars(cfg):
    chars = cfg["chars"]  # Expect list from YAML

    special_tokens = ["<pad>", "<sos>", "<eos>"]
    if "attention" in cfg["model"]:
        
        all_chars = special_tokens + [c for c in chars if c not in special_tokens]
    else:
        all_chars = [c for c in chars if c not in special_tokens]
    char2idx = {c: i for i, c in enumerate(all_chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    num_classes = len(all_chars)
    return all_chars, char2idx, idx2char, num_classes



def resize_pad_to_1024(image, target_height=32, max_width=1024):
    w, h = image.size
    new_w = int(w * (target_height / h))
    image = resize(image, (target_height, new_w))  # preserve aspect ratio

    if new_w < max_width:
        pad_width = max_width - new_w
        image = pad(image, padding=(0, 0, pad_width, 0), fill=255)  # pad right side
    else:
        image = resize(image, (target_height, max_width))  # downsample if too wide

    return image