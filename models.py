# models.py

import torch
from torch import nn
from torchvision import models
import math

class TransformerOCR(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_layers=3):
        super(TransformerOCR, self).__init__()

        # CNN Backbone (ResNet50 outputs 2048 channels)
        backbone = models.resnet50(weights="IMAGENET1K_V1")
        modules = list(backbone.children())[:-2]  # Remove avgpool & fc

        # Modify the stride of the first Bottleneck in layer4 to reduce downsampling
        modules[-1][0].conv2.stride = (1, 1)
        modules[-1][0].downsample[0].stride = (1, 1)

        self.cnn = nn.Sequential(*modules)

        # Adaptive pooling to collapse height -> (B, 2048, 1, W)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # Project CNN output (2048) to Transformer dimension (d_model)
        self.channel_projector = nn.Linear(2048, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer for class prediction
        self.fc = nn.Linear(d_model, num_classes)

        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, x):
        features = self.cnn(x)                     # (B, 2048, H, W)
        pooled = self.adaptive_pool(features)     # (B, 2048, 1, W)
        squeezed = pooled.squeeze(2)              # (B, 2048, W)
        squeezed = squeezed.permute(0, 2, 1)      # (B, W, 2048)

        projected = self.channel_projector(squeezed)  # (B, W, d_model)
        seq = projected.permute(1, 0, 2)              # (T, B, d_model)

        seq = self.pos_encoder(seq)                  # Add positional info
        encoded = self.transformer_encoder(seq)      # (T, B, d_model)
        encoded = encoded.permute(1, 0, 2)           # (B, T, d_model)

        logits = self.fc(encoded)                    # (B, T, num_classes)
        return logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)


class CRNN(nn.Module):
    def __init__(self, num_classes, cnn_out_channels=64, lstm_hidden_size=256, lstm_layers=2):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels, hidden_size=lstm_hidden_size,
            num_layers=lstm_layers, bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = self.adaptive_pool(features)
        features = features.squeeze(2)
        features = features.permute(2, 0, 1)

        lstm_out, _ = self.lstm(features)
        output = self.fc(lstm_out)
        return output.log_softmax(2)

class CRNN_ResNet18(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=256, lstm_layers=2, dropout=0.3):
        super(CRNN_ResNet18, self).__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        # Remove the final FC and avgpool layers
        self.cnn = nn.Sequential(*list(resnet.children())[:-3])  # Output shape: [B, 256, H/8, W/8]

        self.conv_out_channels = 256  # resnet18 last conv output channels
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # to make height = 1

        self.lstm = nn.LSTM(
            input_size=self.conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)            # [B, 256, H/8, W/8]
        features = self.pool(features)    # [B, 256, 1, W']
        features = features.squeeze(2)    # [B, 256, W']
        features = features.permute(2, 0, 1)  # [W', B, 256]

        lstm_out, _ = self.lstm(features)
        output = self.fc(self.dropout(lstm_out))
        return output.log_softmax(2) # [W', B, num_classes]
        # [T, B, num_classes]
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return x

class CRNN_ResNet18_Line(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=256, lstm_layers=2, dropout=0.3):
        super(CRNN_ResNet18_Line, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-3])  # [B, 256, H/8, W/8]
        self.pool = nn.AdaptiveAvgPool2d((1, None))              # [B, 256, 1, W']
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)               # [B, 256, H/8, W/8]
        x = self.pool(x).squeeze(2)   # [B, 256, W']
        x = x.permute(2, 0, 1)        # [W', B, 256]
        x, _ = self.lstm(x)           # [W', B, 2*hidden]
        x = self.fc(self.dropout(x))  # [W', B, num_classes]
        return x.log_softmax(2)

class TransformerOCR_Line(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_layers=3):
        super(TransformerOCR_Line, self).__init__()

        # CNN Backbone (ResNet50 outputs 2048 channels)
        backbone = models.resnet50(weights="IMAGENET1K_V1")
        modules = list(backbone.children())[:-2]  # Remove avgpool & fc

        # Reduce stride of the first block in layer4 to keep spatial resolution higher
        modules[-1][0].conv2.stride = (1, 1)
        modules[-1][0].downsample[0].stride = (1, 1)

        self.cnn = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # Output: (B, 2048, 1, W)
        self.channel_projector = nn.Linear(2048, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_encoder = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        features = self.cnn(x)                  # (B, 2048, H, W)
        pooled = self.adaptive_pool(features)   # (B, 2048, 1, W)
        squeezed = pooled.squeeze(2)            # (B, 2048, W)
        squeezed = squeezed.permute(0, 2, 1)    # (B, W, 2048)

        projected = self.channel_projector(squeezed)  # (B, W, d_model)
        seq = projected.permute(1, 0, 2)              # (T, B, d_model)

        encoded = self.transformer_encoder(self.pos_encoder(seq))  # (T, B, d_model)
        logits = self.fc(encoded)                    # (T, B, C)

        return logits.log_softmax(2)  

class TransformerAttentionOCR(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_layers=3, max_len=100):
        super().__init__()
        # CNN backbone (ResNet50)
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        # modules = list(resnet.children())[:-2]  # remove avgpool & fc
        # self.cnn = nn.Sequential(*modules)

        self.cnn = nn.Sequential(
            resnet.conv1,   # [B, 64, H/2, W/2]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, # [B, 64, H/4, W/4]
            resnet.layer1,  # [B, 64, H/4, W/4]
            resnet.layer2,  # [B, 128, H/8, W/8]
            resnet.layer3   # [B, 256, H/16, W/16]
            # NOTE: skip layer4 to preserve resolution
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, None))

        self.input_proj = nn.Linear(256, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.embedding = nn.Embedding(num_classes, d_model)
        self.fc_out = nn.Linear(d_model, num_classes)
        self.max_len = max_len



    def forward(self, images, tgt_input):
        features = self.cnn(images)
        #print("CNN features:", features.shape)  # [B, 2048, H, W] or [B, 512, H, W] if resnet18

        pooled = self.adaptive_pool(features)
        #print("Pooled:", pooled.shape)  # [B, C, 4, W]

        B, C, H, W = pooled.shape
        pooled = pooled.permute(0, 2, 3, 1).reshape(B, H * W, C)
        #print("Flattened:", pooled.shape)  # [B, S, C]

        memory = self.input_proj(pooled)
        #print("After input_proj:", memory.shape)  # [B, S, d_model]

        memory = self.pos_encoder(memory)
        #print("After pos_encoder:", memory.shape)

        tgt_emb = self.embedding(tgt_input)
        #print("Target embedding:", tgt_emb.shape)  # [B, T, d_model]

        tgt = self.pos_decoder(tgt_emb)
        #print("After pos_decoder:", tgt.shape)

        T = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(T, tgt.device)
        #print("Target mask:", tgt_mask.shape)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        #print("Transformer output:", output.shape)  # [B, T, d_model]

        logits = self.fc_out(output)
        #print("Final logits:", logits.shape)  # [B, T, num_classes]
        return logits



    def generate_square_subsequent_mask(self, sz, device):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)

    def generate(self, image, sos_token, eos_token, max_len=100, min_len=4):
        self.eval()
        with torch.no_grad():
            #print(f"[generate] Input image shape: {image.shape}")  # [B, C, H, W]

            features = self.cnn(image)  # [B, C, H', W']
            #print(f"[generate] CNN features shape: {features.shape}")

            pooled = self.adaptive_pool(features)  # [B, C, 4, W]
            #print(f"[generate] Pooled features shape: {pooled.shape}")

            B, C, H, W = pooled.shape
            pooled = pooled.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, S, C]
            #print(f"[generate] Flattened pooled shape: {pooled.shape}")

            memory = self.input_proj(pooled)  # [B, S, d_model]
            #print(f"[generate] Memory shape after input_proj: {memory.shape}")

            memory = self.pos_encoder(memory)
            #print(f"[generate] Memory shape after pos_encoder: {memory.shape}")

            generated = torch.full((B, 1), sos_token, dtype=torch.long, device=image.device)
            finished = torch.zeros(B, dtype=torch.bool, device=image.device)
            #print(f"[generate] Initial generated shape: {generated.shape}")

            for t in range(max_len):
                T = generated.size(1)
                tgt_emb = self.embedding(generated)  # [B, T, d_model]
                #print(f"[generate] Step {t}: tgt_emb shape: {tgt_emb.shape}")

                tgt = self.pos_decoder(tgt_emb)
                #print(f"[generate] Step {t}: tgt after pos_decoder shape: {tgt.shape}")

                tgt_mask = self.generate_square_subsequent_mask(T, image.device)
                #print(f"[generate] Step {t}: tgt_mask shape: {tgt_mask.shape}")

                out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # [B, T, d_model]
                #print(f"[generate] Step {t}: decoder output shape: {out.shape}")

                logits = self.fc_out(out[:, -1])  # [B, num_classes]
                #print(f"[generate] Step {t}: logits shape: {logits.shape}")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                #print(f"[generate] Step {t}: next_token: {next_token.squeeze().tolist()}")

                generated = torch.cat([generated, next_token], dim=1)
                #print(f"[generate] Step {t}: generated shape: {generated.shape}")

                if t < 3:
                    logits[:, eos_token] = -float('inf')  # prevent early <eos>

                if t + 1 >= min_len:
                    finished |= (next_token.squeeze(1) == eos_token)

                if finished.all():
                    #print(f"[generate] All sequences finished at step {t}")
                    break

            return generated[:, 1:]  # Remove <sos>


class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, attn_weights = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # shape: [B, num_heads, T, S]
        )
        self.last_attn_weights = attn_weights.detach()  # Save weights for inspection

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt