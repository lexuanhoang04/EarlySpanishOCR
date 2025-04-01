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
        return output.log_softmax(2)
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (T, B, d_model)
        returns: (T, B, d_model) with position added
        """
        return x + 0.1 * self.pe[:x.size(0)]