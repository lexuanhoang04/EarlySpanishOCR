# models.py

import torch
from torch import nn
from torchvision import models

class TransformerOCR(nn.Module):
    def __init__(self, num_classes, d_model=2048, nhead=8, num_layers=3):
        super(TransformerOCR, self).__init__()
        backbone = models.resnet50(pretrained=True)
        modules = list(backbone.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        pooled = self.adaptive_pool(features)
        B, C, H, W = pooled.size()
        squeezed = pooled.squeeze(2)
        seq = squeezed.permute(2, 0, 1)
        encoded = self.transformer_encoder(seq)
        encoded = encoded.permute(1, 0, 2)
        logits = self.fc(encoded)
        return logits.log_softmax(2).permute(1, 0, 2)  # (B, T, C)


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

