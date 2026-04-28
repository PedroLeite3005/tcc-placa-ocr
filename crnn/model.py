"""CRNN + CTC para reconhecimento de placas veiculares.

Arquitetura baseada em Shi et al. (2015).

Fluxo para entrada (B, 3, 64, 256):
  CNN backbone:    (B, 512,  4, 64)
  mean(dim=2):     (B, 512, 64)
  permute(2,0,1):  (T=64, B, 512)
  BiLSTM x2:       (T,    B, 512)
  Linear:          (T,    B, 37)
  permute(1,0,2):  (B,  T=64, 37)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]


class CRNN(nn.Module):
    """CRNN clássico com cabeça CTC.

    Args:
        in_ch: canais da imagem de entrada (padrão 3).
        num_classes: tamanho do vocabulário incluindo blank=0 (padrão 37).
        hidden: unidades por direção do BiLSTM (padrão 256).
    """

    def __init__(
        self,
        in_ch: int = 3,
        num_classes: int = 37,
        hidden: int = 256,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            *_conv_bn_relu(in_ch, 64),
            nn.MaxPool2d((2, 2)),            # H/2,  W/2  → 32×128
            *_conv_bn_relu(64, 128),
            nn.MaxPool2d((2, 2)),            # H/4,  W/4  → 16×64
            *_conv_bn_relu(128, 256),        #             → 16×64
            *_conv_bn_relu(256, 256),
            nn.MaxPool2d((2, 1)),            # H/8,  W/4  → 8×64
            *_conv_bn_relu(256, 512),        #             → 8×64
            *_conv_bn_relu(512, 512),
            nn.MaxPool2d((2, 1)),            # H/16, W/4  → 4×64
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )

        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, H, W)
        Returns:
            logits: (B, T, num_classes)  T = W // 4
        """
        x = self.cnn(x)           # (B, 512, H/16, W/4)
        x = x.mean(dim=2)         # (B, 512, W/4)  — colapsa H
        x = x.permute(2, 0, 1)    # (T, B, 512)
        x, _ = self.lstm(x)       # (T, B, hidden*2)
        x = self.head(x)          # (T, B, num_classes)
        return x.permute(1, 0, 2) # (B, T, num_classes)
