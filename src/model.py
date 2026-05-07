from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .audio import MelSpectrogram, SpecAugment


class AudioCNN(nn.Module):
    """timm CNN backbone over a mel-spectrogram, multi-label classification head."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool,
        drop_rate: float,
        drop_path_rate: float,
        in_channels: int,
        sr: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        fmin: int,
        fmax: int,
        top_db: float,
    ) -> None:
        super().__init__()
        self.mel = MelSpectrogram(sr, n_fft, hop_length, n_mels, fmin, fmax, top_db)
        self.spec_aug = SpecAugment()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T)
        spec = self.mel(wav).unsqueeze(1)  # (B, 1, n_mels, T')
        spec = self.spec_aug(spec)
        feats = self.backbone.forward_features(spec)
        if feats.dim() == 3:  # transformer-style
            feats = feats.mean(dim=1)
        else:
            feats = self.pool(feats).flatten(1)
        return self.head(feats)
