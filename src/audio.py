from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio


def load_audio(path: str, sr: int) -> np.ndarray:
    wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        t = torch.from_numpy(wav).unsqueeze(0)
        t = torchaudio.functional.resample(t, file_sr, sr)
        wav = t.squeeze(0).numpy()
    return wav.astype(np.float32, copy=False)


def crop_or_pad(wav: np.ndarray, target_len: int, training: bool, rng: np.random.Generator | None = None) -> np.ndarray:
    n = wav.shape[0]
    if n == target_len:
        return wav
    if n < target_len:
        pad = target_len - n
        if training and rng is not None:
            left = int(rng.integers(0, pad + 1))
        else:
            left = pad // 2
        return np.pad(wav, (left, pad - left), mode="constant")
    if training and rng is not None:
        start = int(rng.integers(0, n - target_len + 1))
    else:
        start = (n - target_len) // 2
    return wav[start : start + target_len]


class MelSpectrogram(nn.Module):
    def __init__(self, sr: int, n_fft: int, hop_length: int, n_mels: int, fmin: int, fmax: int, top_db: float) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.mel(wav)
        # silent input → power 0 → log(0) = -inf; add epsilon to keep things finite
        spec = self.to_db(spec.clamp(min=1e-10))
        spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
        # per-sample standardization
        mean = spec.mean(dim=(-1, -2), keepdim=True)
        std = spec.std(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        return (spec - mean) / std


class SpecAugment(nn.Module):
    def __init__(self, freq_mask: int = 24, time_mask: int = 40, num_masks: int = 2, p: float = 0.5) -> None:
        super().__init__()
        self.freq = torchaudio.transforms.FrequencyMasking(freq_mask)
        self.time = torchaudio.transforms.TimeMasking(time_mask)
        self.num_masks = num_masks
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return spec
        for _ in range(self.num_masks):
            spec = self.freq(spec)
            spec = self.time(spec)
        return spec
