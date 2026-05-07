from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths (override for local vs Kaggle)
    data_root: Path = Path("/kaggle/input/birdclef-2026")
    train_audio_dir: Path = field(init=False)
    train_soundscapes_dir: Path = field(init=False)
    test_soundscapes_dir: Path = field(init=False)
    train_csv: Path = field(init=False)
    train_soundscapes_labels_csv: Path = field(init=False)
    taxonomy_csv: Path = field(init=False)
    sample_submission_csv: Path = field(init=False)

    work_dir: Path = Path("./work")
    ckpt_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    # Audio
    sample_rate: int = 32000
    clip_seconds: float = 5.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: int = 50
    fmax: int = 16000
    top_db: float = 80.0

    # Model
    backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    pretrained: bool = True
    drop_rate: float = 0.2
    drop_path_rate: float = 0.2
    in_channels: int = 1

    # Training
    seed: int = 42
    folds: int = 5
    fold: int = 0
    epochs: int = 25
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.5
    label_smoothing: float = 0.005
    grad_clip: float = 5.0
    use_amp: bool = True

    # Inference
    test_batch_size: int = 16
    tta: bool = False
    threshold: float = 0.0  # we submit raw probabilities

    def __post_init__(self) -> None:
        self.train_audio_dir = self.data_root / "train_audio"
        self.train_soundscapes_dir = self.data_root / "train_soundscapes"
        self.test_soundscapes_dir = self.data_root / "test_soundscapes"
        self.train_csv = self.data_root / "train.csv"
        self.train_soundscapes_labels_csv = self.data_root / "train_soundscapes_labels.csv"
        self.taxonomy_csv = self.data_root / "taxonomy.csv"
        self.sample_submission_csv = self.data_root / "sample_submission.csv"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.work_dir / "ckpt"
        self.cache_dir = self.work_dir / "cache"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)
