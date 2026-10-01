"""Dataset classes para PARSeq com RodoSol e BJ7."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _strip_prefix(s: str, prefix: str) -> str:
    """Equivalente a str.removeprefix (Python 3.9+) — Jetson usa Python 3.6."""
    return s[len(prefix):] if s.startswith(prefix) else s


def _otsu_binarize(img: Image.Image) -> Image.Image:
    """Grayscale + binarização por limiar de Otsu (adaptativo por imagem).

    Pré-processamento mais leve pra Jetson: reduz a entrada de 3 canais (RGB)
    pra 1 canal real preto/branco, não apenas grayscale replicado.
    """
    arr = np.array(img.convert("L"))
    _, binarized = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized)


def _tail_transform_steps(binarize: bool) -> list:
    """Últimos passos do pipeline: binariza (1 canal) ou normaliza em RGB (3 canais)."""
    if binarize:
        return [
            transforms.Lambda(_otsu_binarize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


def _sanitize_plate(text: str) -> str:
    """Normaliza placa: uppercase + remove tudo que não é alfanumérico.

    Cobre espaços, hífens, quebras de linha, e qualquer outro lixo do JSON
    que quebraria o tokenizer do PARSeq (KeyError no vocabulário).
    """
    return "".join(c for c in text.upper() if c.isalnum())


def make_transform(w: int = 256, h: int = 64, binarize: bool = False) -> transforms.Compose:
    """Transformação padrão para entrada do PARSeq (val/test)."""
    return transforms.Compose(
        [transforms.Resize((h, w))] + _tail_transform_steps(binarize)
    )


def make_transform_train(w: int = 256, h: int = 64, binarize: bool = False) -> transforms.Compose:
    """Transformação de treino com augmentations leves para placas.

    `binarize=True`: as augmentations de cor/blur são aplicadas antes da
    binarização (simula variação real de iluminação/foco que existiria antes
    de um binarizador de verdade — a placa já sai preto/branco pro modelo).
    """
    return transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
            ),
            transforms.RandomAffine(
                degrees=2,
                translate=(0.02, 0.05),
                scale=(0.95, 1.05),
                fill=0,
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ] + _tail_transform_steps(binarize)
    )


class RodoSolDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_path: Path,
        split: str,
        transform=None,
    ) -> None:
        self.transform = transform or make_transform()
        self.samples = []  # type: List[Tuple[Path, str]]

        with open(split_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, s = line.split(";")
                if s != split:
                    continue

                parts = Path(_strip_prefix(rel, "./"))
                category = parts.parts[1]
                stem = parts.stem

                crop_path = data_root / "crops" / category / f"{stem}.jpg"
                txt_path = data_root / "images" / category / f"{stem}.txt"

                plate = _read_plate_rodosol(txt_path)
                if plate and crop_path.exists():
                    self.samples.append((crop_path, plate))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, plate = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, plate


class BJ7Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_path: Path,
        split: str,
        transform=None,
    ) -> None:
        self.transform = transform or make_transform()
        self.samples = []  # type: List[Tuple[Path, str]]
        self.metadata = []  # type: List[dict]

        with open(split_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, s = line.split(";")
                if s != split:
                    continue

                track_dir = data_root / _strip_prefix(rel, "./")
                ann_path = track_dir / "annotations.json"
                if not ann_path.exists():
                    continue

                try:
                    with open(ann_path, encoding="utf-8") as f2:
                        ann = json.load(f2)
                    plate = _sanitize_plate(ann.get("plate_text", ""))
                    if not plate:
                        continue
                except Exception:
                    continue

                ext = ".png" if (track_dir / "hr-001.png").exists() else ".jpg"
                track_id = track_dir.name
                for prefix in ("hr", "lr"):
                    for i in range(1, 6):
                        img_path = track_dir / f"{prefix}-{i:03d}{ext}"
                        if img_path.exists():
                            self.samples.append((img_path, plate))
                            self.metadata.append(
                                {
                                    "track_id": track_id,
                                    "image_type": prefix,
                                    "image_idx": i,
                                }
                            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, plate = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, plate


def _read_plate_rodosol(txt_path: Path) -> Optional[str]:
    try:
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                key, _, val = line.partition(":")
                if key.strip() == "plate":
                    return _sanitize_plate(val) or None
    except Exception:
        pass
    return None


def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)

