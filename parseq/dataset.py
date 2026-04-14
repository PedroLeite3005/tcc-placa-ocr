"""Dataset classes para PARSeq com RodoSol e BJ7."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_transform(w: int = 128, h: int = 32) -> transforms.Compose:
    """Transformação padrão para entrada do PARSeq."""
    return transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
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
        self.samples: list[tuple[Path, str]] = []

        with open(split_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, s = line.split(";")
                if s != split:
                    continue

                parts = Path(rel.removeprefix("./"))
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
        self.samples: list[tuple[Path, str]] = []

        with open(split_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, s = line.split(";")
                if s != split:
                    continue

                track_dir = data_root / rel.removeprefix("./")
                ann_path = track_dir / "annotations.json"
                if not ann_path.exists():
                    continue

                try:
                    with open(ann_path, encoding="utf-8") as f2:
                        ann = json.load(f2)
                    plate = ann.get("plate_text", "").upper()
                    if not plate:
                        continue
                except Exception:
                    continue

                ext = ".png" if (track_dir / "hr-001.png").exists() else ".jpg"
                for prefix in ("hr", "lr"):
                    for i in range(1, 6):
                        img_path = track_dir / f"{prefix}-{i:03d}{ext}"
                        if img_path.exists():
                            self.samples.append((img_path, plate))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, plate = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, plate


def _read_plate_rodosol(txt_path: Path) -> str | None:
    try:
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                key, _, val = line.partition(":")
                if key.strip() == "plate":
                    return val.strip().upper()
    except Exception:
        pass
    return None


def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)

