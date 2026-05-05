"""Dataset classes para RodoSol e BJ7."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Conjunto de caracteres: blank=0, '0'=1 … '9'=10, 'A'=11 … 'Z'=36
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # 37 (inclui blank)


def encode(plate: str) -> list[int]:
    return [CHAR2IDX[c] for c in plate.upper() if c in CHAR2IDX]


def decode(indices: list[int]) -> str:
    return "".join(IDX2CHAR[i] for i in indices if i in IDX2CHAR)


def make_transform(w: int = 256, h: int = 64) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )


class RodoSolDataset(Dataset):
    """Lê crops de placas do RodoSol usando o split.txt.

    Estrutura esperada em data_root:
      split.txt           (ou passado via split_path)
      images/{cat}/img_XXXXX.txt   ← label
      crops/{cat}/img_XXXXX.jpg    ← imagem de entrada
    """

    def __init__(
        self,
        data_root: Path,
        split_path: Path,
        split: str,
        transform=None,
    ) -> None:
        self.transform = transform or make_transform()
        self.samples: list[tuple[Path, str]] = []

        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, s = line.split(";")
                if s != split:
                    continue

                # rel: ./images/cars-br/img_000003.jpg
                parts = Path(rel.removeprefix("./"))  # images/cars-br/img_000003.jpg
                category = parts.parts[1]             # cars-br
                stem = parts.stem                     # img_000003

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
        label = torch.tensor(encode(plate), dtype=torch.long)
        return img, label


class BJ7Dataset(Dataset):
    """Lê crops de placas do BJ7 usando o split.txt.

    Cada track gera 10 amostras: hr-001..hr-005 + lr-001..lr-005.
    Estrutura esperada em data_root:
      split.txt            (ou passado via split_path)
      Scenario-A/Brazilian/track_XXXXX/annotations.json
      Scenario-A/Brazilian/track_XXXXX/hr-001.png  (ou .jpg no Scenario-B)
      ...
    """

    def __init__(
        self,
        data_root: Path,
        split_path: Path,
        split: str,
        transform=None,
    ) -> None:
        self.transform = transform or make_transform()
        self.samples: list[tuple[Path, str]] = []
        self.metadata: list[dict] = []

        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, s = line.split(";")
                if s != split:
                    continue

                # rel: ./Scenario-A/Brazilian/track_01384
                track_dir = data_root / rel.removeprefix("./")
                ann_path = track_dir / "annotations.json"
                if not ann_path.exists():
                    continue

                try:
                    with open(ann_path) as f2:
                        ann = json.load(f2)
                    plate = ann.get("plate_text", "").upper()
                    if not plate:
                        continue
                except Exception:
                    continue

                # Detecta extensão (Scenario-A → PNG, Scenario-B → JPG)
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
        label = torch.tensor(encode(plate), dtype=torch.long)
        return img, label


def _read_plate_rodosol(txt_path: Path) -> str | None:
    try:
        with open(txt_path) as f:
            for line in f:
                key, _, val = line.partition(":")
                if key.strip() == "plate":
                    return val.strip().upper()
    except Exception:
        pass
    return None


def collate_fn(batch):
    """Mantém labels como lista de tensors (necessário para CTCLoss)."""
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)
