"""
Recorta as placas do dataset RodoSol usando warp de perspectiva.

Para cada imagem em rodosol/images/{categoria}/, lê o .txt correspondente,
aplica uma homografia com os 4 corners anotados e salva o crop 256×64 em
rodosol/crops/{categoria}/.

Uso:
    python crop_rodosol.py
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

WARP_W = 256
WARP_H = 64
INPUT_ROOT = Path("rodosol/images")
OUTPUT_ROOT = Path("rodosol/crops")
CATEGORIES = ["cars-br", "cars-me", "motorcycles-br", "motorcycles-me"]

DST_PTS = np.array(
    [
        [0, 0],
        [WARP_W - 1, 0],
        [WARP_W - 1, WARP_H - 1],
        [0, WARP_H - 1],
    ],
    dtype=np.float32,
)


def parse_corners(value: str) -> np.ndarray:
    """Converte 'x1,y1 x2,y2 x3,y3 x4,y4' em array (4,2) float32.

    Ordem esperada: TL → TR → BR → BL (sentido horário).
    """
    pairs = value.strip().split()
    if len(pairs) != 4:
        raise ValueError(f"Esperados 4 corners, encontrados {len(pairs)}: {value!r}")
    pts = []
    for p in pairs:
        x, y = p.split(",")
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.float32)


def parse_txt(txt_path: Path) -> np.ndarray:
    """Lê o arquivo de anotação e retorna os corners como ndarray."""
    data: dict[str, str] = {}
    with open(txt_path) as f:
        for raw in f:
            key, _, val = raw.partition(":")
            data[key.strip()] = val.strip()
    if "corners" not in data:
        raise KeyError(f"Campo 'corners' ausente em {txt_path}")
    return parse_corners(data["corners"])


def process_one(txt_path: Path) -> str | None:
    """Processa uma imagem. Retorna None em caso de sucesso ou mensagem de erro."""
    try:
        src_pts = parse_txt(txt_path)
        jpg_path = txt_path.with_suffix(".jpg")
        img = cv2.imread(str(jpg_path))
        if img is None:
            return f"SKIP (imagem inválida): {jpg_path}"

        M = cv2.getPerspectiveTransform(src_pts, DST_PTS)
        warped = cv2.warpPerspective(
            img,
            M,
            (WARP_W, WARP_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        out_path = OUTPUT_ROOT / txt_path.parent.name / jpg_path.name
        cv2.imwrite(str(out_path), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return None
    except Exception as e:
        return f"SKIP ({e}): {txt_path}"


def main() -> None:
    for cat in CATEGORIES:
        (OUTPUT_ROOT / cat).mkdir(parents=True, exist_ok=True)

    all_txts = sorted(
        p for cat in CATEGORIES for p in (INPUT_ROOT / cat).glob("*.txt")
    )
    total = len(all_txts)
    print(f"Total de imagens a processar: {total}")

    errors: list[str] = []
    with Pool(os.cpu_count()) as pool:
        for i, err in enumerate(
            pool.imap_unordered(process_one, all_txts, chunksize=50), 1
        ):
            if err:
                errors.append(err)
            if i % 500 == 0 or i == total:
                print(f"[{i}/{total}] erros até agora: {len(errors)}")

    print(f"\nConcluído: {total - len(errors)} ok | {len(errors)} erros")
    for e in errors:
        print(e)


if __name__ == "__main__":
    main()
