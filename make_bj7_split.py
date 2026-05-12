"""Gera bj7/split.txt para a estrutura nova do dataset.

- Treino: bj7/train/Scenario-{A,B}/{Brazilian,Mercosur}/track_XXXXX
  Estratificado por (Scenario, Categoria), 80% training / 20% validation.
- Teste:  bj7/test/track_XXXXX (todos como 'testing').

Saída: linhas no formato './caminho/relativo;split', compatível com o leitor
de BJ7Dataset em crnn/svtr/parseq.
"""

from __future__ import annotations

import random
from pathlib import Path

ROOT = Path("bj7")
SEED = 42
TRAIN_FRAC = 0.80

TRAIN_GROUPS = [
    ("Scenario-A", "Brazilian"),
    ("Scenario-A", "Mercosur"),
    ("Scenario-B", "Brazilian"),
    ("Scenario-B", "Mercosur"),
]


def _list_tracks(group_dir: Path) -> list[Path]:
    return sorted(p for p in group_dir.glob("track_*") if p.is_dir())


def _train_val_split(tracks: list[Path], rng: random.Random) -> tuple[list[Path], list[Path]]:
    shuffled = list(tracks)
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * TRAIN_FRAC))
    return shuffled[:n_train], shuffled[n_train:]


def _rel(path: Path) -> str:
    return "./" + path.relative_to(ROOT).as_posix()


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(f"Diretório {ROOT} não encontrado.")

    rng = random.Random(SEED)
    lines: list[str] = []
    counts = {"training": 0, "validation": 0, "testing": 0}

    print(f"Gerando split com seed={SEED}, fração de treino={TRAIN_FRAC}")
    print()

    for scenario, category in TRAIN_GROUPS:
        group_dir = ROOT / "train" / scenario / category
        if not group_dir.exists():
            print(f"  AVISO: {group_dir} não existe, pulando.")
            continue
        tracks = _list_tracks(group_dir)
        train_tracks, val_tracks = _train_val_split(tracks, rng)
        for t in train_tracks:
            lines.append(f"{_rel(t)};training")
        for t in val_tracks:
            lines.append(f"{_rel(t)};validation")
        counts["training"] += len(train_tracks)
        counts["validation"] += len(val_tracks)
        print(
            f"  {scenario}/{category}: total={len(tracks)} | "
            f"train={len(train_tracks)} | val={len(val_tracks)}"
        )

    test_dir = ROOT / "test"
    if test_dir.exists():
        test_tracks = _list_tracks(test_dir)
        for t in test_tracks:
            lines.append(f"{_rel(t)};testing")
        counts["testing"] += len(test_tracks)
        print(f"  test: total={len(test_tracks)}")
    else:
        print(f"  AVISO: {test_dir} não existe.")

    out_path = ROOT / "split.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(
        f"Total: training={counts['training']} | "
        f"validation={counts['validation']} | testing={counts['testing']}"
    )
    print(f"Escrito em: {out_path} ({sum(counts.values())} linhas)")


if __name__ == "__main__":
    main()
