"""Fusão de predições por track e entre modelos para o dataset BJ7.

Lê os CSVs de predições gerados por cada modelo (crnn/svtr/parseq) ao final
do treino e produz uma tabela TXT com:
  - Acurácia HR fundida (voto majoritário entre as 5 imagens HR de cada track).
  - Acurácia LR fundida (voto majoritário entre as 5 imagens LR de cada track).
  - Acurácia da fusão entre modelos (1 voto por modelo, baseado no voto
    combinado HR+LR de cada modelo).

Cada CSV deve ter colunas: track_id, image_type, image_idx, gt, pred.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def majority_vote(preds: list[str]) -> str:
    """Retorna a string mais votada. Em caso de empate, vence a primeira ocorrência.

    Counter.most_common é estável quanto à ordem de inserção em Python 3.7+.
    Se ['bbb7777', 'eee7777', 'bbb7777', 'eee7777', 'bbb7777'] entra,
    a saída é 'bbb7777' (3 votos contra 2).
    """
    if not preds:
        return ""
    counter = Counter(preds)
    return counter.most_common(1)[0][0]


def load_preds_csv(path: Path) -> dict[str, dict]:
    """Lê CSV e agrupa por track_id.

    Retorna dict: {track_id: {"gt": str, "hr": [str, ...], "lr": [str, ...]}}
    As listas hr/lr preservam a ordem das imagens (image_idx 1..5).
    """
    by_track: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows.sort(key=lambda r: (r["track_id"], r["image_type"], int(r["image_idx"])))

    for row in rows:
        tid = row["track_id"]
        if tid not in by_track:
            by_track[tid] = {"gt": row["gt"], "hr": [], "lr": []}
        by_track[tid][row["image_type"]].append(row["pred"])
    return by_track


def compute_intra_model(preds_by_track: dict[str, dict]) -> dict[str, float]:
    """Calcula acurácia HR-fundida e LR-fundida sobre os tracks."""
    n = len(preds_by_track)
    if n == 0:
        return {"hr_fusion_acc": 0.0, "lr_fusion_acc": 0.0, "n_tracks": 0}

    hr_correct = 0
    lr_correct = 0
    for data in preds_by_track.values():
        gt = data["gt"]
        if data["hr"]:
            hr_correct += int(majority_vote(data["hr"]) == gt)
        if data["lr"]:
            lr_correct += int(majority_vote(data["lr"]) == gt)

    return {
        "hr_fusion_acc": hr_correct / n,
        "lr_fusion_acc": lr_correct / n,
        "n_tracks": n,
    }


def _combined_track_pred(data: dict) -> str:
    """Voto majoritário sobre todas as 10 imagens (HR+LR) de um track."""
    return majority_vote(list(data["hr"]) + list(data["lr"]))


def compute_inter_model(
    all_models_preds: dict[str, dict[str, dict]],
) -> dict[str, float]:
    """Acurácia da fusão entre modelos (1 voto por modelo, voto combinado HR+LR)."""
    if not all_models_preds:
        return {"inter_model_acc": 0.0, "n_tracks": 0}

    # Interseção dos tracks que todos os modelos avaliaram.
    track_sets = [set(p.keys()) for p in all_models_preds.values()]
    common_tracks = set.intersection(*track_sets) if track_sets else set()

    if not common_tracks:
        return {"inter_model_acc": 0.0, "n_tracks": 0}

    correct = 0
    for tid in common_tracks:
        # GT deve ser o mesmo entre modelos; pega do primeiro.
        first_model = next(iter(all_models_preds))
        gt = all_models_preds[first_model][tid]["gt"]
        votes = [
            _combined_track_pred(all_models_preds[m][tid])
            for m in all_models_preds
        ]
        correct += int(majority_vote(votes) == gt)

    return {
        "inter_model_acc": correct / len(common_tracks),
        "n_tracks": len(common_tracks),
    }


def format_table(
    intra: dict[str, dict[str, float]],
    inter: dict[str, float],
) -> str:
    """Gera o TXT formatado com a tabela de resultados."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("BJ7 - Fusao por track (voto majoritario)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Acuracia de sequencia (1 predicao por track)")
    lines.append("")
    lines.append("Modelo   | HR fusion | LR fusion | Tracks")
    lines.append("---------|-----------|-----------|-------")
    for model_name, stats in intra.items():
        lines.append(
            f"{model_name:<8} |   {stats['hr_fusion_acc']:.4f}  |"
            f"   {stats['lr_fusion_acc']:.4f}  |  {stats['n_tracks']}"
        )
    lines.append("")

    if inter["n_tracks"] > 0 and len(intra) >= 2:
        lines.append(
            "Fusao inter-modelos (1 voto por modelo, voto combinado HR+LR):"
        )
        lines.append(
            f"  Acuracia por track: {inter['inter_model_acc']:.4f} "
            f"({inter['n_tracks']} tracks em comum)"
        )
    else:
        lines.append(
            "Fusao inter-modelos: nao calculada "
            "(precisa de >=2 modelos com CSVs disponiveis)."
        )
    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def run_fusion(
    models: list[str],
    dataset_name: str,
    logs_dir: Path,
) -> Path | None:
    """Lê CSVs disponíveis, calcula fusões e escreve a tabela TXT.

    Retorna o caminho do arquivo gerado, ou None se nenhum CSV foi encontrado.
    """
    if dataset_name != "bj7":
        return None

    all_preds: dict[str, dict[str, dict]] = {}
    intra_results: dict[str, dict[str, float]] = {}

    for model in models:
        csv_path = logs_dir / f"{dataset_name}_{model}" / (
            f"{dataset_name}_{model}_preds.csv"
        )
        if not csv_path.exists():
            print(f"[fusion] Aviso: CSV de {model} não encontrado em {csv_path}.")
            continue
        preds_by_track = load_preds_csv(csv_path)
        all_preds[model] = preds_by_track
        intra_results[model] = compute_intra_model(preds_by_track)
        print(
            f"[fusion] {model}: HR={intra_results[model]['hr_fusion_acc']:.4f} "
            f"LR={intra_results[model]['lr_fusion_acc']:.4f} "
            f"({intra_results[model]['n_tracks']} tracks)"
        )

    if not intra_results:
        print("[fusion] Nenhum CSV encontrado. Tabela não foi gerada.")
        return None

    inter_results = compute_inter_model(all_preds) if len(all_preds) >= 2 else {
        "inter_model_acc": 0.0,
        "n_tracks": 0,
    }
    if len(all_preds) >= 2:
        print(
            f"[fusion] inter-modelos: {inter_results['inter_model_acc']:.4f} "
            f"({inter_results['n_tracks']} tracks)"
        )

    table = format_table(intra_results, inter_results)
    out_path = logs_dir / f"{dataset_name}_fusion_table.txt"
    out_path.write_text(table, encoding="utf-8")
    print(f"[fusion] Tabela salva em: {out_path}")
    return out_path
