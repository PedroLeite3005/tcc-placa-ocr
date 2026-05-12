"""Fusão de predições por track e entre modelos para o dataset BJ7.

Lê os CSVs de predições gerados por cada modelo (crnn/svtr/parseq) ao final
do treino e produz uma tabela TXT com:
  - Acurácia HR fundida (voto ponderado por confiança entre as 5 imagens HR de cada track).
  - Acurácia LR fundida (voto ponderado por confiança entre as 5 imagens LR de cada track).
  - Acurácia da fusão entre modelos (1 voto ponderado por modelo, peso = soma de
    confianças do voto combinado HR+LR).

Cada CSV deve ter colunas:
  track_id, image_type, image_idx, gt, pred, conf_seq, conf_chars
"""

from __future__ import annotations

import csv
from pathlib import Path


def weighted_vote(
    preds_with_weights: list[tuple[str, float]],
) -> tuple[str, float]:
    """Soma pesos por string e retorna (string vencedora, soma_de_pesos_da_vencedora).

    Em caso de empate na soma de pesos (raro com floats), `max` mantém a primeira
    inserção, equivalente ao "primeira ocorrência ganha".
    """
    if not preds_with_weights:
        return "", 0.0
    totals: dict[str, float] = {}
    for s, w in preds_with_weights:
        totals[s] = totals.get(s, 0.0) + float(w)
    winner = max(totals.items(), key=lambda kv: kv[1])
    return winner[0], winner[1]


def load_preds_csv(path: Path) -> dict[str, dict]:
    """Lê CSV e agrupa por track_id.

    Retorna dict: {track_id: {"gt": str,
                              "hr": [(pred, conf_seq), ...],
                              "lr": [(pred, conf_seq), ...],
                              "hr_chars": [str, ...],
                              "lr_chars": [str, ...]}}
    As listas hr/lr preservam a ordem das imagens (image_idx 1..5).

    Levanta ValueError se o CSV não tiver a coluna 'conf_seq' (formato antigo).
    """
    by_track: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "conf_seq" not in reader.fieldnames:
            raise ValueError(
                f"CSV {path} esta em formato antigo (sem coluna 'conf_seq'). "
                f"Apague o arquivo e regere com a versao atual de dump_test_predictions."
            )
        rows = list(reader)

    rows.sort(key=lambda r: (r["track_id"], r["image_type"], int(r["image_idx"])))

    for row in rows:
        tid = row["track_id"]
        if tid not in by_track:
            by_track[tid] = {
                "gt": row["gt"],
                "hr": [],
                "lr": [],
                "hr_chars": [],
                "lr_chars": [],
            }
        try:
            conf_seq = float(row.get("conf_seq", "0") or "0")
        except ValueError:
            conf_seq = 0.0
        by_track[tid][row["image_type"]].append((row["pred"], conf_seq))
        by_track[tid][f"{row['image_type']}_chars"].append(
            row.get("conf_chars", "") or ""
        )
    return by_track


def compute_intra_model(preds_by_track: dict[str, dict]) -> dict[str, float]:
    """Calcula acurácia HR-fundida e LR-fundida (voto ponderado por conf_seq)."""
    n = len(preds_by_track)
    if n == 0:
        return {"hr_fusion_acc": 0.0, "lr_fusion_acc": 0.0, "n_tracks": 0}

    hr_correct = 0
    lr_correct = 0
    for data in preds_by_track.values():
        gt = data["gt"]
        if data["hr"]:
            winner, _ = weighted_vote(data["hr"])
            hr_correct += int(winner == gt)
        if data["lr"]:
            winner, _ = weighted_vote(data["lr"])
            lr_correct += int(winner == gt)

    return {
        "hr_fusion_acc": hr_correct / n,
        "lr_fusion_acc": lr_correct / n,
        "n_tracks": n,
    }


def _combined_track_vote(data: dict) -> tuple[str, float]:
    """Voto ponderado sobre todas as imagens (HR+LR) de um track.

    Retorna (string vencedora, soma de pesos da vencedora).
    """
    return weighted_vote(list(data["hr"]) + list(data["lr"]))


def compute_inter_model(
    all_models_preds: dict[str, dict[str, dict]],
) -> dict[str, float]:
    """Acurácia da fusão entre modelos (1 voto ponderado por modelo).

    Cada modelo gera 1 voto por track via voto ponderado das 10 imagens (HR+LR).
    O peso desse voto agregado é a soma dos pesos da string vencedora.
    Os modelos votam (ponderado) entre si com esses pesos.
    """
    if not all_models_preds:
        return {"inter_model_acc": 0.0, "n_tracks": 0}

    track_sets = [set(p.keys()) for p in all_models_preds.values()]
    common_tracks = set.intersection(*track_sets) if track_sets else set()

    if not common_tracks:
        return {"inter_model_acc": 0.0, "n_tracks": 0}

    correct = 0
    for tid in common_tracks:
        first_model = next(iter(all_models_preds))
        gt = all_models_preds[first_model][tid]["gt"]
        votes_with_weights: list[tuple[str, float]] = []
        for m in all_models_preds:
            pred, weight = _combined_track_vote(all_models_preds[m][tid])
            votes_with_weights.append((pred, weight))
        winner, _ = weighted_vote(votes_with_weights)
        correct += int(winner == gt)

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
    lines.append("BJ7 - Fusao por track (voto ponderado por confianca)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Acuracia de sequencia (1 predicao por track, voto ponderado por conf_seq)")
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
            "Fusao inter-modelos (1 voto ponderado por modelo, peso = soma de conf_seq):"
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
    """Lê CSVs disponíveis, calcula fusões ponderadas e escreve a tabela TXT.

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
        try:
            preds_by_track = load_preds_csv(csv_path)
        except ValueError as exc:
            print(f"[fusion] Aviso: pulando {model} - {exc}")
            continue
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
