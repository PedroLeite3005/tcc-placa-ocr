"""Fusão de predições por track e entre modelos para o dataset BJ7.

Lê os CSVs de predições gerados por cada modelo (crnn/svtr/parseq) ao final
do treino e produz uma tabela TXT com duas estratégias de fusão:

  - MV   (Voto Majoritário por sequência completa).
         Cada amostra dá 1 voto na string inteira, peso = conf_seq.
  - MVCP (Voto Majoritário por Caractere/Posição).
         Para cada posição da placa, cada amostra contribui com seu caractere
         daquela posição, com peso = conf_chars[posição]. A string final é
         composta posição a posição.

Ambas estratégias produzem:
  - Acurácia HR fundida (entre as 5 imagens HR de cada track).
  - Acurácia LR fundida (entre as 5 imagens LR de cada track).
  - Acurácia da fusão entre modelos (1 voto por modelo, baseado no voto
    combinado HR+LR de cada modelo).

Cada CSV deve ter colunas:
  track_id, image_type, image_idx, gt, pred, conf_seq, conf_chars
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

# Tipo de cada amostra dentro do dict por track:
#   (pred, conf_seq, conf_chars)
Sample = tuple[str, float, list[float]]


# ---------------------------------------------------------------------------
# Parsing de CSV
# ---------------------------------------------------------------------------

def _parse_conf_chars(s: str) -> list[float]:
    """Converte 'p1|p2|p3' em [p1, p2, p3]. Retorna [] se vazio ou inválido."""
    if not s:
        return []
    out: list[float] = []
    for tok in s.split("|"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def load_preds_csv(path: Path) -> dict[str, dict]:
    """Lê CSV e agrupa por track_id.

    Retorna dict: {track_id: {"gt": str,
                              "hr": [(pred, conf_seq, conf_chars), ...],
                              "lr": [(pred, conf_seq, conf_chars), ...]}}
    As listas hr/lr preservam a ordem das imagens (image_idx 1..5).

    Levanta ValueError se o CSV não tiver as colunas 'conf_seq' e 'conf_chars'.
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
            by_track[tid] = {"gt": row["gt"], "hr": [], "lr": []}
        try:
            conf_seq = float(row.get("conf_seq", "0") or "0")
        except ValueError:
            conf_seq = 0.0
        conf_chars = _parse_conf_chars(row.get("conf_chars", ""))
        sample: Sample = (row["pred"], conf_seq, conf_chars)
        by_track[tid][row["image_type"]].append(sample)
    return by_track


# ---------------------------------------------------------------------------
# Estratégia MV: voto majoritário por sequência completa
# ---------------------------------------------------------------------------

def weighted_vote(
    preds_with_weights: list[tuple[str, float]],
) -> tuple[str, float]:
    """Soma pesos por string. Retorna (string vencedora, soma de pesos da vencedora).

    Em caso de empate na soma de pesos, `max` mantém a primeira inserção,
    equivalente ao "primeira ocorrência ganha".
    """
    if not preds_with_weights:
        return "", 0.0
    totals: dict[str, float] = {}
    for s, w in preds_with_weights:
        totals[s] = totals.get(s, 0.0) + float(w)
    winner = max(totals.items(), key=lambda kv: kv[1])
    return winner[0], winner[1]


def _samples_to_seq_votes(samples: list[Sample]) -> list[tuple[str, float]]:
    return [(pred, conf_seq) for pred, conf_seq, _ in samples]


def compute_intra_model_mv(preds_by_track: dict[str, dict]) -> dict[str, float]:
    """Acurácia HR/LR-fundida via MV (voto na sequência completa)."""
    n = len(preds_by_track)
    if n == 0:
        return {"hr_fusion_acc": 0.0, "lr_fusion_acc": 0.0, "n_tracks": 0}

    hr_correct = 0
    lr_correct = 0
    for data in preds_by_track.values():
        gt = data["gt"]
        if data["hr"]:
            winner, _ = weighted_vote(_samples_to_seq_votes(data["hr"]))
            hr_correct += int(winner == gt)
        if data["lr"]:
            winner, _ = weighted_vote(_samples_to_seq_votes(data["lr"]))
            lr_correct += int(winner == gt)

    return {
        "hr_fusion_acc": hr_correct / n,
        "lr_fusion_acc": lr_correct / n,
        "n_tracks": n,
    }


def _combined_track_mv(data: dict) -> tuple[str, float]:
    """Voto MV sobre todas as 10 imagens (HR+LR) de um track."""
    return weighted_vote(
        _samples_to_seq_votes(list(data["hr"]) + list(data["lr"]))
    )


def compute_inter_model_mv(
    all_models_preds: dict[str, dict[str, dict]],
) -> dict[str, float]:
    """Inter-modelos via MV (1 voto MV por modelo, voto entre os modelos)."""
    return _compute_inter_model_generic(all_models_preds, _combined_track_mv)


# ---------------------------------------------------------------------------
# Estratégia MVCP: voto majoritário por caractere/posição
# ---------------------------------------------------------------------------

def weighted_vote_per_char(
    preds_with_char_confs: list[tuple[str, list[float]]],
) -> tuple[str, float]:
    """Voto por posição/caractere, ponderado por conf_chars[posição].

    Recebe: [(pred, [conf_pos_0, conf_pos_1, ...]), ...]
    Retorna: (string composta, soma das probs vencedoras de cada posição)

    O comprimento alvo é a moda dos comprimentos das predições não-vazias.
    Se uma amostra não tem conf_chars[pos] disponível, contribui com peso 0
    naquela posição (não vota nela).
    """
    valid = [(p, c) for p, c in preds_with_char_confs if p]
    if not valid:
        return "", 0.0

    lengths = [len(p) for p, _ in valid]
    target_len = Counter(lengths).most_common(1)[0][0]

    result_chars: list[str] = []
    total_score = 0.0
    for pos in range(target_len):
        votes: dict[str, float] = {}
        for pred, confs in valid:
            if pos >= len(pred):
                continue
            w = confs[pos] if pos < len(confs) else 0.0
            ch = pred[pos]
            votes[ch] = votes.get(ch, 0.0) + float(w)
        if not votes:
            break
        best_char, best_weight = max(votes.items(), key=lambda kv: kv[1])
        result_chars.append(best_char)
        total_score += best_weight

    return "".join(result_chars), total_score


def _samples_to_char_votes(samples: list[Sample]) -> list[tuple[str, list[float]]]:
    return [(pred, conf_chars) for pred, _, conf_chars in samples]


def compute_intra_model_mvcp(preds_by_track: dict[str, dict]) -> dict[str, float]:
    """Acurácia HR/LR-fundida via MVCP (voto por posição)."""
    n = len(preds_by_track)
    if n == 0:
        return {"hr_fusion_acc": 0.0, "lr_fusion_acc": 0.0, "n_tracks": 0}

    hr_correct = 0
    lr_correct = 0
    for data in preds_by_track.values():
        gt = data["gt"]
        if data["hr"]:
            winner, _ = weighted_vote_per_char(_samples_to_char_votes(data["hr"]))
            hr_correct += int(winner == gt)
        if data["lr"]:
            winner, _ = weighted_vote_per_char(_samples_to_char_votes(data["lr"]))
            lr_correct += int(winner == gt)

    return {
        "hr_fusion_acc": hr_correct / n,
        "lr_fusion_acc": lr_correct / n,
        "n_tracks": n,
    }


def _combined_track_mvcp(data: dict) -> tuple[str, float]:
    """Voto MVCP sobre todas as 10 imagens (HR+LR) de um track."""
    return weighted_vote_per_char(
        _samples_to_char_votes(list(data["hr"]) + list(data["lr"]))
    )


def compute_inter_model_mvcp(
    all_models_preds: dict[str, dict[str, dict]],
) -> dict[str, float]:
    """Inter-modelos via MVCP (cada modelo gera string MVCP, voto MV entre modelos)."""
    return _compute_inter_model_generic(all_models_preds, _combined_track_mvcp)


# ---------------------------------------------------------------------------
# Helper genérico para inter-modelos
# ---------------------------------------------------------------------------

def _compute_inter_model_generic(
    all_models_preds: dict[str, dict[str, dict]],
    combiner,
) -> dict[str, float]:
    """Acurácia inter-modelos: cada modelo gera 1 voto via `combiner(data)`,
    e os modelos votam entre si com peso = soma das probs vencedoras.
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
            pred, weight = combiner(all_models_preds[m][tid])
            votes_with_weights.append((pred, weight))
        winner, _ = weighted_vote(votes_with_weights)
        correct += int(winner == gt)

    return {
        "inter_model_acc": correct / len(common_tracks),
        "n_tracks": len(common_tracks),
    }


# ---------------------------------------------------------------------------
# Formatação da tabela
# ---------------------------------------------------------------------------

def _format_section(
    title: str,
    inter_caption: str,
    intra: dict[str, dict[str, float]],
    inter: dict[str, float],
) -> list[str]:
    lines: list[str] = []
    lines.append(f"== {title} ==")
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
        lines.append(inter_caption)
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
    return lines


def format_table(
    intra_mv: dict[str, dict[str, float]],
    inter_mv: dict[str, float],
    intra_mvcp: dict[str, dict[str, float]],
    inter_mvcp: dict[str, float],
) -> str:
    """Gera o TXT formatado com as duas seções (MV e MVCP, ambas ponderadas por confianca)."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("BJ7 - Fusao por track (voto ponderado por confianca)")
    lines.append("=" * 80)
    lines.append("")

    lines.extend(
        _format_section(
            title="MV (Voto Majoritario por sequencia, peso = conf_seq)",
            inter_caption=(
                "Fusao inter-modelos (1 voto MV por modelo, "
                "peso = soma de conf_seq):"
            ),
            intra=intra_mv,
            inter=inter_mv,
        )
    )

    lines.extend(
        _format_section(
            title="MVCP (Voto Majoritario por Caractere/Posicao, peso = conf_chars[pos])",
            inter_caption=(
                "Fusao inter-modelos (1 voto MVCP por modelo, "
                "peso = soma de conf_chars vencedores):"
            ),
            intra=intra_mvcp,
            inter=inter_mvcp,
        )
    )

    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------

def run_fusion(
    models: list[str],
    dataset_name: str,
    logs_dir: Path,
) -> Path | None:
    """Lê CSVs disponíveis, calcula fusões MV e MVCP (ponderadas por confianca),
    e escreve a tabela TXT.

    Retorna o caminho do arquivo gerado, ou None se nenhum CSV foi encontrado.
    """
    if dataset_name != "bj7":
        return None

    all_preds: dict[str, dict[str, dict]] = {}
    intra_mv_results: dict[str, dict[str, float]] = {}
    intra_mvcp_results: dict[str, dict[str, float]] = {}

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
        intra_mv_results[model] = compute_intra_model_mv(preds_by_track)
        intra_mvcp_results[model] = compute_intra_model_mvcp(preds_by_track)
        print(
            f"[fusion] {model}: "
            f"MV   HR={intra_mv_results[model]['hr_fusion_acc']:.4f} "
            f"LR={intra_mv_results[model]['lr_fusion_acc']:.4f} | "
            f"MVCP HR={intra_mvcp_results[model]['hr_fusion_acc']:.4f} "
            f"LR={intra_mvcp_results[model]['lr_fusion_acc']:.4f} "
            f"({intra_mv_results[model]['n_tracks']} tracks)"
        )

    if not intra_mv_results:
        print("[fusion] Nenhum CSV encontrado. Tabela não foi gerada.")
        return None

    if len(all_preds) >= 2:
        inter_mv_results = compute_inter_model_mv(all_preds)
        inter_mvcp_results = compute_inter_model_mvcp(all_preds)
        print(
            f"[fusion] inter-modelos: "
            f"MV={inter_mv_results['inter_model_acc']:.4f} | "
            f"MVCP={inter_mvcp_results['inter_model_acc']:.4f} "
            f"({inter_mv_results['n_tracks']} tracks)"
        )
    else:
        inter_mv_results = {"inter_model_acc": 0.0, "n_tracks": 0}
        inter_mvcp_results = {"inter_model_acc": 0.0, "n_tracks": 0}

    table = format_table(
        intra_mv_results, inter_mv_results, intra_mvcp_results, inter_mvcp_results
    )
    out_path = logs_dir / f"{dataset_name}_fusion_table.txt"
    out_path.write_text(table, encoding="utf-8")
    print(f"[fusion] Tabela salva em: {out_path}")
    return out_path


# Aliases retrocompatíveis (caso main.py ou outro código importe os nomes antigos)
compute_intra_model = compute_intra_model_mv
compute_inter_model = compute_inter_model_mv
_combined_track_vote = _combined_track_mv


if __name__ == "__main__":
    run_fusion(
        models=["svtr", "crnn", "parseq"],
        dataset_name="bj7",
        logs_dir=Path("logs"),
    )
