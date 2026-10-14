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

import argparse
import csv
from collections import Counter
from itertools import combinations
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
    inter: dict[str, dict[str, float]],
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
    if inter:
        lines.append(inter_caption)
        for combo, res in inter.items():
            lines.append(
                f"  {combo:<20} acuracia por track: {res['inter_model_acc']:.4f} "
                f"({res['n_tracks']} tracks em comum)"
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
    inter_mv: dict[str, dict[str, float]],
    intra_mvcp: dict[str, dict[str, float]],
    inter_mvcp: dict[str, dict[str, float]],
) -> str:
    """Gera o TXT com as duas seções (MV e MVCP). `inter_*` mapeia cada
    subconjunto de modelos (ensemble, ex: "crnn+svtr") para seu resultado."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("BJ7 - Fusao por track (voto ponderado por confianca)")
    lines.append("=" * 80)
    lines.append("")
    lines.extend(
        _format_section(
            title="MV (Voto Majoritario por sequencia, peso = conf_seq)",
            inter_caption="Ensemble / fusao inter-modelos (1 voto MV por modelo, peso = soma de conf_seq):",
            intra=intra_mv,
            inter=inter_mv,
        )
    )
    lines.extend(
        _format_section(
            title="MVCP (Voto Majoritario por Caractere/Posicao, peso = conf_chars[pos])",
            inter_caption="Ensemble / fusao inter-modelos (1 voto MVCP por modelo, peso = soma de conf_chars vencedores):",
            intra=intra_mvcp,
            inter=inter_mvcp,
        )
    )
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def _write_summary_csv(
    path: Path,
    csv_paths: dict[str, Path],
    intra_mv: dict[str, dict[str, float]],
    intra_mvcp: dict[str, dict[str, float]],
    inter_mv: dict[str, dict[str, float]],
    inter_mvcp: dict[str, dict[str, float]],
) -> None:
    """Resumo legível por máquina (lido pelo report.py): 1 linha por modelo
    (fusão intra, sobre as imagens LR do track) e 1 por ensemble (inter-modelos)."""
    fields = ["config", "kind", "members", "preds_csvs", "mv_acc", "mvcp_acc", "n_tracks"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in intra_mv:
            w.writerow({
                "config": name, "kind": "intra", "members": name,
                "preds_csvs": csv_paths[name].as_posix(),
                "mv_acc": f"{intra_mv[name]['lr_fusion_acc']:.4f}",
                "mvcp_acc": f"{intra_mvcp[name]['lr_fusion_acc']:.4f}",
                "n_tracks": intra_mv[name]["n_tracks"],
            })
        for combo in inter_mv:
            members = combo.split("+")
            w.writerow({
                "config": combo, "kind": "ensemble", "members": ";".join(members),
                "preds_csvs": ";".join(csv_paths[m].as_posix() for m in members),
                "mv_acc": f"{inter_mv[combo]['inter_model_acc']:.4f}",
                "mvcp_acc": f"{inter_mvcp[combo]['inter_model_acc']:.4f}",
                "n_tracks": inter_mv[combo]["n_tracks"],
            })


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------

def run_fusion(
    models: list[str],
    dataset_name: str,
    logs_dir: Path,
    csv_paths: dict[str, Path] | None = None,
    out_path: Path | None = None,
    summary_csv: Path | None = None,
) -> Path | None:
    """Lê CSVs de predições, calcula fusões MV e MVCP (intra-modelo) e o
    ensemble (inter-modelos) para todo subconjunto de >=2 modelos, e escreve
    a tabela TXT (+ resumo CSV opcional).

    Sem `csv_paths`: usa o caminho padrão `logs/{dataset}_{model}/{dataset}_{model}_preds.csv`
    (comportamento do main.py). Com `csv_paths` ({nome: caminho}): qualquer
    CSV — ex: predições geradas na Jetson, ou variantes binarizadas/comprimidas.

    Retorna o caminho da tabela gerada, ou None se nenhum CSV foi encontrado.
    """
    if csv_paths is None:
        if dataset_name != "bj7":
            return None
        csv_paths = {
            m: logs_dir / f"{dataset_name}_{m}" / f"{dataset_name}_{m}_preds.csv"
            for m in models
        }

    all_preds: dict[str, dict[str, dict]] = {}
    intra_mv_results: dict[str, dict[str, float]] = {}
    intra_mvcp_results: dict[str, dict[str, float]] = {}

    for name, csv_path in csv_paths.items():
        if not csv_path.exists():
            print(f"[fusion] Aviso: CSV de {name} não encontrado em {csv_path}.")
            continue
        try:
            preds_by_track = load_preds_csv(csv_path)
        except ValueError as exc:
            print(f"[fusion] Aviso: pulando {name} - {exc}")
            continue
        all_preds[name] = preds_by_track
        intra_mv_results[name] = compute_intra_model_mv(preds_by_track)
        intra_mvcp_results[name] = compute_intra_model_mvcp(preds_by_track)
        print(
            f"[fusion] {name}: "
            f"MV   HR={intra_mv_results[name]['hr_fusion_acc']:.4f} "
            f"LR={intra_mv_results[name]['lr_fusion_acc']:.4f} | "
            f"MVCP HR={intra_mvcp_results[name]['hr_fusion_acc']:.4f} "
            f"LR={intra_mvcp_results[name]['lr_fusion_acc']:.4f} "
            f"({intra_mv_results[name]['n_tracks']} tracks)"
        )

    if not intra_mv_results:
        print("[fusion] Nenhum CSV encontrado. Tabela não foi gerada.")
        return None

    # Ensemble: todo subconjunto de >=2 modelos (ex: crnn+svtr, ..., os 3).
    inter_mv_results: dict[str, dict[str, float]] = {}
    inter_mvcp_results: dict[str, dict[str, float]] = {}
    names = list(all_preds)
    for r in range(2, len(names) + 1):
        for combo in combinations(names, r):
            subset = {m: all_preds[m] for m in combo}
            label = "+".join(combo)
            inter_mv_results[label] = compute_inter_model_mv(subset)
            inter_mvcp_results[label] = compute_inter_model_mvcp(subset)
            print(
                f"[fusion] ensemble {label}: "
                f"MV={inter_mv_results[label]['inter_model_acc']:.4f} | "
                f"MVCP={inter_mvcp_results[label]['inter_model_acc']:.4f} "
                f"({inter_mv_results[label]['n_tracks']} tracks)"
            )

    table = format_table(
        intra_mv_results, inter_mv_results, intra_mvcp_results, inter_mvcp_results
    )
    if out_path is None:
        out_path = logs_dir / f"{dataset_name}_fusion_table.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")
    print(f"[fusion] Tabela salva em: {out_path}")

    if summary_csv is not None:
        found = {m: csv_paths[m] for m in all_preds}
        _write_summary_csv(
            summary_csv, found, intra_mv_results, intra_mvcp_results,
            inter_mv_results, inter_mvcp_results,
        )
        print(f"[fusion] Resumo CSV salvo em: {summary_csv}")
    return out_path


# Aliases retrocompatíveis (caso main.py ou outro código importe os nomes antigos)
compute_intra_model = compute_intra_model_mv
compute_inter_model = compute_inter_model_mv
_combined_track_vote = _combined_track_mv


def _parse_preds_args(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--preds espera nome=caminho, recebeu: {item!r}")
        out[name] = Path(path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fusão MV/MVCP intra-modelo + ensemble inter-modelos a partir de CSVs de predições.",
    )
    ap.add_argument(
        "--preds", nargs="+", default=None, metavar="NOME=CSV",
        help="Ex: crnn=logs/jetson/preds_bj7_crnn_none.csv svtr=... (sem isso: CSVs padrão de logs/bj7_*/).",
    )
    ap.add_argument("--out", type=Path, default=None, help="Tabela TXT (padrão: logs/bj7_fusion_table.txt).")
    ap.add_argument("--out-csv", type=Path, default=None, help="Resumo CSV (lido pelo report.py).")
    args = ap.parse_args()
    run_fusion(
        models=["svtr", "crnn", "parseq"],
        dataset_name="bj7",
        logs_dir=Path("logs"),
        csv_paths=_parse_preds_args(args.preds) if args.preds else None,
        out_path=args.out,
        summary_csv=args.out_csv,
    )
