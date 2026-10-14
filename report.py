"""Tabela final do TCC2: acurácia × FPS × energia × tamanho, por configuração.

Junta os CSVs de benchmark dos `*/infer.py` (`--out-csv`) com o resumo do
`fusion.py --out-csv`, ligando cada predição ao seu benchmark pela coluna
`preds_csv` (comparada pelo nome do arquivo, pra funcionar com CSVs vindos da
Jetson). Roda no desktop.

Três tipos de linha:
- individual: cada run de benchmark (acurácia por imagem, sem fusão).
- intra: fusão MV/MVCP das imagens LR de um mesmo track (1 modelo).
- ensemble: fusão inter-modelos. Latência/energia/tamanho = soma dos membros
  (os modelos rodam em sequência na mesma GPU); FPS = 1000 / latência somada.

`placas_s` = FPS / imagens por track — na fusão por track, cada placa custa
todas as suas imagens (5 LR no split de teste).

Exemplo:
    python report.py --bench logs/jetson/bench_all.csv --fusion logs/jetson/fusion_summary.csv \\
        --out-csv logs/jetson/report.csv
"""

import argparse
import csv
from pathlib import Path

FIELDS = [
    "kind", "config", "hardware", "power_mode", "binarize", "compress", "decode_mode",
    "acc_imagem", "acc_mv", "acc_mvcp", "fps", "placas_s", "latency_ms",
    "energia_j_por_img", "avg_power_w", "model_size_mb", "model_size_gz_mb", "sparsity", "n_tracks",
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(x, spec=".4f"):
    return "" if x is None else format(x, spec)


def _energy_per_img(row):
    e, n = _f(row.get("energy_wh")), _f(row.get("n_images"))
    return e * 3600.0 / n if e is not None and n else None


def _load_bench(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _config_name(row):
    """Ex: crnn, svtr_bin, crnn_prune30 — binarização e pruning visíveis no nome."""
    name = row.get("model", "")
    if str(row.get("binarize", "")).lower() == "true":
        name += "_bin"
    sp = _f(row.get("sparsity"))
    if sp:
        name += "_prune{}".format(int(round(sp * 100)))
    return name


def _individual(row):
    return {
        "kind": "individual",
        "config": _config_name(row),
        "hardware": row.get("hardware", ""),
        "power_mode": row.get("power_mode", ""),
        "binarize": row.get("binarize", ""),
        "compress": row.get("compress", ""),
        "decode_mode": row.get("decode_mode", ""),
        "acc_imagem": row.get("seq_acc", ""),
        "fps": row.get("fps", ""),
        "latency_ms": row.get("avg_latency_ms", ""),
        "energia_j_por_img": _fmt(_energy_per_img(row)),
        "avg_power_w": row.get("avg_power_w", ""),
        "model_size_mb": row.get("model_size_mb", ""),
        "model_size_gz_mb": row.get("model_size_gz_mb", ""),
        "sparsity": row.get("sparsity", ""),
    }


def _join(values):
    uniq = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
    return "|".join(str(v) for v in uniq)


def _fused(frow, members):
    latency = [_f(m.get("avg_latency_ms")) for m in members]
    energy = [_energy_per_img(m) for m in members]
    sizes = [_f(m.get("model_size_mb")) for m in members]
    sizes_gz = [_f(m.get("model_size_gz_mb")) for m in members]
    lat = sum(latency) if None not in latency else None
    fps = 1000.0 / lat if lat else None
    n_tracks = _f(frow.get("n_tracks"))
    n_images = _f(members[0].get("n_images")) if len(members) == 1 else None
    # imagens por track: do próprio benchmark quando possível (5 LR no teste do BJ7)
    imgs_per_track = (n_images / n_tracks) if (n_images and n_tracks) else 5.0
    return {
        "kind": frow["kind"],
        "config": frow["config"],
        "hardware": _join(m.get("hardware", "") for m in members),
        "power_mode": _join(m.get("power_mode", "") for m in members),
        "binarize": _join(m.get("binarize", "") for m in members),
        "compress": _join(m.get("compress", "") for m in members),
        "decode_mode": _join(m.get("decode_mode", "") for m in members),
        "acc_imagem": members[0].get("seq_acc", "") if len(members) == 1 else "",
        "acc_mv": frow.get("mv_acc", ""),
        "acc_mvcp": frow.get("mvcp_acc", ""),
        "fps": _fmt(fps),
        "placas_s": _fmt(fps / imgs_per_track if fps else None),
        "latency_ms": _fmt(lat),
        "energia_j_por_img": _fmt(sum(energy) if None not in energy else None),
        "model_size_mb": _fmt(sum(sizes) if None not in sizes else None),
        "model_size_gz_mb": _fmt(sum(sizes_gz) if None not in sizes_gz else None),
        "n_tracks": frow.get("n_tracks", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, nargs="+", required=True, help="CSVs de benchmark dos */infer.py.")
    ap.add_argument("--fusion", type=Path, nargs="*", default=[], help="Resumos do fusion.py --out-csv.")
    ap.add_argument("--out-csv", type=Path, default=Path("logs/report.csv"))
    args = ap.parse_args()

    bench = _load_bench(args.bench)
    by_preds = {}
    for row in bench:
        if row.get("preds_csv"):
            by_preds[Path(row["preds_csv"]).name] = row  # última execução vence

    out = [_individual(r) for r in bench]
    for fpath in args.fusion:
        with open(fpath, encoding="utf-8", newline="") as f:
            for frow in csv.DictReader(f):
                names = [Path(p).name for p in frow["preds_csvs"].split(";")]
                missing = [n for n in names if n not in by_preds]
                if missing:
                    print("Aviso: sem benchmark para {} (config {}), pulando.".format(missing, frow["config"]))
                    continue
                out.append(_fused(frow, [by_preds[n] for n in names]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    cols = ["kind", "config", "binarize", "compress", "decode_mode", "acc_imagem", "acc_mvcp", "fps", "placas_s", "energia_j_por_img", "model_size_gz_mb"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in out:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    print("\nRelatório salvo em: {}".format(args.out_csv))


if __name__ == "__main__":
    main()
