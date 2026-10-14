"""Gera checkpoints podados (pruning global L1 não-estruturado) a partir de um já treinado.

Roda no desktop. Sem fine-tune (compressão pós-treino). Cada `--amount` vira um
run novo `logs/<run>_prune<NN>/<run>_prune<NN>_best.pt`, então `sync_jetson.sh
ckpt` e os `*/infer.py` funcionam sem mudança nenhuma (carregam como qualquer
checkpoint — só com mais zeros).

Exemplos:
    python prune_model.py --model crnn --ckpt logs/bj7_crnn/bj7_crnn_best.pt --amount 0.3 0.5 0.7
    python prune_model.py --model svtr --ckpt logs/bj7_svtr_bin/bj7_svtr_bin_best.pt --binarize --amount 0.5
    python prune_model.py --model parseq --ckpt logs/bj7_parseq/bj7_parseq_best.pt --img-h 32 --img-w 128 --amount 0.5
"""

import argparse
from pathlib import Path

import torch

from compress import model_size_gz_mb, model_size_mb, prune_global, sparsity


def _load(model_name: str, ckpt: Path, binarize: bool, img_h: int, img_w: int):
    cpu = torch.device("cpu")
    if model_name == "crnn":
        from crnn.infer import load_model
        return load_model(ckpt, cpu, binarize=binarize)
    if model_name == "svtr":
        from svtr.infer import load_model
        return load_model(ckpt, cpu, binarize=binarize)
    if model_name == "parseq":
        from parseq.infer import load_model
        return load_model(ckpt, cpu, img_size=(img_h, img_w), binarize=binarize)
    raise ValueError(model_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["crnn", "svtr", "parseq"], required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--amount", type=float, nargs="+", required=True, help="Fração de pesos zerados, ex: 0.3 0.5 0.7.")
    ap.add_argument("--binarize", action="store_true", help="Checkpoint treinado com binarização (1 canal).")
    ap.add_argument("--img-h", type=int, default=64, help="PARSeq: resolução do checkpoint (legado: 32).")
    ap.add_argument("--img-w", type=int, default=256, help="PARSeq: resolução do checkpoint (legado: 128).")
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    args = ap.parse_args()

    run = args.ckpt.parent.name  # ex: bj7_crnn, bj7_svtr_bin
    for amount in args.amount:
        model = _load(args.model, args.ckpt, args.binarize, args.img_h, args.img_w)
        before_mb, before_gz = model_size_mb(model), model_size_gz_mb(model)
        prune_global(model, amount)
        out_run = "{}_prune{}".format(run, int(round(amount * 100)))
        out_path = args.logs_dir / out_run / "{}_best.pt".format(out_run)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(
            "{} | amount={:.2f} | esparsidade={:.2%} | {:.1f}MB -> {:.1f}MB | gzip {:.1f}MB -> {:.1f}MB | {}".format(
                out_run, amount, sparsity(model), before_mb, model_size_mb(model),
                before_gz, model_size_gz_mb(model), out_path.as_posix(),
            )
        )


if __name__ == "__main__":
    main()
