"""Inferência standalone do CRNN — desktop ou Jetson, sem depender de treino.

Diferença em relação ao `eval_only` de `crnn/train.py::run_crnn`: aqui só o
split pedido (por padrão "testing") é carregado — não se constrói `ds_train`/
`ds_val`/seus DataLoaders, que `run_crnn` monta incondicionalmente mesmo em
modo avaliação. Também não depende do `SimpleNamespace` de `main.py`
(learning_rate, epochs, early_stop_patience, ...): só os parâmetros que
inferência de fato usa.

Uso típico (na Jetson, depois de copiar o checkpoint treinado no desktop):

    OPENBLAS_CORETYPE=ARMV8 python -m crnn.infer \\
        --ckpt logs/bj7_crnn/bj7_crnn_best.pt \\
        --data-root bj7 --dataset bj7 --split testing \\
        --hardware jetson --batch-size 1 --limit 200 \\
        --out-csv logs/bench_crnn.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bench import (
    Timer,
    append_csv_row,
    detect_hardware,
    gpu_memory_mb,
    reset_peak_memory,
    resolve_device,
)

from .dataset import (
    BJ7Dataset,
    NUM_CLASSES,
    RodoSolDataset,
    collate_fn,
    decode,
    make_transform,
)
from .model import CRNN
from .train import _format_conf, dump_test_predictions, greedy_decode_with_conf

_CSV_FIELDS = [
    "timestamp", "hardware", "device", "model", "dataset", "split",
    "n_images", "batch_size", "warmup_batches",
    "total_infer_time_s", "avg_latency_ms", "fps",
    "seq_acc", "char_acc", "peak_gpu_mem_mb",
]


def _char_acc(preds: list[str], targets: list[str]) -> tuple[int, int]:
    """Acertos posicionais / total de caracteres (mesma lógica do parseq/train.py)."""
    matches = total = 0
    for p, t in zip(preds, targets):
        n = max(len(p), len(t))
        total += n
        for i in range(min(len(p), len(t))):
            matches += p[i] == t[i]
    return matches, total


def load_model(ckpt_path: Path, device: torch.device) -> CRNN:
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_test_dataset(
    dataset_name: str, data_root: Path, split_path: Path, split: str, limit: int | None
):
    tf = make_transform()
    if dataset_name == "rodosol":
        ds = RodoSolDataset(data_root, split_path, split, transform=tf)
    elif dataset_name == "bj7":
        ds = BJ7Dataset(data_root, split_path, split, transform=tf)
    else:
        raise ValueError(f"Dataset desconhecido: {dataset_name!r}")
    if limit is not None and limit < len(ds):
        ds.samples = ds.samples[:limit]
        if hasattr(ds, "metadata"):
            ds.metadata = ds.metadata[:limit]
    return ds


@torch.no_grad()
def run_inference(
    model: CRNN,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 3,
) -> dict:
    """Roda o loader inteiro, medindo tempo/FPS/memória/acurácia.

    Os primeiros `warmup_batches` não entram na medição de tempo (cuDNN/CUDA
    context, cache de kernels — sem isso o primeiro batch distorce o FPS,
    principalmente na Jetson).
    """
    correct_seq = total_seq = 0
    correct_char = total_char = 0
    total_time = 0.0
    n_timed_images = 0

    reset_peak_memory(device)

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        timed = batch_idx >= warmup_batches

        with Timer(device) as t:
            logits = model(imgs)
            preds, _ = greedy_decode_with_conf(logits.cpu())

        targets = [decode(lbl.tolist()) for lbl in labels]
        correct_seq += sum(p == t for p, t in zip(preds, targets))
        total_seq += len(targets)
        m, n = _char_acc(preds, targets)
        correct_char += m
        total_char += n

        if timed:
            total_time += t.elapsed
            n_timed_images += imgs.size(0)

    seq_acc = correct_seq / total_seq if total_seq else 0.0
    char_acc = correct_char / total_char if total_char else 0.0
    fps = n_timed_images / total_time if total_time > 0 else 0.0
    avg_latency_ms = (total_time / n_timed_images * 1000) if n_timed_images else 0.0

    return {
        "n_images": total_seq,
        "n_timed_images": n_timed_images,
        "total_infer_time_s": total_time,
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
        "seq_acc": seq_acc,
        "char_acc": char_acc,
        "peak_gpu_mem_mb": gpu_memory_mb(device),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint .pt (state_dict) treinado no desktop.")
    ap.add_argument("--dataset", choices=["bj7", "rodosol"], default="bj7")
    ap.add_argument("--data-root", type=Path, default=None, help="Padrão: <dataset>/ na raiz do projeto.")
    ap.add_argument("--split-path", type=Path, default=None, help="Padrão: <data-root>/split.txt.")
    ap.add_argument("--split", default="testing", help="training | validation | testing.")
    ap.add_argument("--batch-size", type=int, default=1, help="1 = latência realista de borda; maior = throughput.")
    ap.add_argument("--limit", type=int, default=None, help="Limita a N imagens (smoke test rápido).")
    ap.add_argument("--warmup-batches", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="cuda | cpu | mps.")
    ap.add_argument("--hardware", default=None, help="jetson | desktop. Padrão: autodetectado.")
    ap.add_argument("--out-csv", type=Path, default=Path("logs/bench_crnn.csv"))
    ap.add_argument("--dump-preds", type=Path, default=None, help="Se dado, salva CSV de predições (formato fusion.py) — só para dataset=bj7.")
    args = ap.parse_args()

    data_root = args.data_root or Path(args.dataset)
    split_path = args.split_path or data_root / "split.txt"
    device = resolve_device(args.device)
    hardware = (args.hardware or detect_hardware()).lower()

    print(f"Hardware: {hardware} | device: {device}")
    print(f"Carregando checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CRNN — parâmetros: {n_params:,}")

    ds = build_test_dataset(args.dataset, data_root, split_path, args.split, args.limit)
    print(f"Amostras ({args.split}): {len(ds)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    stats = run_inference(model, loader, device, warmup_batches=args.warmup_batches)
    print(
        f"\nseq_acc={stats['seq_acc']:.4f} | char_acc={stats['char_acc']:.4f} | "
        f"fps={stats['fps']:.2f} | avg_latency={stats['avg_latency_ms']:.2f}ms | "
        f"peak_gpu_mem={stats['peak_gpu_mem_mb']}"
    )

    row = {
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware,
        "device": str(device),
        "model": "crnn",
        "dataset": args.dataset,
        "split": args.split,
        "n_images": stats["n_images"],
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "total_infer_time_s": f"{stats['total_infer_time_s']:.4f}",
        "avg_latency_ms": f"{stats['avg_latency_ms']:.4f}",
        "fps": f"{stats['fps']:.4f}",
        "seq_acc": f"{stats['seq_acc']:.4f}",
        "char_acc": f"{stats['char_acc']:.4f}",
        "peak_gpu_mem_mb": stats["peak_gpu_mem_mb"],
    }
    append_csv_row(args.out_csv, row, _CSV_FIELDS)
    print(f"Métricas registradas em: {args.out_csv}")

    if args.dump_preds is not None:
        if args.dataset != "bj7":
            print("Aviso: --dump-preds só é suportado para dataset=bj7 (precisa de ds.metadata).")
        else:
            dump_test_predictions(model, loader, ds, device, args.dump_preds)
            print(f"Predições salvas em: {args.dump_preds}")


if __name__ == "__main__":
    main()
