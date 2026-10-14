"""Inferência standalone do SVTR-Tiny — desktop ou Jetson, sem depender de treino.

Diferença em relação ao `eval_only` de `svtr/train.py::run_svtr`: aqui só o
split pedido (por padrão "testing") é carregado — não se constrói `ds_train`/
`ds_val`/seus DataLoaders, que `run_svtr` monta incondicionalmente mesmo em
modo avaliação. Também não depende do `SimpleNamespace` de `main.py`
(learning_rate, epochs, early_stop_patience, ...): só os parâmetros que
inferência de fato usa.

`--compress fp16|int8` aplica compressão na carga (ver `compress.py`);
checkpoints podados vêm do `prune_model.py`. `--dump-preds` grava as predições
por imagem (formato do `fusion.py`) no mesmo passe cronometrado.

Uso típico (na Jetson, depois de copiar o checkpoint treinado no desktop):

    OPENBLAS_CORETYPE=ARMV8 python -m svtr.infer \\
        --ckpt logs/bj7_svtr/bj7_svtr_best.pt \\
        --data-root bj7 --dataset bj7 --split testing \\
        --hardware jetson --batch-size 1 --limit 200 \\
        --out-csv logs/bench_svtr.csv
"""

import argparse
import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from bench import (
    PowerLogger,
    Timer,
    append_csv_row,
    detect_hardware,
    fmt_opt,
    gpu_memory_mb,
    power_mode,
    reset_peak_memory,
    resolve_device,
    write_preds_csv,
)
from compress import COMPRESS_MODES, apply_compression, model_size_gz_mb, model_size_mb, sparsity

from .dataset import (
    BJ7Dataset,
    NUM_CLASSES,
    RodoSolDataset,
    collate_fn,
    decode,
    make_transform,
)
from .model import SVTRTiny
from .train import greedy_decode_with_conf

_CSV_FIELDS = [
    "timestamp", "hardware", "power_mode", "device", "model", "dataset", "split", "binarize",
    "compress", "decode_mode", "sparsity", "model_size_mb", "model_size_gz_mb",
    "n_images", "batch_size", "warmup_batches",
    "total_infer_time_s", "avg_latency_ms", "fps",
    "seq_acc", "char_acc", "peak_gpu_mem_mb",
    "avg_temp_c", "max_temp_c", "avg_power_w", "energy_wh",
    "ckpt", "preds_csv",
]


def _char_acc(preds: List[str], targets: List[str]) -> Tuple[int, int]:
    """Acertos posicionais / total de caracteres (mesma lógica do parseq/train.py)."""
    matches = total = 0
    for p, t in zip(preds, targets):
        n = max(len(p), len(t))
        total += n
        for i in range(min(len(p), len(t))):
            matches += p[i] == t[i]
    return matches, total


def load_model(ckpt_path: Path, device: torch.device, binarize: bool = False) -> SVTRTiny:
    model = SVTRTiny(num_classes=NUM_CLASSES, in_ch=1 if binarize else 3).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_test_dataset(
    dataset_name: str, data_root: Path, split_path: Path, split: str, limit: Optional[int],
    binarize: bool = False,
):
    tf = make_transform(binarize=binarize)
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
    model: SVTRTiny,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 3,
    hardware: Optional[str] = None,
    input_dtype: torch.dtype = torch.float32,
) -> dict:
    """Roda o loader inteiro, medindo tempo/FPS/memória/acurácia/temperatura/energia.

    Os primeiros `warmup_batches` não entram na medição de tempo (cuDNN/CUDA
    context, cache de kernels — sem isso o primeiro batch distorce o FPS,
    principalmente na Jetson). Também guarda (gt, pred, conf_chars) de cada
    imagem em `result["pred_rows"]`, pra gravar o CSV de predições sem uma
    segunda passada.
    """
    correct_seq = total_seq = 0
    correct_char = total_char = 0
    total_time = 0.0
    n_timed_images = 0
    pred_rows = []  # type: List[Tuple[str, str, List[float]]]

    reset_peak_memory(device)

    with PowerLogger(device, interval_s=0.5, hardware=hardware) as power_log:
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device=device, dtype=input_dtype)
            timed = batch_idx >= warmup_batches
            if batch_idx == warmup_batches:
                # Pico de memória só dos batches cronometrados: o warmup inclui o
                # autotune do cudnn.benchmark, que aloca workspaces enormes.
                reset_peak_memory(device)

            with Timer(device) as t:
                logits = model(imgs)
                preds, confs = greedy_decode_with_conf(logits.float().cpu())

            targets = [decode(lbl.tolist()) for lbl in labels]
            correct_seq += sum(p == t for p, t in zip(preds, targets))
            total_seq += len(targets)
            m, n = _char_acc(preds, targets)
            correct_char += m
            total_char += n
            pred_rows.extend(zip(targets, preds, confs))

            if timed:
                total_time += t.elapsed
                n_timed_images += imgs.size(0)

    seq_acc = correct_seq / total_seq if total_seq else 0.0
    char_acc = correct_char / total_char if total_char else 0.0
    fps = n_timed_images / total_time if total_time > 0 else 0.0
    avg_latency_ms = (total_time / n_timed_images * 1000) if n_timed_images else 0.0

    result = {
        "n_images": total_seq,
        "n_timed_images": n_timed_images,
        "total_infer_time_s": total_time,
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
        "seq_acc": seq_acc,
        "char_acc": char_acc,
        "peak_gpu_mem_mb": gpu_memory_mb(device),
        "pred_rows": pred_rows,
    }
    result.update(power_log.summary())
    return result


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
    ap.add_argument("--binarize", action="store_true", help="Grayscale + binarização Otsu (1 canal) — tem que bater com o checkpoint carregado.")
    ap.add_argument("--compress", choices=COMPRESS_MODES, default="none", help="fp16 (GPU) | int8 (força CPU). Pruning: usar checkpoint do prune_model.py.")
    ap.add_argument("--out-csv", type=Path, default=Path("logs/bench_svtr.csv"))
    ap.add_argument("--dump-preds", type=Path, default=None, help="Se dado, salva CSV de predições (formato fusion.py) — só para dataset=bj7.")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    data_root = args.data_root or Path(args.dataset)
    split_path = args.split_path or data_root / "split.txt"
    device = resolve_device(args.device)
    hardware = (args.hardware or detect_hardware()).lower()

    print(f"Hardware: {hardware} | device: {device} | compress: {args.compress}")
    print(f"Carregando checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, device, binarize=args.binarize)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_sparsity = sparsity(model)
    model, device, input_dtype = apply_compression(model, args.compress, device)
    size_mb, size_gz_mb = model_size_mb(model), model_size_gz_mb(model)
    print(f"SVTR-Tiny — parâmetros: {n_params:,} | esparsidade: {model_sparsity:.2%} | tamanho: {size_mb:.1f}MB ({size_gz_mb:.1f}MB gzip)")

    ds = build_test_dataset(args.dataset, data_root, split_path, args.split, args.limit, binarize=args.binarize)
    print(f"Amostras ({args.split}): {len(ds)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    stats = run_inference(
        model, loader, device, warmup_batches=args.warmup_batches,
        hardware=hardware, input_dtype=input_dtype,
    )
    print(
        f"\nseq_acc={stats['seq_acc']:.4f} | char_acc={stats['char_acc']:.4f} | "
        f"fps={stats['fps']:.2f} | avg_latency={stats['avg_latency_ms']:.2f}ms | "
        f"peak_gpu_mem={stats['peak_gpu_mem_mb']} | "
        f"avg_temp={fmt_opt(stats['avg_temp_c'], '.1f')}C | "
        f"avg_power={fmt_opt(stats['avg_power_w'], '.1f')}W | "
        f"energy={fmt_opt(stats['energy_wh'])}Wh"
    )

    preds_csv = ""
    if args.dump_preds is not None:
        if args.dataset != "bj7":
            print("Aviso: --dump-preds só é suportado para dataset=bj7 (precisa de ds.metadata).")
        else:
            write_preds_csv(args.dump_preds, ds.metadata, stats["pred_rows"])
            preds_csv = args.dump_preds.as_posix()
            print(f"Predições salvas em: {args.dump_preds}")

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware,
        "power_mode": power_mode(hardware),
        "device": str(device),
        "model": "svtr",
        "dataset": args.dataset,
        "split": args.split,
        "binarize": args.binarize,
        "compress": args.compress,
        "decode_mode": "ctc_greedy",
        "sparsity": f"{model_sparsity:.4f}",
        "model_size_mb": f"{size_mb:.4f}",
        "model_size_gz_mb": f"{size_gz_mb:.4f}",
        "n_images": stats["n_images"],
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "total_infer_time_s": f"{stats['total_infer_time_s']:.4f}",
        "avg_latency_ms": f"{stats['avg_latency_ms']:.4f}",
        "fps": f"{stats['fps']:.4f}",
        "seq_acc": f"{stats['seq_acc']:.4f}",
        "char_acc": f"{stats['char_acc']:.4f}",
        "peak_gpu_mem_mb": stats["peak_gpu_mem_mb"],
        "avg_temp_c": fmt_opt(stats["avg_temp_c"]),
        "max_temp_c": fmt_opt(stats["max_temp_c"]),
        "avg_power_w": fmt_opt(stats["avg_power_w"]),
        "energy_wh": fmt_opt(stats["energy_wh"]),
        "ckpt": args.ckpt.as_posix(),
        "preds_csv": preds_csv,
    }
    append_csv_row(args.out_csv, row, _CSV_FIELDS)
    print(f"Métricas registradas em: {args.out_csv}")


if __name__ == "__main__":
    main()
