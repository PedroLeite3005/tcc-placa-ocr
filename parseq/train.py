"""Loop de treino e avaliação do PARSeq."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    BJ7Dataset,
    RodoSolDataset,
    collate_fn,
    make_transform,
    make_transform_train,
)
from .model import load_parseq, predict_strings


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA indisponível — usando MPS (Apple Silicon).")
            return torch.device("mps")
        print("CUDA indisponível — usando CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def dump_test_predictions(
    model: nn.Module,
    loader: DataLoader,
    ds_test,
    device: torch.device,
    out_path: Path,
) -> None:
    """Salva CSV com track_id, image_type, image_idx, gt, pred do test set BJ7."""
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("track_id,image_type,image_idx,gt,pred\n")
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = predict_strings(model, imgs)
            targets = [lbl.lower() for lbl in labels]
            for pred, gt in zip(preds, targets):
                meta = ds_test.metadata[idx]
                f.write(
                    f"{meta['track_id']},{meta['image_type']},"
                    f"{meta['image_idx']},{gt.upper()},{pred.upper()}\n"
                )
                idx += 1


def _char_acc(preds: list[str], targets: list[str]) -> tuple[int, int]:
    """Conta acertos posicionais e total de caracteres (penaliza diferença de tamanho)."""
    matches = 0
    total = 0
    for p, t in zip(preds, targets):
        n = max(len(p), len(t))
        total += n
        for i in range(min(len(p), len(t))):
            if p[i] == t[i]:
                matches += 1
    return matches, total


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Retorna (seq_acc, char_acc) sobre o loader."""
    model.eval()
    correct_seq = total_seq = 0
    correct_char = total_char = 0
    last_preds: list[str] = []
    last_targets: list[str] = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = predict_strings(model, imgs)
        targets = [lbl.lower() for lbl in labels]
        correct_seq += sum(p == t for p, t in zip(preds, targets))
        total_seq += len(targets)
        m, n = _char_acc(preds, targets)
        correct_char += m
        total_char += n
        last_preds, last_targets = preds, targets
    if total_seq and correct_seq == 0:
        print(f"DEBUG preds: {last_preds[:5]} | gt: {last_targets[:5]}")
    seq_acc = correct_seq / total_seq if total_seq else 0.0
    char_acc = correct_char / total_char if total_char else 0.0
    return seq_acc, char_acc


def run_parseq(p: SimpleNamespace) -> None:
    torch.manual_seed(p.seed)
    random.seed(p.seed)

    device = _resolve_device(p.device)
    data_root = Path(p.data_root)
    split_path = Path(p.split_path) if p.split_path else data_root / "split.txt"

    tf_train = make_transform_train()
    tf_eval = make_transform()

    if p.dataset == "rodosol":
        def make_ds(split: str):
            tf = tf_train if split == "training" else tf_eval
            return RodoSolDataset(data_root, split_path, split, transform=tf)
    elif p.dataset == "bj7":
        def make_ds(split: str):
            tf = tf_train if split == "training" else tf_eval
            return BJ7Dataset(data_root, split_path, split, transform=tf)
    else:
        raise ValueError(f"Dataset desconhecido: {p.dataset!r}")

    print("Carregando datasets…")
    ds_train = make_ds("training")
    ds_val = make_ds("validation")
    ds_test = make_ds("testing")
    print(
        f"Amostras — treino: {len(ds_train)} | "
        f"val: {len(ds_val)} | teste: {len(ds_test)}"
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        ds_train,
        batch_size=p.batch_size,
        shuffle=True,
        num_workers=p.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=p.batch_size * 2,
        shuffle=False,
        num_workers=p.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=p.batch_size * 2,
        shuffle=False,
        num_workers=p.num_workers,
        collate_fn=collate_fn,
    )

    model = load_parseq(
        variant=p.parseq_variant,
        pretrained=p.parseq_pretrained,
        decode_ar=p.parseq_decode_ar,
        refine_iters=p.parseq_refine_iters,
    ).to(device)

    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"PARSeq — parâmetros treináveis: {n_params:,}")

    if p.resume:
        ckpt = torch.load(p.resume, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Retomando de: {p.resume}")

    ckpt_path = Path(p.out_dir) / f"{p.run_name}_best.pt"
    log_path = Path(p.out_dir) / f"{p.run_name}_log.txt" if p.write_txt else None

    if p.eval_only:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        seq_acc, char_acc = evaluate(model, test_loader, device)
        print(
            f"Teste — seq_acc: {seq_acc:.4f} | char_acc: {char_acc:.4f}"
        )
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=p.learning_rate)
    if log_path:
        log_path.write_text(
            "epoch,train_loss,val_seq_acc,val_char_acc\n", encoding="utf-8"
        )

    best_char_acc = -1.0
    best_seq_acc = -1.0
    lr_bad_epochs = 0
    stop_bad_epochs = 0
    lr_was_reduced = False

    for epoch in range(1, p.epochs + 1):
        model.train()
        total_loss = n_batches = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = [lbl.lower() for lbl in labels]

            _, loss, _ = model.forward_logits_loss(imgs, labels)

            if epoch == 1 and n_batches == 0:
                print(f"DEBUG loss inicial (batch 1): {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        seq_acc, char_acc = evaluate(model, val_loader, device)

        print(
            f"Época {epoch:3d}/{p.epochs} | "
            f"loss: {avg_loss:.4f} | "
            f"seq_acc: {seq_acc:.4f} | char_acc: {char_acc:.4f}"
        )

        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{avg_loss:.6f},{seq_acc:.6f},{char_acc:.6f}\n"
                )

        if char_acc > best_char_acc:
            best_char_acc = char_acc
            best_seq_acc = max(best_seq_acc, seq_acc)
            lr_bad_epochs = 0
            stop_bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        elif epoch >= p.min_epochs:
            if not lr_was_reduced:
                lr_bad_epochs += 1
                if lr_bad_epochs >= p.lr_patience:
                    for g in optimizer.param_groups:
                        g["lr"] *= p.lr_factor
                    new_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Agendador: lr reduzido para {new_lr:.2e} na época {epoch}."
                    )
                    lr_was_reduced = True
                    lr_bad_epochs = 0
                    stop_bad_epochs = 0
            else:
                stop_bad_epochs += 1
                if stop_bad_epochs >= p.early_stop_patience:
                    print(
                        f"Early stop na época {epoch}. "
                        f"Melhor char_acc: {best_char_acc:.4f} | "
                        f"melhor seq_acc: {best_seq_acc:.4f}"
                    )
                    break

    if not ckpt_path.exists():
        print(
            f"Aviso: nenhum checkpoint 'best' foi salvo "
            f"(best_char_acc={best_char_acc:.4f}). "
            f"Usando estado final do modelo para avaliação."
        )
        torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_seq_acc, test_char_acc = evaluate(model, test_loader, device)
    print(
        f"\nTeste — seq_acc: {test_seq_acc:.4f} | "
        f"char_acc: {test_char_acc:.4f}"
    )

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"test_acc,{test_seq_acc:.6f},{test_char_acc:.6f}\n"
            )

    if p.dataset == "bj7":
        preds_csv = Path(p.out_dir) / f"{p.run_name}_preds.csv"
        dump_test_predictions(model, test_loader, ds_test, device, preds_csv)
        print(f"Predições do teste salvas em: {preds_csv}")

