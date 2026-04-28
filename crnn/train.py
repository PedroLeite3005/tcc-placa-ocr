"""Loop de treino e avaliação do CRNN + CTC."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import BJ7Dataset, NUM_CLASSES, RodoSolDataset, collate_fn, decode
from .model import CRNN


def greedy_decode(logits: torch.Tensor, blank: int = 0) -> list[str]:
    """Decodifica logits (B, T, C) via greedy CTC."""
    indices = logits.argmax(-1).tolist()
    results = []
    for seq in indices:
        chars, prev = [], blank
        for idx in seq:
            if idx != blank and idx != prev:
                chars.append(idx)
            prev = idx
        results.append(decode(chars))
    return results


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ctc_loss: nn.CTCLoss,
    T: int,
) -> float:
    """Retorna a CTCLoss média no loader (usada para early stopping)."""
    model.eval()
    total = n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        targets_cat = torch.cat(labels)
        target_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
        input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long)
        logits = model(imgs)
        log_probs = logits.permute(1, 0, 2).log_softmax(2)
        total += ctc_loss(log_probs, targets_cat, input_lengths, target_lengths).item()
        n += 1
    return total / n if n else float("inf")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Retorna a acurácia de sequência (match exato pós-normalização)."""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = greedy_decode(model(imgs).cpu())
        targets = [decode(lbl.tolist()) for lbl in labels]
        correct += sum(p == t for p, t in zip(preds, targets))
        total += len(targets)
    return correct / total if total else 0.0


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA indisponível — usando MPS (Apple Silicon).")
            return torch.device("mps")
        print("CUDA indisponível — usando CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def run_crnn(p: SimpleNamespace) -> None:
    torch.manual_seed(p.seed)
    random.seed(p.seed)

    device = _resolve_device(p.device)
    data_root = Path(p.data_root)
    split_path = Path(p.split_path) if p.split_path else data_root / "split.txt"

    if p.dataset == "rodosol":
        def make_ds(split: str):
            return RodoSolDataset(data_root, split_path, split)
    elif p.dataset == "bj7":
        def make_ds(split: str):
            return BJ7Dataset(data_root, split_path, split)
    else:
        raise ValueError(f"Dataset desconhecido: {p.dataset!r}")

    print("Carregando datasets…")
    ds_train = make_ds("training")
    ds_val   = make_ds("validation")
    ds_test  = make_ds("testing")
    print(
        f"Amostras — treino: {len(ds_train)} | "
        f"val: {len(ds_val)} | teste: {len(ds_test)}"
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        ds_train, batch_size=p.batch_size, shuffle=True,
        num_workers=p.num_workers, collate_fn=collate_fn, pin_memory=pin,
    )
    val_loader = DataLoader(
        ds_val, batch_size=p.batch_size * 2, shuffle=False,
        num_workers=p.num_workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        ds_test, batch_size=p.batch_size * 2, shuffle=False,
        num_workers=p.num_workers, collate_fn=collate_fn,
    )

    model = CRNN(num_classes=NUM_CLASSES).to(device)

    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"CRNN — parâmetros treináveis: {n_params:,}")

    if p.resume:
        ckpt = torch.load(p.resume, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Retomando de: {p.resume}")

    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    ckpt_path = Path(p.out_dir) / f"{p.run_name}_best.pt"
    log_path  = Path(p.out_dir) / f"{p.run_name}_log.txt" if p.write_txt else None

    if p.eval_only:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        acc = evaluate(model, test_loader, device)
        print(f"Acurácia de sequência (teste): {acc:.4f}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, p.warp_h, p.warp_w, device=device)
        T = model(dummy).size(1)

    if log_path:
        log_path.write_text("epoch,train_loss,val_loss,val_acc\n", encoding="utf-8")

    best_val_acc    = -1.0
    lr_bad_epochs   = 0
    stop_bad_epochs = 0
    lr_was_reduced  = False

    for epoch in range(1, p.epochs + 1):
        model.train()
        total_loss = n_batches = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            targets_cat    = torch.cat(labels)
            target_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
            input_lengths  = torch.full((imgs.size(0),), T, dtype=torch.long)

            logits    = model(imgs)
            log_probs = logits.permute(1, 0, 2).log_softmax(2)
            loss      = ctc_loss(log_probs, targets_cat, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_train_loss = total_loss / n_batches
        val_loss       = evaluate_loss(model, val_loader, device, ctc_loss, T)
        val_acc        = evaluate(model, val_loader, device)

        print(
            f"Época {epoch:3d}/{p.epochs} | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{avg_train_loss:.6f},{val_loss:.6f},{val_acc:.6f}\n"
                )

        if val_acc > best_val_acc:
            best_val_acc    = val_acc
            lr_bad_epochs   = 0
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
                    lr_was_reduced  = True
                    lr_bad_epochs   = 0
                    stop_bad_epochs = 0
            else:
                stop_bad_epochs += 1
                if stop_bad_epochs >= p.early_stop_patience:
                    print(
                        f"Early stop na época {epoch}. "
                        f"Melhor val_acc: {best_val_acc:.4f}"
                    )
                    break

    if not ckpt_path.exists():
        print(
            f"Aviso: nenhum checkpoint 'best' foi salvo (best_acc={best_val_acc:.4f}). "
            f"Usando estado final do modelo para avaliação."
        )
        torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"\nAcurácia de sequência (teste): {test_acc:.4f}")

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"test_acc,{test_acc:.6f}\n")
