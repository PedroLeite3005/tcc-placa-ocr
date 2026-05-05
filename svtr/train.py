"""Loop de treino e avaliação do SVTR."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import BJ7Dataset, NUM_CLASSES, RodoSolDataset, collate_fn, decode
from .model import SVTRTiny


# ---------------------------------------------------------------------------
# Decodificação CTC (greedy)
# ---------------------------------------------------------------------------

def greedy_decode(logits: torch.Tensor, blank: int = 0) -> list[str]:
    """Decodifica logits (B, T, C) para lista de strings via greedy CTC."""
    indices = logits.argmax(-1).tolist()   # (B, T)
    results = []
    for seq in indices:
        chars, prev = [], blank
        for idx in seq:
            if idx != blank and idx != prev:
                chars.append(idx)
            prev = idx
        results.append(decode(chars))
    return results


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Retorna a acurácia de sequência no loader (previsão 100% correta)."""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = greedy_decode(model(imgs).cpu())
        targets = [decode(lbl.tolist()) for lbl in labels]
        correct += sum(p == t for p, t in zip(preds, targets))
        total += len(targets)
    return correct / total if total else 0.0


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
            preds = greedy_decode(model(imgs).cpu())
            targets = [decode(lbl.tolist()) for lbl in labels]
            for pred, gt in zip(preds, targets):
                meta = ds_test.metadata[idx]
                f.write(
                    f"{meta['track_id']},{meta['image_type']},"
                    f"{meta['image_idx']},{gt.upper()},{pred.upper()}\n"
                )
                idx += 1


# ---------------------------------------------------------------------------
# Resolução do dispositivo
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA indisponível — usando MPS (Apple Silicon).")
            return torch.device("mps")
        print("CUDA indisponível — usando CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Ponto de entrada principal
# ---------------------------------------------------------------------------

def run_svtr(p: SimpleNamespace) -> None:
    torch.manual_seed(p.seed)
    random.seed(p.seed)

    device = _resolve_device(p.device)
    data_root = Path(p.data_root)
    split_path = Path(p.split_path) if p.split_path else data_root / "split.txt"

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
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
    ds_val = make_ds("validation")
    ds_test = make_ds("testing")
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

    # ------------------------------------------------------------------
    # Modelo
    # ------------------------------------------------------------------
    model = SVTRTiny(
        img_h=p.warp_h, img_w=p.warp_w, num_classes=NUM_CLASSES,
    ).to(device)

    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"SVTR-Tiny — parâmetros treináveis: {n_params:,}")

    if p.resume:
        ckpt = torch.load(p.resume, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Retomando de: {p.resume}")

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------
    ckpt_path = Path(p.out_dir) / f"{p.run_name}_best.pt"
    log_path = Path(p.out_dir) / f"{p.run_name}_log.txt" if p.write_txt else None

    if p.eval_only:
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device)
        )
        acc = evaluate(model, test_loader, device)
        print(f"Acurácia de sequência (teste): {acc:.4f}")
        return

    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate)
    T = p.warp_w // 4   # comprimento da sequência CTC = 64

    if log_path:
        log_path.write_text("epoch,train_loss,val_acc\n", encoding="utf-8")

    best_acc = -1.0
    lr_bad_epochs = 0
    stop_bad_epochs = 0
    lr_was_reduced = False

    for epoch in range(1, p.epochs + 1):
        model.train()
        total_loss = n_batches = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)

            targets_cat = torch.cat(labels)                         # (sum_len,)
            target_lengths = torch.tensor(
                [len(lbl) for lbl in labels], dtype=torch.long
            )
            input_lengths = torch.full(
                (imgs.size(0),), T, dtype=torch.long
            )

            logits = model(imgs)                                    # (B, T, C)
            log_probs = logits.permute(1, 0, 2).log_softmax(2)     # (T, B, C)

            loss = ctc_loss(log_probs, targets_cat, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        val_acc = evaluate(model, val_loader, device)

        print(
            f"Época {epoch:3d}/{p.epochs} | "
            f"loss: {avg_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{avg_loss:.6f},{val_acc:.6f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
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
                        f"Melhor val_acc: {best_acc:.4f}"
                    )
                    break

    # ------------------------------------------------------------------
    # Avaliação final no conjunto de teste
    # ------------------------------------------------------------------
    if not ckpt_path.exists():
        print(
            f"Aviso: nenhum checkpoint 'best' foi salvo (best_acc={best_acc:.4f}). "
            f"Usando estado final do modelo para avaliação."
        )
        torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"\nAcurácia de sequência (teste): {test_acc:.4f}")

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"test_acc,{test_acc:.6f}\n")

    if p.dataset == "bj7":
        preds_csv = Path(p.out_dir) / f"{p.run_name}_preds.csv"
        dump_test_predictions(model, test_loader, ds_test, device, preds_csv)
        print(f"Predições do teste salvas em: {preds_csv}")
