"""Utilitários para carregar e usar PARSeq."""

from __future__ import annotations

import torch


def load_parseq(
    *,
    variant: str = "parseq",
    pretrained: bool = True,
    decode_ar: bool = True,
    refine_iters: int = 1,
    img_size: tuple[int, int] = (64, 256),
) -> torch.nn.Module:
    """Carrega PARSeq via Torch Hub sem prompt interativo de confiança.

    variant: nome do entrypoint no hub baudm/parseq.
    Opções: 'parseq' (base, ~24M) ou 'parseq_tiny' (~6M).
    img_size: (h, w) da entrada — alinhado com CRNN/SVTR (256×64) em vez do
    padrão do hub (128×32), repassado como kwarg de config ao criar o modelo.
    """
    return torch.hub.load(
        "baudm/parseq",
        variant,
        pretrained=pretrained,
        decode_ar=decode_ar,
        refine_iters=refine_iters,
        img_size=list(img_size),
        trust_repo=True,
        skip_validation=True,
    )


@torch.no_grad()
def predict_strings(model: torch.nn.Module, images: torch.Tensor) -> list[str]:
    """Retorna strings previstas para um batch de imagens."""
    logits = model(images)
    probs = logits.softmax(-1)
    labels, _ = model.tokenizer.decode(probs)
    return labels


@torch.no_grad()
def predict_strings_with_conf(
    model: torch.nn.Module, images: torch.Tensor
) -> tuple[list[str], list[list[float]]]:
    """Retorna (labels, conf_chars) — uma lista de probs por caractere predito."""
    logits = model(images)
    probs = logits.softmax(-1)
    labels, batch_probs = model.tokenizer.decode(probs)
    confs: list[list[float]] = []
    for label, p in zip(labels, batch_probs):
        # batch_probs[i] já vem filtrado pelo decoder; cortar para alinhar com o label.
        confs.append([float(x) for x in p.tolist()[: len(label)]])
    return labels, confs

