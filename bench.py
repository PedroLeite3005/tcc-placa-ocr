"""Utilitários compartilhados de benchmark de inferência (desktop x Jetson).

Usado pelos scripts `*/infer.py` de cada modelo (CRNN, SVTR, PARSeq) para que
a coleta de métricas (hardware, dispositivo, memória, CSV de resultados) seja
idêntica entre os três, permitindo comparação direta desktop x Jetson.

Métricas de temperatura/energia (jtop na Jetson, pynvml/nvidia-smi no desktop)
ainda não estão implementadas aqui — `jetson-stats` precisa ser instalado na
Jetson antes disso (ver anotações do projeto). `gpu_memory_mb` já funciona nos
dois ambientes hoje porque usa a API de alocação do próprio PyTorch, que
existe tanto no build de desktop quanto no build CUDA 10.2 da Jetson Nano.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch


def detect_hardware() -> str:
    """Detecta automaticamente 'jetson' vs 'desktop'.

    Jetson/L4T expõe `/etc/nv_tegra_release`, que não existe em desktops.
    Pode ser sobrescrito explicitamente via variável de ambiente `HARDWARE`
    ou pelo argumento --hardware de cada script de inferência.
    """
    env = os.environ.get("HARDWARE")
    if env:
        return env.lower()
    return "jetson" if Path("/etc/nv_tegra_release").exists() else "desktop"


def resolve_device(requested: str) -> torch.device:
    """Resolve o device pedido, com fallback gracioso (mesma lógica dos train.py)."""
    if requested == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA indisponível — usando MPS (Apple Silicon).")
            return torch.device("mps")
        print("CUDA indisponível — usando CPU.")
        return torch.device("cpu")
    return torch.device(requested)


class Timer:
    """Cronômetro simples com sincronização CUDA (essencial para medir tempo de GPU)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._t0 = 0.0
        self.elapsed = 0.0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __enter__(self) -> "Timer":
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._sync()
        self.elapsed = time.perf_counter() - self._t0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def gpu_memory_mb(device: torch.device) -> float | None:
    """Pico de memória alocada pelo PyTorch no device, em MB (None se não for CUDA)."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Adiciona uma linha ao CSV de métricas, criando o cabeçalho se necessário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
