"""Utilitários compartilhados de benchmark de treino/inferência (desktop x Jetson).

Usado pelos `*/train.py` e `*/infer.py` de cada modelo (CRNN, SVTR, PARSeq) para
que a coleta de métricas (hardware, dispositivo, memória, temperatura, energia,
CSV de resultados) seja idêntica entre os três, permitindo comparação direta
desktop x Jetson.

`gpu_memory_mb` funciona nos dois ambientes hoje porque usa a API de alocação
do próprio PyTorch (existe tanto no build de desktop quanto no build CUDA 10.2
da Jetson Nano). Temperatura/potência/energia (`PowerLogger`) usam `nvidia-smi`
no desktop (sem dependência nova) e `jetson-stats` (jtop) na Jetson — requer
`sudo -H pip3 install -U jetson-stats` + reboot lá (ver README seção 14). Se o
pacote `jtop` não estiver instalado/a conexão falhar, `PowerLogger` avisa uma
vez e segue retornando `None` pros campos de temp/potência/energia, sem quebrar
o treino/inferência.
"""

import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

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


def gpu_memory_mb(device: torch.device) -> Optional[float]:
    """Pico de memória alocada pelo PyTorch no device, em MB (None se não for CUDA)."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def fmt_opt(x: Optional[float], spec: str = ".4f") -> str:
    """Formata um float opcional pro CSV — string vazia se `None` (telemetria indisponível)."""
    return "" if x is None else format(x, spec)


def append_csv_row(path: Path, row: dict, fieldnames: List[str]) -> None:
    """Adiciona uma linha ao CSV de métricas, criando o cabeçalho se necessário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Temperatura / potência / energia
# ---------------------------------------------------------------------------

def _safe_float(s) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _nvidia_smi_query(fields: List[str]) -> Optional[List[str]]:
    """Roda `nvidia-smi --query-gpu=...` e retorna os valores como strings, ou
    None se o comando falhar/não existir (ex: não é uma máquina com GPU NVIDIA)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return [v.strip() for v in out.stdout.strip().split(",")]
    except Exception:
        return None


def _nvidia_smi_power_fallback() -> Optional[float]:
    """Algumas GPUs (ex: mobile/laptop) retornam 'N/A' no campo `power.draw` do
    `--query-gpu`, mas expõem a potência via `nvidia-smi -q -d POWER` (seção
    'Power Samples > Avg'). Usado como fallback quando o campo direto falha."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-q", "-d", "POWER"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0:
            return None
        m = re.search(r"Power Samples\s*\n(?:.*\n)*?\s*Avg\s*:\s*([\d.]+)\s*W", out.stdout)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _desktop_temp_power() -> Tuple[Optional[float], Optional[float]]:
    vals = _nvidia_smi_query(["temperature.gpu", "power.draw"])
    if vals is None:
        return None, None
    temp_c = _safe_float(vals[0])
    power_w = _safe_float(vals[1]) if len(vals) > 1 else None
    if power_w is None:
        power_w = _nvidia_smi_power_fallback()
    return temp_c, power_w


_jtop_handle = None  # conexão jtop reaproveitada entre amostras (abrir uma por chamada seria caro)
_jtop_unavailable = False  # trava depois da 1ª falha, pra não tentar reconectar a cada amostra


def _get_jtop_handle():
    """Abre (uma vez) e mantém a conexão com o serviço jtop (jetson-stats).

    Lazy + cacheada: `PowerLogger` chama `gpu_temp_power` a cada `interval_s`,
    então reabrir a conexão a cada amostra seria um overhead desnecessário.
    """
    global _jtop_handle, _jtop_unavailable
    if _jtop_unavailable:
        return None
    if _jtop_handle is not None:
        return _jtop_handle
    try:
        from jtop import jtop
    except ImportError:
        _jtop_unavailable = True
        print(
            "Aviso: pacote 'jtop' (jetson-stats) não encontrado — temp/potência "
            "ficarão vazias. Instale com: sudo -H pip3 install -U jetson-stats (+ reboot)."
        )
        return None
    try:
        handle = jtop()
        handle.start()
        _jtop_handle = handle
        return handle
    except Exception as exc:
        _jtop_unavailable = True
        print(
            f"Aviso: não foi possível conectar ao serviço jtop ({exc}) — confirme "
            f"'sudo systemctl status jtop.service' e que o usuário está no grupo jtop "
            f"(requer reboot após instalar jetson-stats)."
        )
        return None


def _jetson_temp_power() -> Tuple[Optional[float], Optional[float]]:
    """Lê temperatura/potência via jetson-stats (jtop).

    Os nomes de chave do `jetson.stats` variam entre versões do jetson-stats —
    tenta as variantes mais comuns e cai pra None se nenhuma bater (nunca
    levanta exceção, pra não derrubar o treino/inferência por causa de telemetria).
    """
    jetson = _get_jtop_handle()
    if jetson is None:
        return None, None
    try:
        if not jetson.ok():
            return None, None
        stats = jetson.stats
    except Exception:
        return None, None

    temp_c = None
    for key in ("Temp GPU", "Temp gpu", "Temp CPU", "Temp cpu"):
        if stats.get(key) is not None:
            temp_c = _safe_float(stats[key])
            break

    power_w = None
    for key in ("power cur", "Power TOT", "Power cur", "power avg", "Power avg"):
        if stats.get(key) is not None:
            power_w = _safe_float(stats[key])
            if power_w is not None:
                power_w /= 1000.0  # jtop reporta em mW
            break

    return temp_c, power_w


def gpu_temp_power(device: torch.device, hardware: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
    """Retorna (temp_c, power_w) da GPU. (None, None) se não for CUDA ou indisponível."""
    if device.type != "cuda":
        return None, None
    hardware = (hardware or detect_hardware()).lower()
    if hardware == "jetson":
        return _jetson_temp_power()
    return _desktop_temp_power()


class PowerLogger:
    """Amostra temperatura/potência da GPU em background e integra energia (Wh).

    Uso:
        with PowerLogger(device) as pl:
            ... trecho a medir (época de treino, run de inferência) ...
        stats = pl.summary()  # avg_temp_c, max_temp_c, avg_power_w, energy_wh

    A energia é a integral trapezoidal da potência amostrada ao longo do tempo
    — mais fiel que "potência instantânea × duração total", mas ainda uma
    aproximação (depende de `interval_s`; menor intervalo = mais preciso, mais
    overhead). Campos ficam `None` quando a leitura de hardware não está
    disponível (CPU, ou Jetson antes do jtop estar configurado).
    """

    def __init__(self, device: torch.device, interval_s: float = 1.0, hardware: Optional[str] = None) -> None:
        self.device = device
        self.interval_s = interval_s
        self.hardware = hardware
        self._samples = []  # type: List[Tuple[float, Optional[float], Optional[float]]]
        self._stop_event = threading.Event()
        self._thread = None
        self._t_start = 0.0
        self._t_end = 0.0

    def _run(self) -> None:
        while not self._stop_event.is_set():
            temp_c, power_w = gpu_temp_power(self.device, self.hardware)
            self._samples.append((time.monotonic(), temp_c, power_w))
            self._stop_event.wait(self.interval_s)

    def __enter__(self) -> "PowerLogger":
        self._stop_event.clear()
        self._samples = []
        self._t_start = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
        self._t_end = time.monotonic()

    def summary(self) -> dict:
        temps = [t for _, t, _ in self._samples if t is not None]
        powers = [(ts, p) for ts, _, p in self._samples if p is not None]

        avg_temp = sum(temps) / len(temps) if temps else None
        max_temp = max(temps) if temps else None
        avg_power = sum(p for _, p in powers) / len(powers) if powers else None

        energy_wh = None
        if len(powers) >= 2:
            # Integração trapezoidal — mais fiel para trechos longos (ex: época de treino).
            energy_j = 0.0
            for (t0, p0), (t1, p1) in zip(powers, powers[1:]):
                energy_j += (p0 + p1) / 2.0 * (t1 - t0)
            energy_wh = energy_j / 3600.0
        elif len(powers) == 1:
            # Trecho medido mais curto que `interval_s` (comum em benchmarks de
            # inferência) — só 1 amostra de potência. Aproxima por potência × duração
            # total do bloco `with`, em vez de deixar a energia sem valor nenhum.
            duration_s = self._t_end - self._t_start
            energy_wh = powers[0][1] * duration_s / 3600.0

        return {
            "avg_temp_c": avg_temp,
            "max_temp_c": max_temp,
            "avg_power_w": avg_power,
            "energy_wh": energy_wh,
        }
