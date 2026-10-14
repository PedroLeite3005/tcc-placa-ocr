"""Compressão de modelos já treinados — FP16, quantização INT8 dinâmica e pruning.

Usado pelos `*/infer.py` (FP16/INT8, aplicados na carga do checkpoint) e pelo
`prune_model.py` (pruning, que gera um checkpoint novo). Compatível com Python
3.6 (roda na Jetson).

Limitações de hardware relevantes pra interpretar os resultados:
- INT8 (`quantize_dynamic`) só roda na CPU: a GPU da Jetson Nano (Maxwell,
  compute 5.3) não tem instruções INT8. Esperado: modelo menor, FPS menor que
  FP32 na GPU.
- Pruning não-estruturado zera pesos mas mantém as matrizes densas: não acelera
  a GPU. O ganho aparece no tamanho comprimido (`model_size_gz_mb`).
- FP16 é a técnica que de fato deve subir FPS (a GPU da Nano tem FP16 com
  throughput dobrado).
"""

import gzip
import io
import platform
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

COMPRESS_MODES = ("none", "fp16", "int8")


def _set_quant_engine() -> None:
    engines = torch.backends.quantized.supported_engines
    if platform.machine().lower() in ("aarch64", "arm64") and "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in engines:
        torch.backends.quantized.engine = "fbgemm"


def apply_compression(
    model: nn.Module, mode: str, device: torch.device, int8_exclude: Sequence[str] = ()
) -> Tuple[nn.Module, torch.device, torch.dtype]:
    """Aplica `mode` (none | fp16 | int8) e retorna (modelo, device, dtype da entrada).

    INT8 força CPU (ver docstring do módulo). FP16 exige CUDA.
    `int8_exclude`: caminhos de submódulos (ex: "model.head") que ficam em FP32
    — o `qconfig = None` explícito tem precedência no `quantize_dynamic`.
    """
    if mode == "none":
        return model, device, torch.float32
    if mode == "fp16":
        if device.type != "cuda":
            raise ValueError("--compress fp16 requer --device cuda.")
        return model.half(), device, torch.float16
    if mode == "int8":
        _set_quant_engine()
        cpu = torch.device("cpu")
        model = model.to(cpu).eval()
        for path in int8_exclude:
            sub = model
            for attr in path.split("."):
                sub = getattr(sub, attr)
            sub.qconfig = None
        qmodel = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
        return qmodel, cpu, torch.float32
    raise ValueError("Modo de compressão desconhecido: {!r}".format(mode))


def _prunable_tensors(model: nn.Module) -> List[torch.Tensor]:
    """Pesos de Conv2d / Linear / LSTM / MultiheadAttention (sem bias, norm, embeddings)."""
    tensors = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            tensors.append(module.weight)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters(recurse=False):
                if name.startswith("weight_"):
                    tensors.append(param)
        elif isinstance(module, nn.MultiheadAttention):
            if getattr(module, "in_proj_weight", None) is not None:
                tensors.append(module.in_proj_weight)
    return tensors


@torch.no_grad()
def prune_global(model: nn.Module, amount: float) -> float:
    """Pruning global não-estruturado por magnitude (L1): zera a fração `amount`
    dos pesos de menor |valor|, considerando todas as camadas juntas.

    Equivalente a `torch.nn.utils.prune.global_unstructured(L1Unstructured)` +
    `prune.remove`, mas feito in-place — evita a reparametrização (weight_orig/
    weight_mask) que conflita com os pesos "achatados" do cuDNN no LSTM.
    Retorna o limiar usado.
    """
    if not 0.0 < amount < 1.0:
        raise ValueError("amount deve estar em (0, 1).")
    tensors = _prunable_tensors(model)
    flat = torch.cat([t.detach().abs().flatten().float().cpu() for t in tensors])
    k = max(1, int(amount * flat.numel()))
    threshold = float(flat.kthvalue(k).values)
    for t in tensors:
        t.masked_fill_(t.abs() <= threshold, 0.0)
    return threshold


def sparsity(model: nn.Module) -> float:
    """Fração de zeros nos pesos podáveis (0.0 em modelo não podado)."""
    tensors = _prunable_tensors(model)
    total = sum(t.numel() for t in tensors)
    if total == 0:
        return 0.0
    zeros = sum(int((t == 0).sum()) for t in tensors)
    return zeros / total


def _serialized_bytes(model: nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def model_size_mb(model: nn.Module) -> float:
    """Tamanho do state_dict serializado (MB)."""
    return len(_serialized_bytes(model)) / (1024 ** 2)


def model_size_gz_mb(model: nn.Module) -> float:
    """Tamanho do state_dict serializado + gzip (MB) — mostra o ganho do pruning,
    já que os zeros comprimem bem mesmo com a matriz densa."""
    return len(gzip.compress(_serialized_bytes(model), compresslevel=6)) / (1024 ** 2)
