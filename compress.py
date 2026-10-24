"""Compressão/otimização de modelos já treinados — FP16, INT8, pruning e TensorRT.

Usado pelos `*/infer.py` (FP16/INT8/TensorRT, aplicados na carga do checkpoint)
e pelo `prune_model.py` (pruning, que gera um checkpoint novo). Compatível com
Python 3.6 (roda na Jetson).

O que os benchmarks na Jetson Nano mostraram:
- FP16 (PyTorch): acurácia igual e modelo pela metade, mas FPS praticamente
  igual — com batch 1 o gargalo é o custo de disparar cada operação a partir
  da CPU, não a conta na GPU.
- INT8 (`quantize_dynamic`): não roda na Jetson — o PyTorch da NVIDIA veio sem
  motor de quantização (NoQEngine) e a GPU Maxwell não tem INT8.
- Pruning não-estruturado: zera pesos mas as matrizes continuam densas — não
  acelera. Ganho no tamanho comprimido (`model_size_gz_mb`).
- TensorRT (`trt_fp16`/`trt_fp32`): compila o modelo num motor otimizado pra
  GPU (funde camadas, roda o grafo inteiro de uma vez, escolhe kernels). Ataca
  justamente o gargalo de disparo de operações. Ver `trt_runner.py`.
"""

import gzip
import io
import platform
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

COMPRESS_MODES = ("none", "fp16", "int8", "trt_fp16", "trt_fp32")


def _set_quant_engine() -> None:
    engines = torch.backends.quantized.supported_engines
    if platform.machine().lower() in ("aarch64", "arm64") and "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in engines:
        torch.backends.quantized.engine = "fbgemm"


def apply_compression(
    model: nn.Module, mode: str, device: torch.device, int8_exclude: Sequence[str] = (),
    trt_engine=None,
) -> Tuple[nn.Module, torch.device, torch.dtype]:
    """Aplica `mode` (none | fp16 | int8 | trt_fp16 | trt_fp32) e retorna
    (modelo, device, dtype da entrada).

    INT8 força CPU (ver docstring do módulo). FP16 e TensorRT exigem CUDA.
    `int8_exclude`: caminhos de submódulos (ex: "model.head") que ficam em FP32
    — o `qconfig = None` explícito tem precedência no `quantize_dynamic`.
    `trt_engine`: caminho do .engine (obrigatório nos modos trt_*). O modelo
    PyTorch sai da GPU (só o tokenizer do PARSeq é reaproveitado).
    """
    if mode == "none":
        return model, device, torch.float32
    if mode.startswith("trt_"):
        if trt_engine is None:
            raise ValueError("--compress {} requer --trt-engine <arquivo .engine>.".format(mode))
        from trt_runner import TRTModule

        trt_model = TRTModule(trt_engine, device)
        if hasattr(model, "tokenizer"):
            trt_model.tokenizer = model.tokenizer
        model.to(torch.device("cpu"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return trt_model, device, torch.float32
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
    if hasattr(model, "engine_bytes"):  # TRTModule: tamanho do motor TensorRT
        return model.engine_bytes
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
