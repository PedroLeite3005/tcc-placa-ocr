"""Exporta os checkpoints treinados para ONNX (entrada do TensorRT na Jetson).

Roda no desktop. Não precisa retreinar: usa os mesmos `.pt` dos benchmarks.
Cada ONNX é validado contra o PyTorch (onnxruntime) antes de ser salvo como OK.

- Opset 13: o máximo suportado pelo parser ONNX do TensorRT 8.2 (JetPack 4.6).
- Entrada fixa (1, C, H, W) — batch 1, igual aos benchmarks.
- PARSeq: o modo de decodificação (ar / nar / nar1) fica gravado no grafo, então
  cada modo vira um ONNX separado. Dois ajustes só durante a exportação:
    * atenção "fundida" do timm desligada (TIMM_FUSED_ATTN=0) — a operação
      scaled_dot_product_attention não existe no opset 13;
    * torch.triu (máscaras do decoder) trocado por uma versão equivalente com
      operações básicas — Trilu também não existe no opset 13.
  Os dois mudam só a forma de exportar, não o resultado (validado: previsões
  idênticas ao PyTorch em 200/200 imagens reais).

Saída: logs/onnx/<run>.onnx (CRNN/SVTR) e logs/onnx/<run>_<ar|nar|nar1>.onnx (PARSeq).

Exemplos:
    python export_onnx.py --all
    python export_onnx.py --run bj7_crnn bj7_parseq_legado
"""

import os

os.environ.setdefault("TIMM_FUSED_ATTN", "0")  # antes de importar timm (via PARSeq)

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

PARSEQ_DECODES = {"ar": (True, 1), "nar": (False, 0), "nar1": (False, 1)}


def _triu_basic(input, diagonal=0, *args, **kwargs):
    """torch.triu equivalente, só com operações do opset 13 (arange/compare/where)."""
    n, m = input.shape[-2], input.shape[-1]
    rows = torch.arange(n, device=input.device).unsqueeze(-1)
    cols = torch.arange(m, device=input.device).unsqueeze(0)
    keep = (cols - rows) >= diagonal
    return torch.where(keep, input, torch.zeros_like(input))


def _run_info(run: str):
    """(modelo, binarize, img_h, img_w) a partir do nome do run, igual ao run_bench_matrix.sh."""
    model = run.split("_")[1]
    binarize = "_bin" in run
    h, w = (32, 128) if "_legado" in run else (64, 256)
    return model, binarize, h, w


def _load(model: str, ckpt: Path, binarize: bool, h: int, w: int, decode_ar=True, refine_iters=1):
    cpu = torch.device("cpu")
    if model == "crnn":
        from crnn.infer import load_model
        return load_model(ckpt, cpu, binarize=binarize)
    if model == "svtr":
        from svtr.infer import load_model
        return load_model(ckpt, cpu, binarize=binarize)
    from parseq.infer import load_model
    return load_model(ckpt, cpu, decode_ar=decode_ar, refine_iters=refine_iters, img_size=(h, w), binarize=binarize)


def export_one(model_obj, x: torch.Tensor, out: Path, opset: int) -> float:
    """Exporta e valida com onnxruntime; retorna a maior diferença absoluta vs PyTorch."""
    import onnxruntime as ort

    out.parent.mkdir(parents=True, exist_ok=True)
    orig_triu = torch.triu
    torch.triu = _triu_basic
    try:
        torch.onnx.export(
            model_obj, x, str(out), input_names=["image"], output_names=["logits"],
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    finally:
        torch.triu = orig_triu
    with torch.no_grad():
        ref = model_obj(x).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x.numpy()})[0]
    return float(np.abs(got - ref).max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Todos os checkpoints base em logs/bj7_*/ (sem os podados).")
    ap.add_argument("--run", nargs="*", default=[], help="Runs específicos (nomes das pastas), ex: bj7_crnn bj7_parseq_legado.")
    ap.add_argument("--decodes", nargs="*", default=list(PARSEQ_DECODES), help="PARSeq: modos a exportar (padrão: ar nar nar1).")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--out-dir", type=Path, default=Path("logs/onnx"))
    ap.add_argument("--tol", type=float, default=1e-3, help="Diferença máxima aceita vs PyTorch.")
    args = ap.parse_args()

    runs = list(args.run)
    if args.all:
        for ckpt in sorted(glob.glob("logs/bj7_*/bj7_*_best.pt")):
            run = Path(ckpt).parent.name
            if "_prune" not in run and run not in runs:
                runs.append(run)
    if not runs:
        ap.error("passe --all ou --run <nomes>")

    torch.manual_seed(0)
    n_fail = 0
    for run in runs:
        ckpt = Path("logs") / run / "{}_best.pt".format(run)
        model, binarize, h, w = _run_info(run)
        x = torch.randn(1, 1 if binarize else 3, h, w)
        variants = [(None, True, 1)] if model != "parseq" else [
            (d, PARSEQ_DECODES[d][0], PARSEQ_DECODES[d][1]) for d in args.decodes
        ]
        for dec, ar, ri in variants:
            name = run if dec is None else "{}_{}".format(run, dec)
            out = args.out_dir / "{}.onnx".format(name)
            try:
                m = _load(model, ckpt, binarize, h, w, decode_ar=ar, refine_iters=ri)
                diff = export_one(m, x, out, args.opset)
                status = "OK" if diff <= args.tol else "DIFERENÇA ALTA"
                if diff > args.tol:
                    n_fail += 1
                print("{:<26} {:<16} entrada {} | dif. vs PyTorch {:.1e} | {:.1f}MB | {}".format(
                    name, status, tuple(x.shape), diff, out.stat().st_size / 1e6, out.as_posix()))
            except Exception as exc:
                n_fail += 1
                print("{:<26} FALHOU: {}: {}".format(name, type(exc).__name__, str(exc)[:200]))
    print("\nFalhas: {}".format(n_fail))


if __name__ == "__main__":
    main()
