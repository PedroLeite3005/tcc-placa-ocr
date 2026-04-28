"""
Parâmetros do experimento: edite em obter_parametros().
  run_name, model, dataset, data_root, split_path, seed, batch_size, epochs,
  learning_rate, write_txt, device, resume, eval_only, early_stop_patience,
  num_workers, imgsz, conf, iou, warp_w, warp_h
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

_DATA_ROOTS = {
    "rodosol": Path("rodosol"),
    "bj7": Path("bj7"),
}

_DATASETS = ["rodosol", "bj7"]
# "svtr","parseq","crnn"
_MODELS = ["crnn"]

def obter_parametros(model: str, dataset: str) -> SimpleNamespace:
    """Valores padrão do run; altere aqui quando precisar mudar."""
    run_name = f"{dataset}_{model}"  # Nome único do experimento (dataset + modelo).
    out_dir = Path("logs") / run_name  # Pasta onde logs e checkpoints deste run serão salvos.

    # caminhos
    data_root = _DATA_ROOTS[dataset]  # Diretório base do dataset selecionado.
    split_path: Path | None = None   # Arquivo de split; None usa data_root/split.txt.

    # treino
    seed = 42  # Semente para reprodutibilidade de treino.
    batch_size = 64  # Quantidade de amostras por batch.
    epochs = 30  # Número máximo de épocas de treino.
    learning_rate = 0.0005  # Taxa de aprendizado do otimizador.
    write_txt = True  # Se True, grava histórico de treino em arquivo .txt.
    device = "cuda"  # Dispositivo alvo de execução: cuda | mps | cpu.
    resume: Path | None = None  # Checkpoint para retomar treino, ou None para iniciar do zero.
    eval_only = False  # Se True, apenas avalia sem treinar.
    early_stop_patience = 3  # Quantas épocas sem melhora antes de parar cedo (após o agendador agir).
    min_epochs = 10  # Épocas mínimas de warmup antes de qualquer intervenção automática.
    lr_patience = 3  # Épocas sem melhora (após warmup) para o agendador reduzir o lr.
    lr_factor = 0.1  # Fator aplicado ao lr quando o agendador disparar (uma única vez).
    num_workers = 4  # Processos paralelos para carregamento de dados.

    # YOLO (ignorados se model != yolo)
    imgsz = 640  # Tamanho de entrada para detector YOLO (se usado).
    conf = 0.25  # Limiar mínimo de confiança das detecções YOLO.
    iou = 0.7  # Limiar de IoU para NMS no YOLO.

    # Tamanho do crop de entrada para SVTR / RodoSol warp
    warp_w = 256  # Largura do crop normalizado para OCR (SVTR).
    warp_h = 64  # Altura do crop normalizado para OCR (SVTR).

    # PARSeq
    # parseq_pretrained=True baixa pesos do PARSeq já treinados pelo autor em
    # grandes bases de scene-text (MJSynth, SynthText, COCO-Text, TextOCR, etc.)
    # via torch.hub ("baudm/parseq"). O treino aqui vira fine-tuning: o modelo
    # já sabe reconhecer texto em geral e só aprende o domínio das placas.
    # Se False, a arquitetura é instanciada com pesos aleatórios (treino do zero),
    # o que dá uma comparação mais justa com o SVTR, que não é pré-treinado.
    parseq_pretrained = True  # Carrega pesos pré-treinados do PARSeq.
    parseq_decode_ar = True  # Usa decodificação autoregressiva no PARSeq.
    parseq_refine_iters = 1  # Número de iterações de refinamento no PARSeq.

    return SimpleNamespace(
        run_name=run_name,
        model=model,
        dataset=dataset,
        data_root=data_root,
        split_path=split_path,
        out_dir=out_dir,
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        write_txt=write_txt,
        device=device,
        resume=resume,
        eval_only=eval_only,
        early_stop_patience=early_stop_patience,
        min_epochs=min_epochs,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        num_workers=num_workers,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        warp_w=warp_w,
        warp_h=warp_h,
        parseq_pretrained=parseq_pretrained,
        parseq_decode_ar=parseq_decode_ar,
        parseq_refine_iters=parseq_refine_iters,
    )


def executar_run(p: SimpleNamespace) -> None:
    print(f"Run: {p.run_name} | {p.dataset} | {p.model} | seed={p.seed}")
    if p.model == "svtr":
        from svtr.train import run_svtr
        run_svtr(p)
    elif p.model == "parseq":
        from parseq.train import run_parseq
        run_parseq(p)
    elif p.model == "crnn":
        from crnn.train import run_crnn
        run_crnn(p)
    else:
        raise NotImplementedError(f"Modelo não implementado: {p.model!r}")


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    for model in _MODELS:
        for dataset in _DATASETS:
            p = obter_parametros(model=model, dataset=dataset)
            p.out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== INÍCIO {p.run_name} ===")
            try:
                executar_run(p)
                results.append((p.run_name, True, "ok"))
                print(f"=== FIM {p.run_name} (OK) ===")
            except Exception as exc:
                results.append((p.run_name, False, str(exc)))
                print(f"=== FIM {p.run_name} (ERRO) ===")
                print(f"Motivo: {exc}")

    print("\nResumo final:")
    for run_name, ok, message in results:
        status = "OK" if ok else "ERRO"
        print(f"- {run_name}: {status} | {message}")


if __name__ == "__main__":
    main()
