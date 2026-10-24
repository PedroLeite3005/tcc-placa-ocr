#!/usr/bin/env bash
# Compila cada logs/onnx/*.onnx num motor TensorRT (logs/trt/<nome>_<fp16|fp32>.engine).
# Rodar NA JETSON: o motor é específico da GPU e da versão do TensorRT — um motor
# gerado em outra máquina não carrega.
#
#   mkdir -p logs/trt && nohup bash build_trt_engines.sh > logs/trt/build.log 2>&1 &
#   tail -f logs/trt/build.log
#
# Não rodar junto com benchmarks: a compilação usa a GPU e estraga as medições.
#
# Variáveis (opcionais):
#   PRECISIONS="fp16 fp32"   precisões a compilar (fp16 é o que importa; fp32 isola o
#                            ganho do TensorRT sem a redução de precisão)
#   ONLY="bj7_crnn bj7_svtr" só esses ONNX (nome do arquivo sem .onnx)
#   ONNX_DIR=logs/onnx  TRT_DIR=logs/trt
#
# Cada compilação grava o log completo do trtexec em logs/trt/<nome>_<prec>.log
# (se falhar, a causa está lá). Motores já existentes são pulados.

set -u

ONNX_DIR="${ONNX_DIR:-logs/onnx}"
TRT_DIR="${TRT_DIR:-logs/trt}"
PRECISIONS="${PRECISIONS:-fp16 fp32}"
ONLY="${ONLY:-}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

mkdir -p "$TRT_DIR"
echo "Início: $(date) | ONNX_DIR=$ONNX_DIR | PRECISIONS=$PRECISIONS | ONLY=${ONLY:-todos}"

n_ok=0; n_fail=0; n_skip=0
for onnx in "$ONNX_DIR"/*.onnx; do
    [ -f "$onnx" ] || { echo "Nenhum .onnx em $ONNX_DIR (rodar export_onnx.py no desktop e ./sync_jetson.sh onnx)"; exit 1; }
    name="$(basename "$onnx" .onnx)"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then
        continue
    fi
    for prec in $PRECISIONS; do
        engine="$TRT_DIR/${name}_${prec}.engine"
        log="$TRT_DIR/${name}_${prec}.log"
        if [ -f "$engine" ]; then
            echo "[skip] $engine (já existe)"; n_skip=$((n_skip + 1)); continue
        fi
        flags=()
        [ "$prec" = "fp16" ] && flags=(--fp16)
        echo "=== $(date +%H:%M:%S) $name ($prec) ==="
        if "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine" --workspace=1024 "${flags[@]}" > "$log" 2>&1 && [ -f "$engine" ]; then
            # O trtexec já faz um benchmark rápido depois de compilar: dá uma prévia do ganho.
            echo "OK  $engine ($(du -h "$engine" | cut -f1)) | $(grep -m1 -E 'Throughput:' "$log" | sed 's/.*Throughput: */throughput trtexec: /')"
            n_ok=$((n_ok + 1))
        else
            echo "[FALHOU] $name ($prec) — últimas linhas de $log:"
            grep -E "ERROR|Error|error" "$log" | tail -5
            n_fail=$((n_fail + 1))
        fi
    done
done

echo "Fim: $(date) | ok=$n_ok falhas=$n_fail pulados=$n_skip | motores em $TRT_DIR"
