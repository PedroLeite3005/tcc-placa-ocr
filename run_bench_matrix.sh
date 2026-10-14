#!/usr/bin/env bash
# Roda a matriz inteira de benchmarks na Jetson (ou no desktop) de uma vez:
# todo checkpoint em logs/bj7_*/ × compressão (none/fp16/int8) × decodificação
# do PARSeq (ar/nar/nar1). Gera:
#   logs/jetson/bench_all.csv        — 1 linha por run (acurácia, FPS, energia, tamanho...)
#   logs/jetson/preds_<run>_<modo>.csv — predições por imagem (entrada do fusion.py)
#
# Feito pra rodar desacoplado da sessão SSH (pode desligar o desktop):
#   mkdir -p logs/jetson && nohup bash run_bench_matrix.sh > logs/jetson/matrix.log 2>&1 &
#   tail -f logs/jetson/matrix.log
#
# Antes de rodar na Jetson, fixar o modo de energia (mesmo em todos os runs!):
#   sudo nvpmodel -m 0 && sudo jetson_clocks
#
# Variáveis (opcionais):
#   LIMIT=200            limita imagens por run (padrão: split de teste inteiro)
#   INT8_LIMIT=1000      limite só pro INT8 (roda na CPU, é bem mais lento; vazio = sem limite)
#   MODES="none fp16 int8"
#   PARSEQ_DECODES="ar nar nar1"   ar=autoregressivo | nar=paralelo sem refino | nar1=paralelo + 1 refino
#   PARSEQ_LEGACY=1      bj7_parseq em cor (e seus podados) usam --img-h 32 --img-w 128
#   HARDWARE=jetson      (padrão: autodetecta)
#   SKIP_DONE=1          pula runs cujo preds_*.csv já existe (permite retomar)
#   OUT_DIR=logs/jetson

set -u

OUT_DIR="${OUT_DIR:-logs/jetson}"
LIMIT="${LIMIT:-}"
INT8_LIMIT="${INT8_LIMIT-1000}"
MODES="${MODES:-none fp16 int8}"
PARSEQ_DECODES="${PARSEQ_DECODES:-ar nar nar1}"
PARSEQ_LEGACY="${PARSEQ_LEGACY:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
VENV="${VENV:-/mnt/ssd/projeto-tcc2/venv}"

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export OPENBLAS_CORETYPE=ARMV8
mkdir -p "$OUT_DIR"
BENCH="$OUT_DIR/bench_all.csv"
HW_ARGS=()
[ -n "${HARDWARE:-}" ] && HW_ARGS=(--hardware "$HARDWARE")

command -v nvpmodel >/dev/null 2>&1 && echo "Modo de energia: $(nvpmodel -q 2>/dev/null | head -1)"
echo "Início: $(date) | OUT_DIR=$OUT_DIR | LIMIT=${LIMIT:-todos} | INT8_LIMIT=${INT8_LIMIT:-todos}"

n_ok=0; n_fail=0; n_skip=0
for ckpt in logs/bj7_*/bj7_*_best.pt; do
    [ -f "$ckpt" ] || continue
    run="$(basename "$(dirname "$ckpt")")"          # ex: bj7_svtr_bin_prune50
    model="$(echo "$run" | cut -d_ -f2)"            # crnn | svtr | parseq
    case "$model" in crnn|svtr|parseq) ;; *) continue ;; esac

    extra=()
    [[ "$run" == *_bin* ]] && extra+=(--binarize)
    if [ "$model" = "parseq" ] && [ "$PARSEQ_LEGACY" = "1" ] && [[ "$run" != *_bin* ]]; then
        extra+=(--img-h 32 --img-w 128)
    fi

    # Checkpoint podado: a compressão já está nele — roda só "none".
    modes="$MODES"
    [[ "$run" == *_prune* ]] && modes="none"
    decodes="ctc"
    [ "$model" = "parseq" ] && decodes="$PARSEQ_DECODES"

    for mode in $modes; do
        for dec in $decodes; do
            tag="${run}_${mode}"
            dec_args=()
            if [ "$dec" = "nar" ]; then
                dec_args=(--parseq-nar --parseq-refine-iters 0); tag="${tag}_nar"
            elif [ "$dec" = "nar1" ]; then
                dec_args=(--parseq-nar --parseq-refine-iters 1); tag="${tag}_nar1"
            elif [ "$dec" = "ar" ]; then
                tag="${tag}_ar"
            fi
            preds="$OUT_DIR/preds_${tag}.csv"
            if [ "$SKIP_DONE" = "1" ] && [ -f "$preds" ]; then
                echo "[skip] $tag (já existe)"; n_skip=$((n_skip + 1)); continue
            fi
            lim="$LIMIT"
            [ "$mode" = "int8" ] && [ -n "$INT8_LIMIT" ] && lim="$INT8_LIMIT"
            lim_args=()
            [ -n "$lim" ] && lim_args=(--limit "$lim")

            echo "=== $(date +%H:%M:%S) $tag ==="
            if python3 -m "$model.infer" --ckpt "$ckpt" --dataset bj7 --split testing \
                --batch-size 1 --device cuda --compress "$mode" \
                "${extra[@]}" "${dec_args[@]}" "${lim_args[@]}" "${HW_ARGS[@]}" \
                --out-csv "$BENCH" --dump-preds "$preds"; then
                n_ok=$((n_ok + 1))
            else
                echo "[FALHOU] $tag"; n_fail=$((n_fail + 1))
            fi
        done
    done
done

echo "Fim: $(date) | ok=$n_ok falhas=$n_fail pulados=$n_skip | resultados em $BENCH"
