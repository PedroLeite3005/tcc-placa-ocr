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
#   MODES="none fp16"    padrão sem int8: o PyTorch da Jetson não tem motor de quantização
#                        (erro NoQEngine) — "int8" só funciona no desktop
#   INT8_LIMIT=1000      limite só pro INT8, se incluído em MODES (roda na CPU, é lento)
#                        TensorRT: incluir "trt_fp16" e/ou "trt_fp32" em MODES. O motor é
#                        procurado em $TRT_DIR/<run>[_<decode>]_<fp16|fp32>.engine (gerado
#                        por build_trt_engines.sh); se não existir, o run é pulado.
#   TRT_DIR=logs/trt
#   PARSEQ_DECODES="ar nar nar1"   ar=autoregressivo | nar=paralelo sem refino | nar1=paralelo + 1 refino
#   ONLY="bj7_parseq bj7_parseq_legado"   roda só esses runs (nomes das pastas em logs/)
#   SKIP_PRUNED=1        pula os checkpoints podados (pruning não muda FPS)
#   COOLDOWN_C=55        espera a placa esfriar abaixo de N °C antes de cada run, pra todos
#                        começarem nas mesmas condições térmicas (vazio = não espera)
#   HARDWARE=jetson      (padrão: autodetecta)
#   SKIP_DONE=1          pula runs cujo preds_*.csv já existe (permite retomar)
#   OUT_DIR=logs/jetson
#
# Resolução do PARSeq: runs com "_legado" no nome (bj7_parseq_legado e seus
# podados) são o modelo antigo em 32×128 e rodam com --img-h 32 --img-w 128;
# os demais (bj7_parseq, bj7_parseq_bin) usam o padrão 64×256.

set -u

OUT_DIR="${OUT_DIR:-logs/jetson}"
LIMIT="${LIMIT:-}"
INT8_LIMIT="${INT8_LIMIT-1000}"
MODES="${MODES:-none fp16}"
PARSEQ_DECODES="${PARSEQ_DECODES:-ar nar nar1}"
ONLY="${ONLY:-}"
SKIP_PRUNED="${SKIP_PRUNED:-0}"
COOLDOWN_C="${COOLDOWN_C:-}"
TRT_DIR="${TRT_DIR:-logs/trt}"
SKIP_DONE="${SKIP_DONE:-1}"
VENV="${VENV:-/mnt/ssd/projeto-tcc2/venv}"

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export OPENBLAS_CORETYPE=ARMV8
mkdir -p "$OUT_DIR"
BENCH="$OUT_DIR/bench_all.csv"
HW_ARGS=()
[ -n "${HARDWARE:-}" ] && HW_ARGS=(--hardware "$HARDWARE")

max_temp_c() {
    # Maior temperatura entre os sensores térmicos (em °C); vazio se não houver sensores.
    local t
    t="$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)"
    [ -n "$t" ] && echo $((t / 1000))
}

wait_cool() {
    [ -n "$COOLDOWN_C" ] || return 0
    local t
    t="$(max_temp_c)"
    [ -n "$t" ] || return 0
    if [ "$t" -gt "$COOLDOWN_C" ]; then
        echo "[cooldown] ${t}°C — esperando baixar de ${COOLDOWN_C}°C..."
        while [ "$(max_temp_c)" -gt "$COOLDOWN_C" ]; do sleep 10; done
        echo "[cooldown] ok: $(max_temp_c)°C"
    fi
}

command -v nvpmodel >/dev/null 2>&1 && echo "Modo de energia: $(nvpmodel -q 2>/dev/null | grep -i 'power mode')"
[ -r /sys/devices/pwm-fan/target_pwm ] && echo "Ventoinha (target_pwm 0-255): $(cat /sys/devices/pwm-fan/target_pwm)"
echo "Início: $(date) | OUT_DIR=$OUT_DIR | LIMIT=${LIMIT:-todos} | MODES=$MODES | ONLY=${ONLY:-todos} | SKIP_PRUNED=$SKIP_PRUNED | COOLDOWN_C=${COOLDOWN_C:-desligado}"

n_ok=0; n_fail=0; n_skip=0
for ckpt in logs/bj7_*/bj7_*_best.pt; do
    [ -f "$ckpt" ] || continue
    run="$(basename "$(dirname "$ckpt")")"          # ex: bj7_svtr_bin_prune50
    model="$(echo "$run" | cut -d_ -f2)"            # crnn | svtr | parseq
    case "$model" in crnn|svtr|parseq) ;; *) continue ;; esac
    [ "$SKIP_PRUNED" = "1" ] && [[ "$run" == *_prune* ]] && continue
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $run "* ]]; then
        continue
    fi

    extra=()
    [[ "$run" == *_bin* ]] && extra+=(--binarize)
    [[ "$run" == *_legado* ]] && extra+=(--img-h 32 --img-w 128)

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

            # TensorRT: motor com o mesmo nome do ONNX (run [+ decodificação do PARSeq]).
            trt_args=()
            if [[ "$mode" == trt_* ]]; then
                prec="${mode#trt_}"
                ename="$run"
                [ "$dec" != "ctc" ] && ename="${run}_${dec}"
                engine="$TRT_DIR/${ename}_${prec}.engine"
                if [ ! -f "$engine" ]; then
                    echo "[skip] $tag (sem motor $engine — rodar build_trt_engines.sh)"
                    n_skip=$((n_skip + 1)); continue
                fi
                trt_args=(--trt-engine "$engine")
            fi

            wait_cool
            echo "=== $(date +%H:%M:%S) $tag ($(max_temp_c)°C) ==="
            if python3 -m "$model.infer" --ckpt "$ckpt" --dataset bj7 --split testing \
                --batch-size 1 --device cuda --compress "$mode" \
                "${extra[@]}" "${dec_args[@]}" "${lim_args[@]}" "${HW_ARGS[@]}" "${trt_args[@]}" \
                --out-csv "$BENCH" --dump-preds "$preds"; then
                n_ok=$((n_ok + 1))
            else
                echo "[FALHOU] $tag"; n_fail=$((n_fail + 1))
            fi
        done
    done
done

echo "Fim: $(date) | ok=$n_ok falhas=$n_fail pulados=$n_skip | resultados em $BENCH"
