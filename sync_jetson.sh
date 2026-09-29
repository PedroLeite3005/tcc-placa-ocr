#!/usr/bin/env bash
# Envia pra Jetson tudo que é gitignored (checkpoints .pt e dataset de teste),
# via scp/ssh direto — sem passar pelo git, que não é feito pra binário pesado
# (ver discussão no histórico: git não faz diff de .pt, e cada retreino vira
# um blob novo permanente no histórico).
#
# O código em si continua indo por `git pull` na Jetson normalmente.
#
# Configuração: exporte JETSON_HOST antes de rodar (ou edite o default abaixo).
#   export JETSON_HOST=pedrobastos@192.168.0.42
#
# Uso:
#   ./sync_jetson.sh ckpt bj7_crnn                # 1 checkpoint
#   ./sync_jetson.sh ckpt bj7_crnn bj7_svtr        # vários de uma vez
#   ./sync_jetson.sh testset                       # bj7/split.txt + bj7/test/ (~75MB)
#   ./sync_jetson.sh hubcache                      # cache do torch.hub do PARSeq (strhub já patchado p/ Python 3.6)
#   ./sync_jetson.sh all bj7_crnn                  # testset + hubcache + ckpt num comando só

set -euo pipefail

JETSON_HOST="${JETSON_HOST:-pedrobastos@jetson.local}"
JETSON_DIR="${JETSON_DIR:-/mnt/ssd/projeto-tcc2/tcc-placa-ocr}"
HUB_CACHE_DIR="${HUB_CACHE_DIR:-$HOME/.cache/torch/hub/baudm_parseq_main}"

cmd_ckpt() {
    for run_name in "$@"; do
        local ckpt="logs/${run_name}/${run_name}_best.pt"
        if [[ ! -f "$ckpt" ]]; then
            echo "Aviso: $ckpt não existe, pulando." >&2
            continue
        fi
        echo "Enviando $ckpt ($(du -h "$ckpt" | cut -f1))..."
        ssh "$JETSON_HOST" "mkdir -p $JETSON_DIR/logs/${run_name}"
        scp "$ckpt" "$JETSON_HOST:$JETSON_DIR/logs/${run_name}/"
    done
}

cmd_testset() {
    echo "Empacotando bj7/split.txt + bj7/test/ ($(du -sh bj7/test | cut -f1))..."
    local tmp_tar
    tmp_tar="$(mktemp --suffix=.tar.gz)"
    tar -czf "$tmp_tar" -C bj7 split.txt test

    echo "Enviando $(du -h "$tmp_tar" | cut -f1) para a Jetson..."
    ssh "$JETSON_HOST" "mkdir -p $JETSON_DIR/bj7"
    scp "$tmp_tar" "$JETSON_HOST:/tmp/bj7_testset.tar.gz"
    ssh "$JETSON_HOST" "tar -xzf /tmp/bj7_testset.tar.gz -C $JETSON_DIR/bj7 && rm /tmp/bj7_testset.tar.gz"
    rm "$tmp_tar"
    echo "OK — bj7/split.txt e bj7/test/ prontos na Jetson."
}

cmd_hubcache() {
    if [[ ! -d "$HUB_CACHE_DIR" ]]; then
        echo "Aviso: $HUB_CACHE_DIR não existe (rode load_parseq() localmente pra popular o cache antes)." >&2
        return 1
    fi
    echo "Empacotando cache do torch.hub ($(du -sh "$HUB_CACHE_DIR" | cut -f1)) — strhub já deve estar patchado p/ Python 3.6 (ver README seção 14)..."
    local tmp_tar
    tmp_tar="$(mktemp --suffix=.tar.gz)"
    tar --exclude="__pycache__" -czf "$tmp_tar" -C "$(dirname "$HUB_CACHE_DIR")" "$(basename "$HUB_CACHE_DIR")"

    echo "Enviando $(du -h "$tmp_tar" | cut -f1) para a Jetson..."
    ssh "$JETSON_HOST" "mkdir -p ~/.cache/torch/hub"
    scp "$tmp_tar" "$JETSON_HOST:/tmp/hubcache.tar.gz"
    ssh "$JETSON_HOST" "tar -xzf /tmp/hubcache.tar.gz -C ~/.cache/torch/hub && rm /tmp/hubcache.tar.gz"
    rm "$tmp_tar"
    echo "OK — cache do torch.hub (strhub patchado) pronto na Jetson."
}

cmd_all() {
    cmd_testset
    cmd_hubcache
    cmd_ckpt "$@"
}

case "${1:-}" in
    ckpt) shift; cmd_ckpt "$@" ;;
    testset) cmd_testset ;;
    hubcache) cmd_hubcache ;;
    all) shift; cmd_all "$@" ;;
    *)
        echo "Uso: $0 {ckpt <run_name>... | testset | hubcache | all <run_name>...}" >&2
        echo "Exemplo: $0 all bj7_crnn" >&2
        exit 1
        ;;
esac
