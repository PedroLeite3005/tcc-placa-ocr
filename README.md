# TCC — OCR de placas: processo experimental

Este repositório descreve o **protocolo** para comparar modelos em **dois datasets**, sem misturar dados entre corpora.

## Escopo

- **Tarefa:** reconhecimento de caracteres da placa (sequência alfanumérica).
- **Dois datasets independentes:** resultados são comparados **entre** corpora; não há treino com amostras misturadas.
- **Execução em lote:** combinações `model × dataset` em sequência (mesma ordem de etapas; pesos e dataloaders separados por run).

## Decisões congeladas (checklist)

Valores abaixo foram **fixados** para o TCC; alterar qualquer item implica **nova bateria** documentada.

| Tema | Decisão |
|------|---------|
| **Seed BJ7 (split)** | `42` |
| **Proporção de tracks BJ7** | 40% treino / 20% val / 40% teste; **largest remainder (Hamilton)** para contagens inteiras que somam exatamente o número de *tracks* |
| **Sorteio BJ7** | **Estratificar** por pasta de **layout** no path (`train/<cenário>/<layout>/track_*`), mantendo proporção de layouts o mais equilibrada possível nos três conjuntos |
| **Imagens BJ7** | **Todas** `lr-*` e `hr-*` de cada *track* no conjunto ao qual o *track* pertence |
| **Warp RodoSol** | Homografia para retângulo **256 × 64** px |
| **Ordem dos `corners` (RodoSol)** | Assumir **top-left → top-right → bottom-right → bottom-left** na ordem em que os quatro pares aparecem no `.txt` |
| **Após warp RodoSol** | **Sem** segundo redimensionamento obrigatório (entrada dos modelos neste dataset = **256×64** após o warp) |
| **Entrada BJ7** | Recortes nativos `lr`/`hr` (tamanhos variam); cada implementação pode redimensionar **internamente** conforme o modelo exige, desde que **fixo por modelo** e documentado no run |
| **Normalização do texto (gt e predição)** | Maiúsculas; manter **somente** `A–Z` e `0–9` (remover `·`, hífen, espaços e qualquer outro símbolo) |
| **Charset do modelo** | `A–Z` e `0–9` |
| **Layouts Brazilian / Mercosur** | **Um modelo por dataset** treinado com **ambos** os layouts juntos; comprimentos variáveis conforme o rótulo |
| **Métrica principal (8 runs)** | **Acurácia em nível de placa:** predição normalizada **idêntica** ao gt normalizado (100% dos caracteres corretos). **Sem CER** nesta fase do trabalho |
| **Erros por caractere** | Opcional só para **análise qualitativa** (log); **não** entra como métrica oficial por enquanto |
| **Early stopping** | Paciência **20** épocas; monitorar **loss de validação** |
| **Seed global** | `42` para `random` / `numpy` / `torch` (e CUDA determinístico quando possível), **além** da seed do split BJ7 |
| **Execução atual no código** | `SVTR` e `PARSeq` ativos no `main.py`; `YOLO/CRNN` permanecem como referência metodológica |

**Ainda a documentar na implementação:** hiperparâmetros finais de cada bateria longa (épocas máximas, LR, batch) por run congelado.

## Matriz de experimentos

Cada linha é um treino + avaliação fechados; nomes de run seguem `<dataset>_<modelo>`.

| ID | Dataset | Modelo | Nome do run |
|----|---------|--------|-------------|
| 1 | wYe7pBJ7-train | SVTR | `bj7_svtr` |
| 2 | wYe7pBJ7-train | PARSeq | `bj7_parseq` |
| 3 | RodoSol-ALPR | SVTR | `rodosol_svtr` |
| 4 | RodoSol-ALPR | PARSeq | `rodosol_parseq` |

> Execução padrão atual: `models = ["svtr", "parseq"]` e `datasets = ["rodosol", "bj7"]` no `main.py`.

## Dados — `wYe7pBJ7-train`

- Estrutura típica: `train/<cenário>/<layout>/track_<id>/`.
- **`annotations.json`** por *track:* `plate_text`, `plate_layout`, `corners` por arquivo de imagem.
- Imagens: **`lr-*.png`** (low resolution) e **`hr-*.png`** (high resolution).
- **Congelado:** em cada conjunto entram **todas** as imagens `lr-*` e `hr-*` dos *tracks* daquele conjunto.

## Dados — RodoSol (`tbFcZE-RodoSol-ALPR/tbFcZE-RodoSol-ALPR/`)

- Imagens: `images/<subpasta>/img_XXXXXX.jpg` (ver `README.txt` do dataset).
- Rótulos: `img_XXXXXX.txt` com `plate`, `layout`, `corners`, etc.
- Resolução das cenas: **1280×720**.
- **Congelado:** *warp* com homografia para **256×64** px; ordem dos pontos **TL, TR, BR, BL** conforme sequência no `.txt`; sem redimensionamento extra obrigatório após o warp.
- Aplicar a **mesma** função de normalização de `plate` definida na tabela acima.

## Splits (40% / 20% / 40%)

### BJ7 — por *track*

- Proporção sobre **tracks**; **largest remainder**; **seed `42`**; **estratificação por pasta `<layout>`**.
- Artefato versionado: ex. `splits/bj7_tracks_40_20_40.json` com listas de chaves de *track*, `random_seed`, data ou hash.
- *Asserts:* interseção vazia entre treino, val e teste.

### RodoSol — `split.txt`

- Usar **`split.txt`**: `caminho_relativo.jpg;training|validation|test`.
- Proporção oficial: 8k / 4k / 8k (40% / 20% / 40%).

## Treino e avaliação

- **Validação:** *checkpoints* e early stopping pela **loss de validação** (paciência 20).
- **Teste:** **acurácia de placa** (match exato pós-normalização); métricas finais só no **teste**.
- Registrar por run: versão do código, seeds, split, hiperparâmetros, melhor *checkpoint*.
- Saídas por run em `logs/<dataset>_<modelo>/` (ex.: `logs/rodosol_parseq/`).

## Execução atual no código

- Entrada única: `python main.py` (sem argumentos de CLI).
- O script percorre a fila de `model × dataset` e executa em sequência.
- Se um run falhar, os próximos continuam; ao final é impresso um resumo de status.

## Reprodutibilidade

- Ambiente: Python, PyTorch, CUDA e libs do projeto — instalar com `pip install -r requirements.txt`.
- **Seed global `42`**.
- Artefatos em `logs/<nome_do_run>/`.

## Documentação enxuta para IAs

Para colar em prompts de assistentes, usar [context.md](context.md).
