"""Gera documentação visual do pipeline do projeto (TCC OCR de placas).

Saídas:
  - docs/pipeline.md  -> renderiza nativamente no Cursor/VS Code/GitHub
  - docs/pipeline.html -> arquivo único, abre em qualquer navegador

Re-rodar este script após mudanças no pipeline para atualizar a documentação.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"


def _count_split_lines() -> dict[str, int]:
    """Conta tracks por split lendo bj7/split.txt, se existir."""
    counts = {"training": 0, "validation": 0, "testing": 0}
    split_path = ROOT / "bj7" / "split.txt"
    if not split_path.exists():
        return counts
    for line in split_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ";" not in line:
            continue
        _, _, s = line.partition(";")
        if s in counts:
            counts[s] += 1
    return counts


def _build_markdown(counts: dict[str, int]) -> str:
    train_n = counts["training"]
    val_n = counts["validation"]
    test_n = counts["testing"]
    train_imgs = train_n * 10
    val_imgs = val_n * 10
    test_imgs = test_n * 5  # test só tem LR (5 imagens)

    return f"""# Pipeline completo — TCC OCR de placas (BJ7)

Documento gerado por `gerar_pipeline_doc.py`. Re-rodar o script para atualizar.

---

## Visão 1 — Estrutura do dataset BJ7

```mermaid
flowchart TB
    Root["bj7/"]
    Train["bj7/train/"]
    Test["bj7/test/"]
    Split["bj7/split.txt"]
    Maker["make_bj7_split.py (seed=42, 80/20)"]

    Root --> Train
    Root --> Test
    Root --> Split

    subgraph trainGroup [Train 20000 tracks HR+LR]
        TrA["Scenario-A/Brazilian (5000)"]
        TrAM["Scenario-A/Mercosur (5000)"]
        TrB["Scenario-B/Brazilian (2000)"]
        TrBM["Scenario-B/Mercosur (8000)"]
    end
    Train --> trainGroup

    subgraph testGroup [Test 3000 tracks so LR JPG]
        TestTracks["track_XXXXX com annotations.json + lr-001..005.jpg"]
    end
    Test --> testGroup

    Maker --> Split

    Split -->|"training: {train_n} tracks ({train_imgs} imgs)"| Split
    Split -->|"validation: {val_n} tracks ({val_imgs} imgs)"| Split
    Split -->|"testing: {test_n} tracks ({test_imgs} imgs LR)"| Split
```

**Cada track contém:**

- `annotations.json` com `plate_text` (GT) e `plate_layout`.
- `hr-001..005.{{png,jpg}}` (5 imagens HR) — só nos splits training/validation.
- `lr-001..005.{{png,jpg}}` (5 imagens LR) — todos os splits.

A heurística `ext = ".png" if (track_dir/"hr-001.png").exists() else ".jpg"` em `BJ7Dataset` resolve Scenario-A (PNG) vs Scenario-B (JPG) vs test (sem HR) automaticamente.

---

## Visão 2 — Pipeline de cada modelo (treino + dump)

```mermaid
flowchart TB
    Main["main.py - loop sobre _MODELS x _DATASETS"]
    Params["obter_parametros(model, dataset)"]
    Run["executar_run(p)"]
    Main --> Params --> Run

    Run -->|"model='svtr'"| RunSVTR["svtr/train.py run_svtr"]
    Run -->|"model='crnn'"| RunCRNN["crnn/train.py run_crnn"]
    Run -->|"model='parseq'"| RunPARSEQ["parseq/train.py run_parseq"]

    subgraph eachRun [Dentro de cada run_*]
        DSTrain["BJ7Dataset(training) - augmentation no PARSeq"]
        DSVal["BJ7Dataset(validation)"]
        DSTest["BJ7Dataset(testing)"]
        Model["Modelo: SVTRTiny / CRNN / parseq_tiny"]
        Loop["Loop de epocas: AdamW + clip_grad=5.0"]
        Sched["Scheduler: lr reduzido 1x apos min_epochs + lr_patience sem melhora"]
        Early["Early stop: apos lr_was_reduced + early_stop_patience sem melhora"]
        Best["bj7_{{model}}_best.pt (melhor val_acc)"]
        EvalTest["evaluate(test_loader) -> test_acc no log"]
        Dump["dump_test_predictions -> bj7_{{model}}_preds.csv"]

        DSTrain --> Loop
        DSVal --> Loop
        Loop --> Sched --> Early --> Best
        Best --> EvalTest
        Best --> Dump
        DSTest --> Dump
        DSTest --> EvalTest
    end

    RunSVTR --> eachRun
    RunCRNN --> eachRun
    RunPARSEQ --> eachRun
```

---

## Visão 3 — Parâmetros por modelo (defaults vs overrides)

| Parâmetro | CRNN / SVTR | PARSeq |
|---|---|---|
| `seed` | 42 | 42 |
| `batch_size` | 64 | 64 |
| `learning_rate` | 5e-4 | 5e-4 |
| `epochs` | 30 | **100** |
| `min_epochs` | 10 | **30** |
| `lr_patience` | 3 | **10** |
| `early_stop_patience` | 3 | **15** |
| `lr_factor` | 0.1 | **0.3** |
| Augmentation | não | **ColorJitter + RandomAffine + GaussianBlur** |
| Input size | 256x64 (SVTR) | **128x32** |
| Critério de early-stop | val_acc | **char_acc** |
| Variante | — | `parseq_tiny` (~6M, `pretrained=False`, `decode_ar=True`, `refine_iters=1`) |

**Por que PARSeq tem schedule diferente:** transformer from-scratch precisa de muito mais tempo pra convergir. Com schedule do CRNN/SVTR ele entra em colapso (loss trava em ~2.2, predições viram strings vazias ou ruído estatístico). Schedule longo + augmentation + critério char_acc destrava o aprendizado.

---

## Visão 4 — Conteúdo do CSV (camada que separa o que é fixo do que é reprocessável)

```mermaid
flowchart LR
    subgraph dumpFlow [Dentro de dump_test_predictions]
        Img["imagem (HR ou LR)"]
        Fwd["model forward -> logits"]
        Conf["greedy_decode_with_conf / predict_strings_with_conf"]
        Fmt["_format_conf"]
        Row["1 linha do CSV"]

        Img --> Fwd --> Conf --> Fmt --> Row
    end

    Final["logs/bj7_{{model}}/bj7_{{model}}_preds.csv"]
    Row --> Final
```

**Colunas do CSV:**

| Coluna | Tipo | Exemplo | Significado |
|---|---|---|---|
| `track_id` | str | `track_00001` | Pasta da track |
| `image_type` | str | `hr` ou `lr` | Resolução da imagem |
| `image_idx` | int | `1..5` | Índice dentro do tipo |
| `gt` | str | `ABC1D23` | Ground truth da placa |
| `pred` | str | `ABC1D23` | Predição do modelo |
| `conf_seq` | float | `0.8956` | Média geométrica das confs por char |
| `conf_chars` | str | `0.95\\|0.92\\|...` | Probs por char, separadas por `\\|` |

**Importante:** todo o resto da pipeline (fusão intra e inter) lê **só esses CSVs**. Você pode reescrever `fusion.py` quantas vezes quiser sem retreinar nenhum modelo.

---

## Visão 5 — Fusões intra-modelo e inter-modelos

```mermaid
flowchart TB
    subgraph csvs [3 CSVs - 1 por modelo, 15000 linhas cada]
        csvSVTR["bj7_svtr_preds.csv"]
        csvCRNN["bj7_crnn_preds.csv"]
        csvPAR["bj7_parseq_preds.csv"]
    end

    csvs -->|"load_preds_csv: agrupa por track_id"| Grouped["Por modelo: track_id -> gt, hr[], lr[]"]

    subgraph intra [compute_intra_model - HR e LR ISOLADOS]
        IntraHR["HR fusion: weighted_vote nas 5 (pred, conf_seq) HR"]
        IntraLR["LR fusion: weighted_vote nas 5 (pred, conf_seq) LR"]
        IntraHR --> AccHR["hr_fusion_acc"]
        IntraLR --> AccLR["lr_fusion_acc"]
    end
    Grouped --> intra

    subgraph inter [compute_inter_model - 1 voto por modelo]
        StepA["A: voto combinado HR+LR ponderado das 10 imgs por modelo"]
        StepB["B: 3 votos (1 por modelo) com peso = soma dos pesos da vencedora"]
        StepC["C: weighted_vote final entre os 3"]
        StepA --> StepB --> StepC
    end
    Grouped --> inter

    Out["logs/bj7_fusion_table.txt"]
    intra --> Out
    inter --> Out
```

### Onde HR e LR se misturam vs onde ficam isolados

| Etapa | HR isolado | LR isolado | Misturado |
|---|---|---|---|
| Treino dos modelos | — | — | sim (ambos viram amostras independentes) |
| Dump no CSV | — | — | sim (linhas com `image_type` próprio) |
| `compute_intra_model -> hr_fusion_acc` | **sim** | — | não |
| `compute_intra_model -> lr_fusion_acc` | — | **sim** | não |
| `compute_inter_model -> _combined_track_vote` | — | — | sim (intencionalmente) |
| Linha "HR fusion" da tabela | **sim** | — | não |
| Linha "LR fusion" da tabela | — | **sim** | não |
| Linha "Fusao inter-modelos" da tabela | — | — | sim |

A função `_combined_track_vote(data)` em `fusion.py` é a ÚNICA que mistura HR e LR (faz `data["hr"] + data["lr"]`), e ela só é usada na fusão inter-modelos como agregador por modelo antes do voto entre modelos.

---

## Visão 6 — Voto ponderado em detalhe

```mermaid
flowchart LR
    Input["[(pred1, conf1), (pred2, conf2), ...]"]
    Sum["totals[pred] += conf por entrada"]
    Max["max(totals.items(), key=peso)"]
    Out["(string vencedora, soma_dos_pesos_dela)"]

    Input --> Sum --> Max --> Out

    Empate["Empate exato: max preserva ordem de insercao - primeira vence"]
    Max -.-> Empate
```

**Exemplo prático** (track GT = `ABC1D23`):

| Predição | Conf | Contribui pra... |
|---|---|---|
| `ABC1D23` | 0.95 | total[`ABC1D23`] += 0.95 |
| `ABC1D23` | 0.92 | total[`ABC1D23`] += 0.92 |
| `ABE1D23` | 0.30 | total[`ABE1D23`] += 0.30 |
| `ABC1D23` | 0.97 | total[`ABC1D23`] += 0.97 |
| `ABE1D23` | 0.30 | total[`ABE1D23`] += 0.30 |

Totais: `ABC1D23 = 2.84` vs `ABE1D23 = 0.60` → vence `ABC1D23`.

---

## Visão 7 — Pipeline completo de ponta a ponta

```mermaid
flowchart TB
    A["bj7/ (dataset cru)"]
    B["make_bj7_split.py"]
    C["bj7/split.txt"]
    D["BJ7Dataset (3 modulos)"]
    E1["run_svtr"]
    E2["run_crnn"]
    E3["run_parseq"]
    F1["bj7_svtr_preds.csv"]
    F2["bj7_crnn_preds.csv"]
    F3["bj7_parseq_preds.csv"]
    G["fusion.run_fusion"]
    H["bj7_fusion_table.txt"]

    A --> B --> C
    A --> D
    C --> D
    D --> E1 --> F1
    D --> E2 --> F2
    D --> E3 --> F3
    F1 --> G
    F2 --> G
    F3 --> G
    G --> H

    L1["logs/bj7_*/*_log.txt"]
    L2["logs/bj7_*/*_best.pt"]
    E1 -.-> L1
    E1 -.-> L2
    E2 -.-> L1
    E2 -.-> L2
    E3 -.-> L1
    E3 -.-> L2
```

---

## Resumo das garantias de isolamento

1. **HR vs LR no treino:** misturados (ambos são amostras independentes do mesmo `BJ7Dataset` com `image_type` na metadata). O modelo aprende dos dois sem distinção.
2. **HR vs LR no dump:** cada imagem vira 1 linha do CSV com `image_type` próprio. Não há mistura no arquivo.
3. **HR vs LR na fusão intra:** explicitamente separados — `data["hr"]` e `data["lr"]` são listas distintas no dict por track, processadas em duas chamadas independentes de `weighted_vote`.
4. **HR vs LR na fusão inter:** misturados intencionalmente — cada modelo precisa dar 1 voto único por track, então `_combined_track_vote` concatena `hr + lr`.
5. **Modelos não se contaminam:** cada modelo treina em isolamento total, salva seu próprio checkpoint, gera seu próprio CSV. A fusão inter só acontece em pós-processamento lendo os 3 CSVs.

---

## Estrutura de arquivos

```
ml/
├── main.py                          # orquestrador
├── make_bj7_split.py                # gera bj7/split.txt
├── fusion.py                        # fusão intra/inter ponderada
├── gerar_pipeline_doc.py            # gera este documento
│
├── bj7/                             # dataset
│   ├── train/Scenario-{{A,B}}/{{Brazilian,Mercosur}}/track_XXXXX/
│   ├── test/track_XXXXX/
│   └── split.txt
│
├── crnn/ svtr/ parseq/              # 3 modelos
│   ├── dataset.py                   # BJ7Dataset + RodoSolDataset + augmentations
│   ├── model.py                     # arquitetura
│   └── train.py                     # run_*, evaluate, dump_test_predictions
│
└── logs/bj7_{{model}}/
    ├── bj7_{{model}}_log.txt          # curva de treino (epoch, loss, acc)
    ├── bj7_{{model}}_best.pt          # checkpoint do melhor val_acc
    └── bj7_{{model}}_preds.csv        # predições + confianças no test set
```
"""


def _build_html(markdown_body: str) -> str:
    """Embute o markdown num HTML standalone que renderiza Mermaid via CDN."""
    # Escapa caracteres HTML problemáticos para inserir em <script type="text/markdown">.
    escaped = markdown_body.replace("</script>", "<\\/script>")
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Pipeline TCC - BJ7</title>
<style>
  body {{ max-width: 1100px; margin: 2em auto; padding: 0 1em;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.55; color: #1f2328; background: #fff; }}
  h1, h2, h3 {{ border-bottom: 1px solid #d0d7de; padding-bottom: .2em; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #d0d7de; padding: .35em .7em; }}
  th {{ background: #f6f8fa; }}
  code {{ background: #f6f8fa; padding: .1em .3em; border-radius: 3px; font-size: .92em; }}
  pre {{ background: #f6f8fa; padding: 1em; border-radius: 6px; overflow-x: auto; }}
  .mermaid {{ background: #fff; padding: 1em 0; }}
  hr {{ border: 0; border-top: 1px solid #d0d7de; margin: 2em 0; }}
</style>
</head>
<body>
<div id="content"></div>

<script type="text/markdown" id="md">{escaped}</script>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  const src = document.getElementById("md").textContent;
  // marked: bloco ```mermaid``` vira <pre><code class="language-mermaid">...
  const renderer = new marked.Renderer();
  renderer.code = (code, lang) => {{
    if (lang === "mermaid") {{
      return `<div class="mermaid">${{code}}</div>`;
    }}
    return `<pre><code>${{code}}</code></pre>`;
  }};
  document.getElementById("content").innerHTML = marked.parse(src, {{ renderer }});
  mermaid.initialize({{ startOnLoad: true, theme: "default", securityLevel: "loose" }});
  mermaid.run();
</script>
</body>
</html>
"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    counts = _count_split_lines()

    md = _build_markdown(counts)
    html = _build_html(md)

    md_path = DOCS / "pipeline.md"
    html_path = DOCS / "pipeline.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    print(f"OK - {md_path.relative_to(ROOT)} ({len(md):,} chars)")
    print(f"OK - {html_path.relative_to(ROOT)} ({len(html):,} chars)")
    print()
    print("Para visualizar:")
    print(f"  - Cursor/VS Code: abrir {md_path.relative_to(ROOT)} e usar 'Open Preview'")
    print(f"  - Navegador:      abrir {html_path.relative_to(ROOT)} (clique duplo)")
    if sum(counts.values()) == 0:
        print()
        print("AVISO: bj7/split.txt nao foi encontrado, contagens estao zeradas no diagrama.")
        print("       Rode 'python make_bj7_split.py' primeiro.")


if __name__ == "__main__":
    main()
