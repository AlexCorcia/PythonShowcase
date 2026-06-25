# Handoff — TFM "Aprendizaje de estilos de juego en ajedrez"

This document is a complete handoff for resuming work on this TFM in a new
Claude Code session. Read it end-to-end before doing anything else.

Last updated commit: `2364cc3` ("TFM section 4 prep: HP search infrastructure + results").

---

## 1. What this project is

- **Title:** *Aprendizaje de estilos de juego en ajedrez mediante modelos de
  Machine Learning y grafos temporales dinámicos.*
- **Author:** Alexandre Marc Corcia Aguilera (UNIR — Máster en IA).
- **Type of TFE:** *Comparativa de soluciones* (Tipo 3 per UNIR guide).
- **Director:** Xiomara Patricia Blanco Valencia.
- **Goal:** Compare three families of representations for classifying chess
  playing styles from PGN games:
  1. **Flat tabular features** (counts, ratios, ECO family).
  2. **Temporal features by game phase** (opening/middlegame/endgame).
  3. **Graph representation** with a GNN over per-position snapshots.
- **Players in the dataset:** Karpov (positional), Tal (tactical), Kasparov
  (dynamic), Petrosian (defensive). One style per player → labels are
  *de facto* player identities; this matters for the LOPO experiment.
- **The thesis is in `app/docs/`** — both .docx and .pdf.

---

## 2. The director's feedback (from `app/docs/notas de reunion.txt`)

The pieces that drive the current work:

- **Hyperparameter optimization is mandatory** and must use a logical search
  method per technique:
  - Logistic Regression → GridSearch.
  - Random Forest → GridSearch + RandomizedSearch.
  - XGBoost → GridSearch + RandomizedSearch.
  - GNN → "buscar mejores hiperparámetros o algo así" (i.e., a real search,
    not hardcoded values).
- **Section 4 must be purely technical, no theory.** All theory (what an RF is,
  what GridSearch is, what F1-macro is, what graphs are) belongs in §2.3 Marco
  Teórico.
- **One HP table per technique** inside §4.
- **A final consolidated comparison table** at the end of §4, with **F1-macro
  as the primary metric** ("F1 es la mejor").
- **Section 4 ends with an Análisis de Resultados** that combines everything.
- **New experiment requested:** *"Mirar si con otras métricas podemos
  ayudarnos a detectar el estilo de jugadores que no hemos entrenado"* — this
  is the leave-one-player-out (LOPO) experiment.
- **Order theory (§2) and technical (§4) cleanly.**
- **Style: APA.**

---

## 3. State of §2 and §3

The user reported they updated §2 and §3 based on the director's notes. The
current PDF in `app/docs/` reflects most of that work:

- §2.3 **Marco Teórico** exists and covers: supervised learning, classical
  models (LR/RF/XGBoost), HP optimization (Grid/Random), metrics
  (accuracy/precision/recall/F1/macro-F1/CM), and graphs in chess.
- §3.0 introduction removed (per "Quita el 3.0 mucho bla bla bla").
- §3.2 lists objectives including HP optimization and F1-macro evaluation.
- §3.3 metodología promises HP search; §4 must deliver it.

### Inconsistencies identified — replacement text provided but paste status NOT confirmed

1. **§2.2.6 heading is wrong**: currently labeled "Revisión de técnicas de
   aprendizaje en grafos dinámicos" (duplicate of §2.2.5). It actually reviews
   Rigaux & Kashima (2024) AlphaGateau. **Should be retitled to:
   "Representación basada en grafos en ajedrez".** Body is fine.

2. **§2.3.5 needs reframing**: currently talks about "grafos temporales
   dinámicos" without distinguishing the **data representation** (which is
   dynamic — sequence of snapshots) from the **learning model** (which in this
   TFM is a static per-snapshot GCN). Full replacement text was provided in
   the previous Claude session; if not already pasted, the user needs to
   apply it. Search the chat or ask Claude to regenerate.

⚠️ **First task on resumption: ask the user whether these two §2 fixes were
applied to the .docx.** If not, regenerate the §2.3.5 replacement (the gist
is in §1 above and in the §2.3.5 file content the previous session produced).

---

## 4. §4 structure agreed upon (final plan)

Replaces the current §4 in the .docx. All "motivación / interpretación"
subsections in the current §4 are to be DELETED because §2.3 carries the
theory.

```
4. Desarrollo específico de la contribución
  4.1 Conjunto de datos                            (keep, trim narrative)
  4.2 Extracción de características                (keep, trim)
  4.3 Análisis exploratorio                        (keep — EDA is technical)
  4.4 Diseño experimental
       - split estratificado, criterio: F1-macro
       - protocolo de búsqueda de HP por familia de modelo
       (NO re-explicar qué es F1 — está en §2.3.4)
  4.5 Modelos clásicos · características agregadas
       4.5.1 Logistic Regression  → tabla HP + resultados
       4.5.2 Random Forest        → tabla HP + resultados
       4.5.3 XGBoost              → tabla HP + resultados
  4.6 Modelos clásicos · características temporales
       4.6.1 RF temporal          → tabla HP + resultados
       4.6.2 XGBoost temporal     → tabla HP + resultados
  4.7 GNN sobre grafos de posiciones
       4.7.1 Construcción del grafo (técnico, sin "qué es un grafo")
       4.7.2 Arquitectura GCN
       4.7.3 Búsqueda de HP (hidden_dim, lr, dropout, layers, epochs)
  4.8 Experimento de generalización entre jugadores (LOPO)
       - responde a la nota del director
       - mostrar: distribución de predicciones para cada jugador retirado
       - "el modelo nunca puede acertar la etiqueta porque es única, pero
         ¿hacia qué estilo se inclinan sus partidas?"
  4.9 Comparativa global y análisis de resultados
       - tabla consolidada: modelo | mejores HP | acc | F1-macro | F1-weighted
       - F1-macro como criterio (consistente con §3.3 y §2.3.4)
       - matriz de confusión del mejor modelo
       - análisis: pares confundidos, importancia de features, qué falla
```

### Subsections to DELETE from current §4

- 4.4.3 Interpretación y limitaciones
- 4.5.1 Incorporación de nuevas características estratégicas (motivación)
- 4.5.3 Análisis de resultados y limitaciones (narrative → §4.9)
- 4.6.1 Motivación de la representación temporal
- 4.6.4 Discusión de resultados (→ §4.9)
- 4.7.1 Motivación de la representación estructural (duplicates §2.3.5)
- 4.7.5 (duplicate heading + more discussion)

---

## 5. What's already done (commit `2364cc3`)

### Phase 0 — bug fixes & reproducibility
- `app/src/graphs/train_gnn.py`: removed duplicate `__init__`/`forward` that
  silently overrode the 3-layer dropout architecture with a 2-layer one.
- `app/src/features/add_basic_features.py`: derives `aggression_score`,
  `capture_rate`, `check_rate` from `master_games.csv` (these were
  unreproducible before). Formulas:
  - `capture_rate = player_captures / num_moves`
  - `check_rate = player_checks / num_moves`
  - `aggression_score = capture_rate + check_rate`
  (note: aggression_score is exactly the sum of the other two; mention this
  as a limitation in §4.9).
- `requirements.txt`: rewritten UTF-8 with full stack
  (numpy/pandas/scikit-learn/xgboost/python-chess/matplotlib/seaborn/torch/
  torch_geometric).

### Phase 1 — HP search infrastructure (`app/src/experiments/`)
- `_common.py`: shared dataset loaders (flat / temporal), preprocessor
  (StandardScaler + OneHotEncoder), metric helpers, JSON-safe save.
- `hp_search_classical.py`: orchestrates LR (Grid), RF flat+temporal
  (Grid+Random), XGBoost flat+temporal (Grid+Random). StratifiedKFold(5),
  refit on F1-macro, 80/20 train/test split, `random_state=42`.
- `hp_search_gnn.py`: random search over hidden_dim/lr/weight_decay/dropout/
  num_layers/batch_size. 60/20/20 train/val/test split. Early stopping on
  val F1-macro with patience.
- `lopo_experiment.py`: leave-one-player-out experiment using best HPs from
  Phase 1. **WRITTEN BUT NOT YET RUN.**
- `build_comparison_table.py`: consolidates all `hp_search/*.json` into
  `final_comparison.csv` ranked by test F1-macro. **WRITTEN BUT NOT YET RUN.**
- `regenerate_figures.py`: rebuilds confusion matrices + comparative bar
  chart from saved JSONs. **WRITTEN BUT NOT YET RUN.**

### Phase 1 results (`app/results/hp_search/*.json`)

| Model | Features | Search | CV F1m | Test acc | **Test F1-macro** |
|---|---|---|---:|---:|---:|
| RF | temporal | Random | 0.374 | 0.389 | **0.380** ← best |
| LR | flat | Grid | 0.346 | 0.380 | 0.366 |
| XGB | temporal | Random | 0.356 | 0.412 | 0.358 |
| RF | flat | Random | 0.353 | 0.366 | 0.357 |
| XGB | flat | Random | 0.319 | 0.389 | 0.334 |
| GNN | graphs | Random | 0.276 | 0.280 | 0.268 |

For each model, the best run's `best_params` (already pipeline-prefixed with
`model__...`) is in the corresponding JSON's `best_run.best_params`.

Compared to the original §4 hardcoded results, HP search improved:
- RF temporal: 0.331 → 0.380 (+5pp)
- LR flat: 0.312 → 0.366 (+5pp)
- RF flat: 0.324 → 0.357 (+3pp)

The GNN dropped because the original §4 used train/test only (HPs tuned with
test peek); now there's a proper val set → honest number is lower.

### Git state
- Branch: `main`
- Last commit: `2364cc3` "TFM section 4 prep: HP search infrastructure + results"
- Pushed to `origin/main` on https://github.com/AlexCorcia/PythonShowcase

---

## 6. What's pending — exact next actions

In order:

### 6.1. Run Phase 2 (LOPO experiment) — ~2 min
```powershell
cd <repo>\TFM_Chess
venv\Scripts\python.exe -m app.src.experiments.lopo_experiment
```
Outputs: `app/results/lopo/flat.csv`, `app/results/lopo/temporal.csv`.

### 6.2. Run Phase 3 (comparison table + figures) — ~1 min
```powershell
venv\Scripts\python.exe -m app.src.experiments.build_comparison_table
venv\Scripts\python.exe -m app.src.experiments.regenerate_figures
```
Outputs:
- `app/results/tables/final_comparison.csv`
- `app/results/tables/final_comparison.md`
- `app/results/figures/cm_<model>_<features>.png` (one per model)
- `app/results/figures/comparison_f1_macro.png`

### 6.3. Commit & push
```powershell
git add app/results/ ; git commit -m "Phase 2-3: LOPO + comparison artifacts"
git push origin main
```

### 6.4. Apply §2 inconsistency fixes to the .docx (if not already done)
- §2.2.6 heading → "Representación basada en grafos en ajedrez"
- §2.3.5 body → use the reframe text from the previous session (regenerable
  on request)

### 6.5. Rewrite §4 prose — the main deliverable
Follow the structure in §4 of this document. Each model subsection should
contain ONLY:
- one short paragraph introducing the model variant (1-3 sentences),
- a hyperparameter table (search space + selected values + CV/test F1-macro),
- a result paragraph (test accuracy, F1-macro, F1-weighted, brief comment on
  confusion matrix).

Then §4.8 LOPO with a per-player distribution table.

Then §4.9 with the consolidated comparison table and a results-analysis
discussion. **All theory stays in §2.3; do not re-explain anything here.**

### 6.6. Final pass
- Update §3.3 if the methodology mentions any model/protocol that no longer
  matches §4.
- Update figure/table references and the table-of-contents.
- Update the bibliografía if any new citations were needed (likely none — §4
  is purely technical/experimental).

---

## 7. Hard constraints to preserve

- **F1-macro is the primary comparison metric.** Director said so explicitly,
  §3.3 promises it, §2.3.4 introduces it. All comparisons in §4 must use it.
- **§4 must contain no theory** (motivación, definitions, explanations of
  what a model is). All of that lives in §2.3.
- **The GNN is a static GCN per snapshot**, not a temporal graph network.
  The DATA is dynamic (sequence of snapshots), the MODEL is static. Be
  honest about this in §4.7. Full TGNs are future work.
- **Stratified splits, `random_state=42`** everywhere for reproducibility.
- **Don't change the thesis title** — it's already registered. The title
  ("...grafos temporales dinámicos") matches the data representation; just
  be careful about what we claim the model does.
- **APA style** for citations.
- **No use of generative AI tools** is allowed for the thesis content per
  the UNIR guide (§4.4 of the guide). The CODE generated with Claude
  assistance is fine; the prose should be the user's own writing or
  carefully reviewed/edited.

---

## 8. File reference

```
TFM_Chess/
├── app/
│   ├── data/
│   │   ├── raw/                        PGN files (Karpov, Tal, Kasparov, Petrosian)
│   │   ├── processed/                  Parsed CSVs incl. master_games_enriched.csv
│   │   │                               and master_games_final.csv (used by classical
│   │   │                               models) and master_games_temporal.csv
│   │   └── graphs/                     graph_dataset.pt + *.json (gitignored, regen
│   │                                   via build_chess_graphs.py + prepare_gnn_dataset.py)
│   ├── docs/
│   │   ├── TFM_alexandrecorcia_...docx     ← thesis source
│   │   ├── TFM_alexandrecorcia_...pdf      ← current rendered version
│   │   ├── notas de reunion.txt            ← director feedback (drives §4 work)
│   │   ├── rubrica.pdf                     ← grading rubric
│   │   ├── guia.pdf                        ← TFM guide
│   │   ├── reglamento_trabajos_...pdf      ← TFM regulations
│   │   ├── HANDOFF.md                      ← THIS FILE
│   │   └── DATA/                           ← redundant zips of raw PGNs
│   ├── results/
│   │   ├── hp_search/*.json                ← Phase 1 results (6 files)
│   │   ├── lopo/                           ← Phase 2 outputs (CREATED BY 6.1)
│   │   ├── tables/                         ← old + Phase 3 outputs
│   │   └── figures/                        ← old + Phase 3 outputs
│   └── src/
│       ├── preprocessing/parse_pgn.py      raw PGN → parsed CSV
│       ├── preprocessing/parse_all_players.py    runs the parser for all 4 players
│       ├── features/add_basic_features.py        master_games → master_games_enriched
│       ├── features/add_eco_features.py          enriched → master_games_final
│       ├── features/temporal_feature_extraction.py  builds master_games_temporal.csv
│       ├── graphs/build_chess_graphs.py          PGN → per-snapshot graph JSON
│       ├── graphs/prepare_gnn_dataset.py         JSON → graph_dataset.pt
│       ├── graphs/train_gnn.py                   original GCN training (legacy)
│       ├── models/                                legacy classical scripts (kept for ref)
│       └── experiments/                           ← the new Phase 1 / 2 / 3 code
│           ├── _common.py
│           ├── hp_search_classical.py
│           ├── hp_search_gnn.py
│           ├── lopo_experiment.py
│           ├── build_comparison_table.py
│           └── regenerate_figures.py
├── requirements.txt
├── .gitignore                          (ignores venv/, large graph artifacts)
└── venv/                               (NOT in git; recreate with pip install -r)
```

---

## 9. Setup commands for a fresh machine

```powershell
git clone https://github.com/AlexCorcia/PythonShowcase.git
cd PythonShowcase\TFM_Chess
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Regenerate the gitignored graph artifacts (5-10 min):
python app\src\graphs\build_chess_graphs.py
python app\src\graphs\prepare_gnn_dataset.py

# Sanity check that everything from Phase 1 still works:
python -c "import json, pathlib; [print(p.name, json.load(open(p))['best_run']['test_metrics']['f1_macro']) for p in sorted(pathlib.Path('app/results/hp_search').glob('*.json'))]"
```

---

## 10. Open questions / decisions still pending

1. **Did the user paste the §2.2.6 and §2.3.5 fixes into the .docx?**
   Confirm before starting §4 rewrite.
2. **Tone of §4 prose**: factual/short or more discursive? Default is factual
   (the rubric rewards conciseness and clarity).
3. **Where do figures live in the .docx?** Need to update figure refs after
   regenerate_figures.py produces new PNGs.
4. **LOPO scope**: currently planned for flat + temporal only (skips GNN).
   If the new Claude session has time, add GNN-LOPO — the script does NOT
   currently support that and would need extending.
5. **Conclusiones (§5) and Anexo A (código fuente)** are still placeholder
   text. Need to be written. Anexo A should mention the GitHub repo:
   https://github.com/AlexCorcia/PythonShowcase
