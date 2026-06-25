# Handoff — TFM "Aprendizaje de estilos de juego en ajedrez"

Documento de traspaso para continuar el TFM en una **nueva sesión** (incluida una
sesión de Claude **sin acceso al sistema de archivos**, p. ej. claude.ai web).
Por eso los resultados clave se incluyen **en línea**, no solo como referencias a
ficheros. Léelo entero antes de empezar.

Última actualización: tras la revisión y mejora completa del modelo de grafos
(GNN) y la comparación en igualdad de condiciones. Estos cambios **aún no están
commiteados** en el momento de escribir (ver §11).

---

## 1. Qué es este proyecto

- **Título:** *Aprendizaje de estilos de juego en ajedrez mediante modelos de
  Machine Learning y grafos temporales dinámicos.*
- **Autor:** Alexandre Marc Corcia Aguilera (UNIR — Máster en IA).
- **Tipo:** *Comparativa de soluciones* (Tipo 3).
- **Directora:** Xiomara Patricia Blanco Valencia.
- **Objetivo:** comparar tres familias de representación para clasificar el
  estilo de juego a partir de partidas PGN:
  1. **Características agregadas (flat)** — conteos, ratios, familia ECO.
  2. **Características temporales por fase** — apertura/medio/final.
  3. **Representación en grafo** con una GNN sobre snapshots de posición.
- **Jugadores/estilos:** Karpov (posicional), Tal (táctico), Kaspárov (dinámico),
  Petrosián (defensivo). Un estilo por jugador → la etiqueta de estilo coincide
  con la identidad del jugador (relevante para el experimento LOPO, §4.8).
- **La memoria está en `app/docs/`** (.docx y .pdf).

---

## 2. Feedback de la directora (de `app/docs/notas de reunion.txt`)

- **Optimización de hiperparámetros obligatoria**, con método lógico por técnica:
  LR → GridSearch; RF y XGBoost → Grid + Randomized; GNN → búsqueda real.
- **§4 puramente técnica, sin teoría** (la teoría va en §2.3 Marco Teórico).
- **Una tabla de HP por técnica** en §4.
- **Tabla comparativa consolidada** al final de §4, con **F1-macro como métrica
  principal** ("F1 es la mejor").
- **§4 termina con un Análisis de Resultados.**
- **Experimento nuevo:** detectar el estilo de jugadores no entrenados → LOPO.
- **Estilo APA.**

---

## 3. Estado de §2 y §3

- §2.3 Marco Teórico cubre aprendizaje supervisado, modelos clásicos, optimización
  de HP, métricas y grafos en ajedrez.
- Las dos correcciones de §2 que estaban pendientes (§2.2.6 título →
  "Representación basada en grafos en ajedrez"; §2.3.5 reframe dato dinámico vs
  modelo estático) **el usuario confirmó que ya están aplicadas** en el .docx.
- §3.2/§3.3 prometen optimización de HP y evaluación por F1-macro; §4 lo cumple.

---

## 4. RESULTADOS ACTUALES (lo más importante)

Todos los modelos predicen **una etiqueta de estilo por partida** y se comparan
por **F1-macro**. Azar con 4 clases ≈ 0,25.

### 4.1 Resultados de la búsqueda de hiperparámetros (cada modelo en su propio split)

| Modelo | Características | Test F1-macro | Test acc |
|---|---|---:|---:|
| GNN (estática + agregación) | grafos | 0,393 | 0,424 |
| Random Forest | temporal | 0,380 | 0,389 |
| GNN temporal (GCN+GRU) | grafos | 0,380 | 0,394 |
| Regresión logística | flat | 0,366 | 0,380 |
| XGBoost | temporal | 0,358 | 0,412 |
| Random Forest | flat | 0,357 | 0,366 |
| XGBoost | flat | 0,334 | 0,389 |

(Estos números provienen de `app/results/hp_search/*.json`. Los clásicos usan
split 80/20 + CV(5); las GNN usan 60/20/20 por partida.)

### 4.2 Comparación EN IGUALDAD DE CONDICIONES (mismo train, mismo test, 1.965 partidas)

Para una comparación justa, `app/src/experiments/unified_comparison.py` entrena
todos los modelos con las MISMAS partidas y los evalúa sobre el MISMO test,
emparejadas por clave común. Las GNN se promedian sobre **5 semillas**
(inicialización aleatoria); los clásicos son deterministas.

| Modelo | Características | Test F1-macro (media ± std) |
|---|---|---:|
| **GNN (estática + agregación)** | grafos | **0,371 ± 0,008** |
| XGBoost | temporal | 0,369 |
| **GNN temporal (GCN+GRU)** | grafos | **0,369 ± 0,007** |
| Random Forest | temporal | 0,365 |
| Regresión logística | flat | 0,362 |
| Random Forest | flat | 0,353 |
| XGBoost | flat | 0,333 |

**Conclusión defendible (usar esta en la memoria):** la representación en grafo
es **competitiva con el mejor modelo clásico**. El mejor modelo de grafos (0,371)
tiene el F1-macro medio más alto, en **empate estadístico** con XGBoost temporal
(0,369; diferencia << std). **NO afirmar** que los grafos "superan a todos los
clásicos" (eso era una única ejecución con semilla favorable). La historia real y
sólida: la GNN pasa del **peor** modelo (0,268, versión original) al **grupo de
cabeza**, a la altura del mejor clásico, **sin características diseñadas a mano**.

---

## 5. Qué se hizo con la GNN (overhaul completo) — para redactar §4.7 y §5

La GNN original era el **peor** modelo (F1-macro 0,268, casi azar). El diagnóstico
fue que el problema NO eran los hiperparámetros, sino los datos y la
representación. Detalle completo en `app/docs/gnn_improvements_log.md`. Resumen:

**Tier 1 — datos y características**
- Usar **todas las partidas** (9.977) en vez de 50 por jugador.
- Muestrear snapshots a lo largo de **toda la partida** (apertura/medio/final),
  no solo los primeros 20 plies (apertura, poco discriminativa). 10 snapshots/
  partida → 99.761 snapshots.
- Características de nodo **one-hot** (tipo de pieza, color) + grados de
  ataque/defensa, valor y posición → 13 dimensiones (antes 5, con `piece_id`
  ordinal mal codificado).
- **Pesos de clase** en la pérdida + BatchNorm.

**Tier 2 — unidad de evaluación**
- Partición **por partida** (no por snapshot) → sin fuga de información.
- **Agregación a nivel de partida**: la predicción de la partida es la media de
  las probabilidades de sus snapshots. (El F1-macro por snapshot se queda en
  ~0,33; agregar a partida sube a ~0,37-0,39. El estilo es propiedad de la
  partida, no de una posición.)

**Tier 3 — modelo temporal** (`app/src/experiments/train_temporal_gnn.py`)
- Codificador GCN por snapshot + **GRU** sobre la secuencia ordenada de snapshots
  → predicción de partida. Es la arquitectura coherente con el título "grafos
  temporales dinámicos". Rinde igual que la estática+agregación (empate).

**Infraestructura**
- Entrenamiento en **GPU** (RTX 5070, CUDA 12.8). El `forward` del modelo
  temporal se **vectorizó** (un bucle Python con `.item()` por snapshot forzaba
  sincronización GPU↔CPU; pasó de >20 min/trial a ~2-3 min/trial).

⚠️ **Reproducibilidad:** el entrenamiento de las GNN no es 100% determinista
(init aleatoria + CUDA). Por eso §4.2 promedia 5 semillas. Una sola ejecución
puede variar ±0,01-0,02. Los modelos clásicos sí son deterministas.

---

## 6. Estado del borrador de §4

`app/docs/seccion4_borrador.md` contiene un **borrador completo de §4 en español**
(v2), técnico y conciso, con la métrica F1-macro, la tabla en igualdad de
condiciones y la nueva sección de grafos. **Pendiente:** el usuario debe
revisarlo, reescribirlo con sus palabras (regla UNIR de no usar IA generativa
para la prosa de la memoria) e incorporarlo al .docx. Estructura:

- 4.1 Datos · 4.2 Características · 4.3 EDA · 4.4 Diseño experimental (incl.
  §4.4.4 comparación en igualdad de condiciones).
- 4.5 Clásicos agregados (LR, RF, XGB) · 4.6 Clásicos temporales (RF, XGB).
- 4.7 Grafos (construcción, GCN estática+agregación, GCN+GRU temporal, búsqueda HP).
- 4.8 LOPO · 4.9 Comparativa global + análisis.

**Aviso de consistencia:** §4.5-4.7 reportan los números de la búsqueda de HP
(p. ej. GNN 0,393), mientras que §4.9 reporta los de igualdad de condiciones
(GNN 0,371). Están etiquetados como tales, pero conviene mantener esa distinción
clara al redactar.

---

## 7. Qué queda pendiente

1. **Revisar e incorporar §4** al .docx desde `seccion4_borrador.md` (prosa
   propia del usuario).
2. **§4.8 LOPO temporal** solo cubre RF y XGB (no hay HP para
   `logistic_regression_temporal`). Opcional: lanzar esa búsqueda para completar.
3. **§5 Conclusiones** y **Anexo A (código fuente)** siguen como placeholder.
   Anexo A debe citar el repo: https://github.com/AlexCorcia/PythonShowcase
4. **Pasada final:** sincronizar §3.3 con §4, actualizar referencias de
   figuras/tablas e índice, revisar bibliografía (APA).
5. **Commit y push** de todo el trabajo (ver §11).

---

## 8. Ficheros relevantes (estructura)

```
TFM_Chess/
├── app/
│   ├── data/
│   │   ├── raw/                         PGNs (Karpov, Tal, Kasparov, Petrosian)
│   │   ├── processed/                   master_games_final.csv (flat),
│   │   │                                master_games_temporal.csv (temporal)
│   │   └── graphs/                      graph_dataset.pt + *.json (gitignored;
│   │                                    regenerar, ver §9)
│   ├── docs/
│   │   ├── TFM_...docx / .pdf           ← memoria
│   │   ├── notas de reunion.txt         ← feedback directora
│   │   ├── HANDOFF.md                   ← ESTE FICHERO
│   │   ├── seccion4_borrador.md         ← borrador de §4 (v2)
│   │   └── gnn_improvements_log.md      ← registro detallado de las mejoras GNN
│   ├── results/
│   │   ├── hp_search/*.json             resultados por modelo (incl. gnn_graphs,
│   │   │                                gnn_temporal_graphs)
│   │   ├── lopo/{flat,temporal}.csv     experimento LOPO
│   │   ├── tables/final_comparison.*    tabla por-modelo
│   │   ├── tables/unified_comparison.*  tabla en igualdad de condiciones
│   │   └── figures/                     cm_*.png (búsqueda),
│   │                                    cm_unified_*.png (igualdad de cond.),
│   │                                    comparison_f1_macro.png,
│   │                                    unified_comparison_f1_macro.png
│   └── src/
│       ├── preprocessing/parse_pgn.py, parse_all_players.py
│       ├── features/add_basic_features.py, add_eco_features.py,
│       │            temporal_feature_extraction.py
│       ├── graphs/build_chess_graphs.py      PGN → snapshots (TODAS las partidas,
│       │                                     toda la partida; CLI --max-games-per-player
│       │                                     0 = todas, --snapshots-per-game N)
│       ├── graphs/prepare_gnn_dataset.py     JSON → graph_dataset.pt (nodos 13-dim)
│       └── experiments/
│           ├── _common.py                    loaders, preprocesador, métricas
│           ├── hp_search_classical.py        LR/RF/XGB (flat y temporal)
│           ├── hp_search_gnn.py              GNN estática + agregación a partida
│           ├── train_temporal_gnn.py         GNN temporal (GCN + GRU)
│           ├── lopo_experiment.py            experimento LOPO
│           ├── build_comparison_table.py     tabla por-modelo
│           ├── regenerate_figures.py         matrices de confusión + barras
│           └── unified_comparison.py         comparación en igualdad de condiciones
│                                             (mismo split + mismo test, 5 semillas)
├── requirements.txt
└── venv/                                (no en git)
```

---

## 9. Puesta en marcha (máquina nueva)

```powershell
git clone https://github.com/AlexCorcia/PythonShowcase.git
cd PythonShowcase\TFM_Chess
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# GPU (NVIDIA): la requirements instala torch CPU. Para usar la GPU, reinstalar
# la rueda CUDA (ajustar cuXXX al CUDA del equipo; cu128 para RTX serie 50):
pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu128

# Regenerar artefactos de grafos (gitignored):
python -m app.src.graphs.build_chess_graphs --max-games-per-player 0 --snapshots-per-game 10
python -m app.src.graphs.prepare_gnn_dataset
```

Reproducir experimentos:
```powershell
python -m app.src.experiments.hp_search_classical          # LR/RF/XGB
python -m app.src.experiments.hp_search_gnn                 # GNN estática+agregación
python -m app.src.experiments.train_temporal_gnn           # GNN temporal
python -m app.src.experiments.lopo_experiment              # LOPO
python -m app.src.experiments.build_comparison_table       # tabla por-modelo
python -m app.src.experiments.regenerate_figures           # figuras
python -m app.src.experiments.unified_comparison           # igualdad de condiciones
```

---

## 10. Restricciones a respetar

- **F1-macro es la métrica principal** de comparación (lo pidió la directora).
- **§4 sin teoría** (definiciones, motivaciones → §2.3).
- **La GNN: el dato es dinámico (secuencia de snapshots), el modelo puede ser
  estático (GCN por snapshot + agregación) o temporal (GCN+GRU).** Ser honesto
  con qué hace cada variante.
- **No sobre-vender los grafos:** en igualdad de condiciones **empatan** con el
  mejor clásico, no lo superan claramente. Esa es la afirmación defendible.
- **Splits estratificados, `random_state=42`.** Las GNN, además, se promedian
  sobre varias semillas por su no-determinismo.
- **No cambiar el título** (ya registrado).
- **APA** en citas.
- **No usar IA generativa para la prosa de la memoria** (regla UNIR). El código
  asistido con Claude es admisible; la redacción debe ser propia/revisada.

---

## 11. Estado de git

- Rama: `main`. Repo: https://github.com/AlexCorcia/PythonShowcase
- Commit del overhaul de la GNN: `fc0c5cf` "GNN overhaul: graphs from worst to
  top-tier" (incluye `build_chess_graphs.py`, `prepare_gnn_dataset.py`,
  `hp_search_gnn.py`, `train_temporal_gnn.py`, `unified_comparison.py`, nuevos
  resultados/figuras y los docs `seccion4_borrador.md`, `gnn_improvements_log.md`
  y este HANDOFF). Commit anterior: `1878467` "Phase 2-3: LOPO + comparison".
- **Estado de push:** confirmar con `git log origin/main..main`. Si `fc0c5cf` no
  está en `origin/main`, hacer `git push origin main` para que otra sesión pueda
  clonarlo.
- Los artefactos de grafos (`app/data/graphs/`) están gitignored: se regeneran
  (ver §9).

---

## 12. Preguntas abiertas / decisiones

1. **Tono de §4:** factual/conciso (elegido). 
2. **§4.8 LOPO:** ¿añadir LOPO para la GNN y/o LR temporal? Actualmente no.
3. **Figuras en el .docx:** actualizar referencias tras las nuevas
   (`unified_comparison_f1_macro.png`, `cm_unified_*.png`).
4. **§5 y Anexo A:** redactar (placeholder).
