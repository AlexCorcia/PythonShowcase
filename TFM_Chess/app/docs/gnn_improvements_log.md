# Registro de mejoras de la GNN (para documentar en §4.7 / §5)

Este documento registra todos los cambios introducidos para mejorar el modelo
basado en grafos, junto con su justificación y su efecto medido. Sirve de base
para redactar la sección técnica y la discusión.

## Punto de partida (commit `1878467`)

GNN = clasificador **estático por snapshot** (GCN de 3 capas + global mean pool).

| Métrica (test) | Valor |
|---|---:|
| F1-macro | 0,268 |
| Exactitud | 0,280 |

Es el peor de los seis modelos; queda prácticamente en el azar (4 clases → 0,25).

### Diagnóstico de las causas (no eran los hiperparámetros)

1. **Datos insuficientes.** `build_chess_graphs.py` usaba
   `max_games_per_player=50` → solo **200 partidas** frente a las **9.936** de
   los modelos clásicos (~50× menos datos).
2. **Solo la apertura.** `max_snapshots=20` → solo los primeros 20 plies
   (10 jugadas) de cada partida. La apertura es la fase **menos**
   discriminativa del estilo (teoría compartida); se descartaban medio juego y
   final, justamente donde el estilo se manifiesta.
3. **Características de nodo pobres y mal codificadas.** Vector de 5 dimensiones
   `[piece_id, color_id, x/7, y/7, value/9]`, con `piece_id` ordinal (peón=1 …
   rey=6): la GCN interpretaba "rey" como 6× "peón".
4. **Fuga de información.** La partición train/val/test se hacía por *snapshot*,
   no por partida: snapshots de la misma partida caían en conjuntos distintos
   → la cifra 0,268 era incluso optimista.
5. **Sin pesos de clase.** `cross_entropy` sin balanceo → la clase `dynamic`
   quedaba prácticamente ignorada (columna casi vacía en la matriz de confusión).

---

## Cambios introducidos

### Tier 1 — datos y características
- [x] Usar **todas** las partidas disponibles (9.977, no 50 por jugador).
- [x] Muestrear snapshots a lo largo de **toda** la partida (apertura, medio
      juego y final), no solo los primeros 20 plies. (10 snapshots/partida
      equiespaciados → 99.761 snapshots.)
- [x] Codificación **one-hot** del tipo de pieza (6) y del color (2); añadir
      características de nodo más ricas (grado de ataque/defensa, valor,
      posición) → vector de nodo de 13 dimensiones (antes 5).
- [x] **Pesos de clase** (inverso de frecuencia) en la función de pérdida.
- [x] BatchNorm en las capas GCN.

### Tier 2 — unidad de evaluación
- [x] Partición **por partida** (sin fuga entre snapshots de la misma partida).
- [x] **Agregación a nivel de partida**: la predicción de estilo de una partida
      es la **media de las probabilidades softmax** de sus snapshots → misma
      unidad que los modelos clásicos.

### Tier 3 — modelo temporal
- [x] Modelo **temporal** (`train_temporal_gnn.py`): codificador GCN por
      snapshot + **GRU** sobre la secuencia ordenada de snapshots de la partida
      → predicción directa a nivel de partida. Coherente con el título
      "grafos temporales dinámicos".

### Infraestructura
- [x] Entrenamiento en **GPU** (RTX 5070, CUDA 12.8). Requiere instalar la
      rueda CUDA de PyTorch:
      `pip install torch --index-url https://download.pytorch.org/whl/cu128`
      (la `requirements.txt` por defecto instala la versión CPU).
- [x] `forward` del modelo temporal **vectorizado** (sin bucle Python con
      `.item()` por snapshot, que forzaba una sincronización GPU↔CPU por
      elemento): de >20 min/trial a ~2-3 min/trial.

---

## Resultados tras las mejoras

Protocolo: partición por partida 60/20/20 estratificada (`random_state=42`),
selección por F1-macro de validación a nivel de partida. Test = 1.996 partidas.

| Variante GNN | Unidad de evaluación | F1-macro | Exactitud | Notas |
|---|---|---:|---:|---|
| Estática por snapshot (**base**) | snapshot | 0,268 | 0,280 | punto de partida (peor modelo) |
| **Estática + agregación a partida** | partida (media de probs) | **0,393** | 0,424 | **mejor modelo global** |
| **Temporal (GCN + GRU)** | partida (secuencia) | 0,380 | 0,394 | coherente con el título |

Mejora neta: **+0,125 F1-macro** (0,268 → 0,393).

### Tabla comparativa global actualizada (ordenada por F1-macro de test)

| Modelo | Características | F1-macro | Exactitud |
|---|---|---:|---:|
| **GNN (estática + agregación)** | grafos | **0,393** | 0,424 |
| Random Forest | temporal | 0,380 | 0,389 |
| GNN temporal (GCN+GRU) | grafos | 0,380 | 0,394 |
| Regresión logística | flat | 0,366 | 0,380 |
| XGBoost | temporal | 0,358 | 0,412 |
| Random Forest | flat | 0,357 | 0,366 |
| XGBoost | flat | 0,334 | 0,389 |

Los dos modelos de grafos pasan de ser el **peor** (0,268) a ocupar los puestos
**1º y 3º**. La representación en grafo deja de ser una limitación y se convierte
en el punto fuerte del trabajo.

### Análisis

- **El factor decisivo no fueron los hiperparámetros.** La búsqueda de HP sobre
  la versión base ya estaba hecha; la mejora vino de los datos y la
  representación (todas las partidas, todas las fases, características de nodo
  correctas) y de la **unidad de evaluación** (partida, no snapshot).
- **La agregación a nivel de partida es clave.** El F1-macro a nivel de snapshot
  se queda en ~0,33 (un solo tablero contiene poca información de estilo); al
  promediar las probabilidades de los ~10 snapshots de una partida sube a 0,393.
  Confirma la hipótesis del diagnóstico: el estilo es una propiedad de la
  *partida*, no de la *posición*.
- **El modelo temporal (GRU) no supera a la agregación simple** (0,380 vs
  0,393) en este conjunto, pero es el que mejor encaja con el marco "temporal
  dinámico" del título; su ligera desventaja es honesta y discutible como
  resultado.
- **Clase más difícil:** `dynamic` (F1 ≈ 0,23-0,29) sigue siendo la peor en
  todos los modelos; se confunde con `tactical` y `positional` (Kaspárov es el
  jugador "intermedio" del conjunto).

### Comparación en igualdad de condiciones (mismo split, mismo test)

Para una comparación pareada estricta (`unified_comparison.py`), todos los
modelos se entrenan con las mismas partidas (60%) y se evalúan sobre el **mismo
conjunto de test (1.965 partidas)**, emparejadas por clave común. Las GNN se
promedian sobre **5 semillas** (su inicialización es aleatoria); los clásicos son
deterministas. Resultado:

| Modelo | Características | Test F1-macro (media ± std) |
|---|---|---:|
| **GNN (estática + agregación)** | grafos | **0,371 ± 0,008** |
| XGBoost | temporal | 0,369 |
| **GNN temporal (GCN+GRU)** | grafos | **0,369 ± 0,007** |
| Random Forest | temporal | 0,365 |
| Regresión logística | flat | 0,362 |
| Random Forest | flat | 0,353 |
| XGBoost | flat | 0,333 |

**Afirmación defendible:** sobre exactamente los mismos datos, la representación
en grafo es **competitiva con el mejor modelo clásico**. El mejor modelo de
grafos (0,371) obtiene el F1-macro medio más alto, en **empate estadístico** con
XGBoost temporal (0,369; la diferencia es muy inferior a la std). NO es correcto
afirmar que los grafos "superan a todos los clásicos": esa lectura provenía de
una única ejecución con semilla favorable. La conclusión robusta es que los
grafos pasan del peor puesto (0,268) al grupo de cabeza, a la altura del mejor
modelo clásico, y lo hacen sin características diseñadas a mano.

⚠️ **Nota de reproducibilidad:** el entrenamiento de las GNN no es totalmente
determinista (init aleatoria + operaciones CUDA). Por eso se promedia sobre
varias semillas. Una única ejecución puede variar ±0,01-0,02 en F1-macro.

### Pendiente de redactar en la memoria

- Reescribir §4.7 con esta arquitectura y estos resultados (la GNN ya no es el
  peor modelo); actualizar §4.9 (tabla consolidada y mejor modelo = GNN).
- Mencionar la GPU/CUDA en el Anexo de reproducibilidad.
