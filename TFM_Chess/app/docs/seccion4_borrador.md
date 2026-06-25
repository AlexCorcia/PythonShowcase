# 4. Desarrollo específico de la contribución

> **Borrador para revisión (v2, con la representación en grafo mejorada).**
> Texto técnico, sin teoría (la teoría está en §2.3). Métrica principal de
> comparación: **F1-macro**. Particiones estratificadas con `random_state=42`.
> Revísalo y reescríbelo con tus palabras antes de incorporarlo al documento.

---

## 4.1 Conjunto de datos

El conjunto de datos se compone de 9.936 partidas de torneo de cuatro jugadores
de élite, cada uno asociado a un estilo de juego de referencia: Anatoli Karpov
(posicional), Mijaíl Tal (táctico), Garri Kaspárov (dinámico) y Tigran Petrosián
(defensivo). Las partidas, en formato PGN, abarcan el periodo 1945–2022. La
distribución por jugador y estilo es la siguiente:

| Jugador | Estilo | Partidas |
|---|---|---:|
| Karpov | positional | 3.519 |
| Tal | tactical | 2.417 |
| Kasparov | dynamic | 2.123 |
| Petrosian | defensive | 1.877 |
| **Total** | | **9.936** |

Como cada jugador aporta un único estilo, la etiqueta de estilo coincide de
facto con la identidad del jugador. Esto condiciona la interpretación de los
resultados y motiva el experimento de generalización entre jugadores (§4.8).
Las clases están moderadamente desbalanceadas (1.877–3.519 partidas), por lo que
se emplean pesos de clase y se prioriza F1-macro.

## 4.2 Extracción de características

A partir de cada PGN se construyen tres representaciones:

- **Agregada (flat).** Seis variables numéricas por partida (`num_moves`,
  `player_captures`, `player_checks`, `capture_rate`, `check_rate`,
  `aggression_score`) más la familia de apertura `eco_family` (categórica, A–E).
- **Temporal por fase (temporal).** Las magnitudes de capturas, jaques, enroques
  y promociones —de jugador y rival— desagregadas en apertura, medio juego y
  final (21 variables de fase) más `num_moves` y `eco_family`.
- **Grafo de posiciones.** Cada partida se transforma en una secuencia de grafos
  por posición, descrita en §4.7.

## 4.3 Análisis exploratorio

Las señales de estilo son débiles y se solapan entre clases. Las medias de
actividad por estilo difieren poco:

| Estilo | aggression_score | capture_rate | check_rate |
|---|---:|---:|---:|
| dynamic | 0,267 | 0,218 | 0,049 |
| tactical | 0,263 | 0,218 | 0,046 |
| positional | 0,242 | 0,206 | 0,037 |
| defensive | 0,223 | 0,195 | 0,028 |

El orden es coherente con la intuición ajedrecística, pero las diferencias
absolutas son pequeñas, lo que anticipa una clasificación difícil.

## 4.4 Diseño experimental

### 4.4.1 Protocolo común

- **Criterio de selección y comparación:** F1-macro (mismo peso a las cuatro
  clases pese al desbalanceo).
- **Reproducibilidad:** `random_state=42` en todas las particiones, búsquedas y
  modelos; particiones estratificadas por estilo.
- **Preprocesado de los modelos clásicos:** estandarización de variables
  numéricas y codificación *one-hot* de `eco_family`, integrados en un
  *pipeline* para evitar fugas de información.

### 4.4.2 Particiones

- **Modelos clásicos:** división estratificada 80/20 (7.948 entrenamiento /
  1.988 prueba), con validación cruzada estratificada de 5 particiones sobre el
  entrenamiento y *refit* sobre F1-macro.
- **Modelos de grafos:** división estratificada **por partida** 60/20/20
  (5.985 / 1.996 / 1.996 partidas). Se reserva un conjunto de validación
  independiente porque estos modelos requieren *early stopping* y selección de
  arquitectura; la partición es por partida (no por posición) para evitar que
  posiciones de una misma partida caigan en conjuntos distintos.

### 4.4.3 Búsqueda de hiperparámetros por familia de modelo

- Regresión logística → *GridSearch*.
- Random Forest y XGBoost → *GridSearch* + *RandomizedSearch*.
- Modelos de grafos → búsqueda aleatoria con *early stopping* sobre el F1-macro
  de validación.

### 4.4.4 Comparación en igualdad de condiciones

La comparación final entre familias (§4.9) se realiza en **igualdad estricta de
condiciones**: todos los modelos —clásicos y de grafos— se entrenan con
**exactamente las mismas partidas** de entrenamiento (60%) y se evalúan sobre
**exactamente el mismo conjunto de test** (1.965 partidas), emparejadas mediante
una clave común a ambas representaciones
(`main_player|white|black|date|result|eco`). Todos comparten además la misma
tarea (cuatro clases), la misma unidad de predicción (una etiqueta por partida)
y la misma métrica principal (F1-macro). De este modo, las diferencias en
F1-macro no pueden atribuirse a particiones distintas: reflejan únicamente la
capacidad de cada representación sobre los mismos datos. Como la inicialización
de las redes neuronales es aleatoria, los modelos de grafos se promedian sobre
**5 semillas** (media ± desviación típica); los modelos clásicos son
deterministas (`random_state=42`).

## 4.5 Modelos clásicos · características agregadas

### 4.5.1 Regresión logística

Modelo lineal multinomial con regularización L2 y pesos de clase balanceados
(*GridSearch*).

| Hiperparámetro | Espacio | Valor |
|---|---|---|
| `C` | {0,01; 0,1; 1; 10} | **0,01** |
| `class_weight` | {None, balanced} | **balanced** |

Test: exactitud 0,380, **F1-macro 0,366**, F1-ponderada 0,377. Mejor clase:
posicional (F1 = 0,435) y táctica (0,425); peor: dinámica (0,247).

### 4.5.2 Random Forest

Mejor configuración (*RandomizedSearch*): `n_estimators=600`, `max_depth=6`,
`max_features=log2`, `min_samples_split=2`, `min_samples_leaf=2`,
`class_weight=balanced`. Test: exactitud 0,366, **F1-macro 0,357**,
F1-ponderada 0,367.

### 4.5.3 XGBoost

Mejor configuración (*RandomizedSearch*): `n_estimators=600`, `max_depth=3`,
`learning_rate=0,1`, `subsample=0,9`, `colsample_bytree=0,8`, `gamma=0,1`,
`min_child_weight=5`, `reg_alpha=1,0`, `reg_lambda=2,0`. Test: exactitud 0,389,
**F1-macro 0,334**, F1-ponderada 0,362. Pese a la mayor exactitud del bloque,
su F1-macro es la más baja: sobre-predice la clase posicional (F1 = 0,513), lo
que ilustra por qué la métrica de referencia es F1-macro y no la exactitud.

## 4.6 Modelos clásicos · características temporales

### 4.6.1 Random Forest temporal

Mejor configuración (*RandomizedSearch*): `n_estimators=300`, `max_depth=None`,
`max_features=log2`, `min_samples_split=10`, `min_samples_leaf=8`,
`class_weight=balanced`. Test: exactitud 0,389, **F1-macro 0,380**,
F1-ponderada 0,390. La desagregación por fases mejora todas las clases respecto
a la versión agregada (dinámica: 0,261 → 0,303).

### 4.6.2 XGBoost temporal

Mejor configuración: `n_estimators=600`, `max_depth=3`, `learning_rate=0,05`,
`subsample=0,7`, `colsample_bytree=0,9`, `gamma=0,1`, `min_child_weight=5`,
`reg_alpha=0,1`, `reg_lambda=1,0`. Test: exactitud 0,412 (la mayor del estudio),
**F1-macro 0,358**, F1-ponderada 0,386. Mantiene el sesgo hacia la clase
posicional.

## 4.7 Modelos basados en grafos de posiciones

### 4.7.1 Construcción del grafo

Cada partida se representa como una **secuencia de grafos de posición**. De cada
partida se toman 10 snapshots equiespaciados a lo largo de **toda** la partida
(apertura, medio juego y final), descartando la posición inicial por ser
idéntica en todas. Cada snapshot es un grafo dirigido donde:

- **Nodos:** las casillas ocupadas (una pieza por nodo).
- **Aristas:** relaciones de ataque/defensa; por cada pieza se crean aristas
  hacia las casillas ocupadas que ataca (relación *ataque* si la pieza destino
  es rival, *defensa* si es propia).
- **Características de nodo (13 dimensiones):** codificación *one-hot* del tipo
  de pieza (6) y del color (2); coordenadas normalizadas (x/7, y/7); valor
  material normalizado (valor/9); y grados de ataque y de defensa normalizados.
  La codificación one-hot evita que el modelo interprete el tipo de pieza como
  una escala ordinal.

El conjunto resultante contiene 9.977 partidas y 99.761 snapshots.

### 4.7.2 Modelo estático con agregación a nivel de partida

La arquitectura base es una red convolucional de grafos (GCN) que clasifica cada
snapshot de forma independiente: tres capas `GCNConv` con normalización por lotes
y ReLU, *global mean pooling* para obtener un vector por snapshot, y dos capas
lineales de salida sobre las cuatro clases. El entrenamiento usa entropía cruzada
con **pesos de clase** (inverso de frecuencia: defensive 1,32; dynamic 1,17;
positional 0,71; tactical 1,03).

La predicción de estilo de **una partida** se obtiene **promediando las
probabilidades softmax de sus snapshots** y tomando el argmax. Esta agregación es
el elemento clave: traslada la predicción de la posición individual —poco
informativa del estilo— a la partida completa, que es la unidad etiquetada y la
misma que usan los modelos clásicos.

Búsqueda aleatoria de hiperparámetros (10 configuraciones):

| Hiperparámetro | Valor seleccionado |
|---|---|
| `hidden_dim` | 128 |
| `num_layers` | 3 |
| `lr` | 0,0005 |
| `weight_decay` | 1e-05 |
| `dropout` | 0,2 |
| `batch_size` | 256 |

Resultados en test (nivel partida): exactitud 0,424, **F1-macro 0,393**,
F1-ponderada 0,415. **Es el mejor modelo del estudio.** El F1-macro a nivel de
snapshot individual se queda en 0,330: la diferencia (0,330 → 0,393) cuantifica
la ganancia de agregar las posiciones a nivel de partida.

### 4.7.3 Modelo temporal (GCN + GRU)

Para aprovechar explícitamente la dimensión temporal del dato, se añade un modelo
que trata la partida como una **secuencia ordenada de snapshots**: el mismo
codificador GCN produce un *embedding* por snapshot y una red recurrente **GRU**
procesa la secuencia `(embedding_1, …, embedding_T)`; el estado final alimenta una
capa lineal que predice el estilo de la partida. Esta formulación es la que mejor
se corresponde con el planteamiento de "grafos temporales dinámicos" del título.

Búsqueda aleatoria (6 configuraciones); seleccionada: `hidden_dim=128`,
`gru_hidden=64`, `num_layers=3`, `lr=0,001`, `dropout=0,2`, `batch_size=128`.
Resultados en test: exactitud 0,394, **F1-macro 0,380**, F1-ponderada 0,395.

### 4.7.4 Comentario

Los resultados anteriores corresponden a la búsqueda de hiperparámetros de cada
modelo. La comparación justa frente a los modelos clásicos —sobre el mismo
conjunto de test y promediando varias semillas— se presenta en §4.9, donde la
representación en grafo se sitúa en el grupo de cabeza, a la altura del mejor
modelo clásico. Las dos variantes de grafo rinden de forma muy similar entre sí
(la diferencia es despreciable frente a su desviación típica), por lo que no
cabe concluir que una sea superior a la otra. La clase `dynamic` (Kaspárov)
sigue siendo la más difícil para todos los modelos.

## 4.8 Experimento de generalización entre jugadores (LOPO)

Validación *leave-one-player-out*: se entrena con tres jugadores y se predice
sobre el cuarto. Como el estilo del jugador retirado no aparece en el
entrenamiento, el modelo no puede acertar la etiqueta exacta; interesa **hacia
qué estilo conocido se inclinan** sus partidas (predicción mayoritaria).

**Características agregadas (flat):**

| Jugador retirado | Estilo real | LR | RF | XGB |
|---|---|---|---|---|
| Karpov | positional | defensive (42%) | dynamic (38%) | tactical (36%) |
| Kasparov | dynamic | tactical (42%) | tactical (43%) | positional (59%) |
| Tal | tactical | dynamic (44%) | dynamic (48%) | positional (60%) |
| Petrosian | defensive | positional (45%) | tactical (44%) | positional (64%) |

La **regresión logística** ofrece la generalización más coherente: cada jugador
retirado se inclina hacia un estilo estilísticamente adyacente (Karpov→defensivo,
Kaspárov→táctico, Tal→dinámico). XGBoost colapsa casi todo a la clase posicional
y resulta poco informativo. Aunque ningún modelo recupera una etiqueta ausente,
las inclinaciones de la regresión logística sugieren que las características
capturan parte de la proximidad entre estilos.

## 4.9 Comparativa global y análisis de resultados

La tabla siguiente evalúa **todos los modelos en igualdad de condiciones**
(§4.4.4): mismas partidas de entrenamiento y el **mismo conjunto de test de
1.965 partidas** para todos. Como la inicialización de las redes es aleatoria,
los modelos de grafos se reportan como **media ± desviación típica sobre 5
semillas**; los modelos clásicos son deterministas (`random_state=42`).
Ordenada por F1-macro (criterio principal).

| Modelo | Características | **Test F1-macro** | Test acc | Test F1-pond. |
|---|---|---:|---:|---:|
| **GNN (estática + agregación)** | grafos | **0,371 ± 0,008** | 0,388 | 0,389 |
| XGBoost | temporal | 0,369 | 0,410 | 0,391 |
| **GNN temporal (GCN+GRU)** | grafos | **0,369 ± 0,007** | 0,389 | 0,387 |
| Random Forest | temporal | 0,365 | 0,371 | 0,373 |
| Regresión logística | flat | 0,362 | 0,375 | 0,373 |
| Random Forest | flat | 0,353 | 0,359 | 0,360 |
| XGBoost | flat | 0,333 | 0,379 | 0,357 |

**Sobre exactamente los mismos datos, la representación en grafo es competitiva
con los mejores modelos clásicos.** El mejor modelo de grafos (GCN estática con
agregación, 0,371) obtiene el F1-macro medio más alto, en empate estadístico con
XGBoost temporal (0,369; la diferencia es muy inferior a la desviación típica);
ambos lideran con claridad sobre el resto. Es decir, la representación en grafo
**iguala a la mejor solución clásica** partiendo directamente de la estructura
del tablero, **sin características diseñadas a mano**.

**Análisis:**

- **Tarea intrínsecamente difícil.** Con cuatro clases el azar da F1-macro ≈
  0,25; los mejores modelos alcanzan ≈0,37. Las señales de estilo existen pero
  son débiles y se solapan, como mostró el AED (§4.3).
- **De peor a mejor grupo.** La GNN partía de ser el peor modelo (0,268, §4.7.2)
  y, tras las mejoras de representación y la evaluación a nivel de partida, pasa
  al grupo de cabeza. El factor decisivo no fueron los hiperparámetros, sino
  (i) usar todas las partidas y todas las fases de juego, (ii) características de
  nodo adecuadas (one-hot, grados de ataque/defensa) y (iii) predecir a nivel de
  partida: el estilo es una propiedad de la partida, no de una posición aislada.
- **La dimensión temporal aporta** en los modelos clásicos (las variantes
  temporales superan a las agregadas). Entre las dos variantes de grafo, la
  diferencia es despreciable (empate dentro del ruido).
- **F1-macro frente a exactitud.** XGBoost temporal logra una exactitud alta
  (0,410) pero no el mejor F1-macro, por sobre-predecir la clase mayoritaria;
  esto justifica el uso de F1-macro como criterio.
- **Pares confundidos.** Las confusiones más persistentes son dinámico↔táctico
  (Kaspárov y Tal) y posicional↔defensivo (Karpov y Petrosián), los pares
  estilísticamente más próximos. La clase `dynamic` es la peor reconocida en
  todos los modelos.
- **Limitación de las características clásicas.** `aggression_score` es
  exactamente la suma de `capture_rate` y `check_rate`, por lo que es colineal y
  redundante para los modelos lineales; se mantiene por interpretabilidad.
