# Actualizaciones necesarias en §2 y §3 (por el overhaul de la GNN)

> **Borrador para revisión.** Texto en español, tono divulgativo coherente con el
> resto del Marco Teórico. Revísalo y reescríbelo con tus palabras antes de
> incorporarlo (regla UNIR de no usar IA generativa para la prosa de la memoria).

## Por qué hacen falta estos cambios

El código de la representación en grafo cambió sustancialmente: ahora hay **dos
arquitecturas** (una GCN estática con agregación a nivel de partida y un modelo
**temporal GCN+GRU**), y la comparación final se hace **en igualdad de
condiciones**. El §4 es técnico, pero la directora pide que **toda la teoría esté
en §2**. Hoy §2.3 explica los grafos *como dato*, pero **no explica**:
1. cómo aprende una **red neuronal sobre grafos (GNN/GCN)** —paso de mensajes y
   pooling—, y
2. qué es un **modelo de secuencia (RNN/GRU)**, que usa el modelo temporal.

Por eso se añaden dos subsecciones a §2.3 y se ajustan dos textos de §3.

---

## CAMBIO 1 — §2.3 "Grafos temporales dinámicos": añadir un párrafo de cierre

Tras el último párrafo actual (el que empieza "En este TFM, la representación
basada en grafos se plantea como una primera aproximación…"), añadir:

> Conviene distinguir entre la **representación del dato** y el **modelo de
> aprendizaje**. El dato es dinámico: una partida es una secuencia ordenada de
> grafos (un grafo por posición) cuya estructura cambia con cada movimiento.
> Sobre ese dato caben dos estrategias de modelado: (a) procesar cada grafo de
> forma independiente con una red neuronal sobre grafos y **agregar después** las
> predicciones a nivel de partida, o (b) procesar la **secuencia completa** con un
> modelo capaz de capturar su evolución temporal. En este trabajo se exploran
> ambas (véase §4.7).

---

## CAMBIO 2 — añadir nueva subsección §2.3.x "Redes neuronales sobre grafos (GNN)"

> Una red neuronal sobre grafos (GNN) es un modelo capaz de aprender directamente
> a partir de la estructura de un grafo, sin convertirlo previamente en una tabla
> de características. Su funcionamiento se basa en el **paso de mensajes**: cada
> nodo actualiza su representación combinando su propia información con la de los
> nodos vecinos a los que está conectado. Al repetir este proceso en varias capas,
> la información se propaga por el grafo y cada nodo acaba codificando tanto sus
> propias características como su contexto estructural.
>
> Para clasificar un grafo completo —como una posición de ajedrez— es necesario
> resumir las representaciones de todos los nodos en un único vector. Esto se
> realiza mediante una operación de **agregación** o *pooling* (por ejemplo, la
> media de los vectores de todos los nodos), que produce una representación global
> del grafo sobre la que se aplica una capa final de clasificación. Una variante
> habitual es la **red convolucional sobre grafos (GCN)**, que define el paso de
> mensajes como una convolución adaptada a la estructura del grafo.

---

## CAMBIO 3 — añadir nueva subsección §2.3.x "Modelos de secuencia (RNN/GRU)"

> Cuando el dato no es un único elemento, sino una **secuencia ordenada** —como la
> sucesión de posiciones de una partida—, resulta útil emplear modelos que
> procesen esa secuencia teniendo en cuenta el orden. Las **redes neuronales
> recurrentes (RNN)** están diseñadas para ello: mantienen un **estado interno**
> que se actualiza a medida que reciben cada elemento de la secuencia, de modo que
> la representación final resume toda la evolución observada.
>
> Una de sus variantes más utilizadas es la **GRU** (*Gated Recurrent Unit*), que
> incorpora mecanismos de puerta para controlar qué información se conserva y cuál
> se descarta en cada paso, lo que facilita aprender dependencias a lo largo de
> secuencias más largas. En este trabajo, una GRU permite combinar las
> representaciones de los sucesivos grafos de una partida en una única predicción
> de estilo, aprovechando así la dimensión temporal del dato (véase §4.7).

---

## CAMBIO 4 — §2.3 "Conclusiones" (último párrafo, el que enumera las técnicas)

El párrafo actual termina: "…incluyendo aprendizaje supervisado, modelos clásicos
de clasificación, optimización de hiperparámetros, métricas de evaluación y
grafos temporales dinámicos."

Sustituir el final por:

> …incluyendo aprendizaje supervisado, modelos clásicos de clasificación,
> optimización de hiperparámetros, métricas de evaluación, grafos temporales
> dinámicos, **redes neuronales sobre grafos (GNN/GCN) y modelos de secuencia
> (GRU)**.

---

## CAMBIO 5 — §3.2 Objetivos específicos (el objetivo sobre la GNN)

**Texto actual:**
> Evaluar una primera arquitectura GNN sobre los grafos construidos y comparar sus
> resultados con los modelos anteriores.

**Texto propuesto:**
> Evaluar arquitecturas de redes neuronales sobre grafos —una red convolucional
> (GCN) con agregación a nivel de partida y un modelo temporal que combina la GCN
> con una red recurrente (GRU) sobre la secuencia de posiciones— y compararlas con
> los modelos anteriores **en igualdad de condiciones** (misma partición de
> entrenamiento y de prueba).

---

## CAMBIO 6 — §3.3 Metodología (el párrafo final sobre grafos)

**Texto actual:**
> Finalmente, se exploró una representación basada en grafos de posiciones. En esta
> aproximación, las piezas se representaron como nodos y las relaciones de ataque y
> defensa como aristas. Sobre estos grafos se evaluó una primera arquitectura GNN
> sencilla, con el objetivo de comprobar si la estructura interna de las posiciones
> podía aportar información adicional a la clasificación de estilos.

**Texto propuesto:**
> Finalmente, se exploró una representación basada en grafos de posiciones. Cada
> partida se modeló como una secuencia de grafos (un grafo por posición), con las
> piezas como nodos y las relaciones de ataque y defensa como aristas. Sobre esta
> representación se evaluaron **dos arquitecturas**: una red convolucional sobre
> grafos (GCN) que clasifica cada posición y **agrega sus predicciones a nivel de
> partida**, y un **modelo temporal** que procesa la secuencia de posiciones
> mediante una red recurrente (GRU). En ambos casos la unidad de predicción es la
> partida, igual que en los modelos clásicos, y sus hiperparámetros se ajustaron
> mediante búsqueda. Para que la comparación entre todas las familias fuera
> rigurosa, los modelos se evaluaron sobre **las mismas particiones de
> entrenamiento y prueba**; como el entrenamiento de las redes neuronales no es
> determinista, sus resultados se **promediaron sobre varias semillas**.

---

## Lo que NO hay que cambiar

- §2.1, §2.2 (estado del arte / revisión de literatura): sin cambios. La revisión
  de grafos temporales dinámicos de §2.2.4 incluso encaja mejor ahora con el
  modelo temporal GCN+GRU.
- §2.3 supervisado / modelos clásicos / HP / métricas: sin cambios.
- El objetivo general (§3.1): sin cambios.

## Discrepancias detectadas en el .docx (verificar)

1. El **índice de contenidos está desactualizado**: muestra §2.3 como
   "Conclusiones" y no lista "Marco Teórico". Hay que **actualizar el índice** tras
   los cambios (clic derecho → Actualizar campo, en Word).
2. **§2.2.5 y §2.2.6 tienen el mismo título** ("Revisión de técnicas de
   aprendizaje en grafos dinámicos") en esta versión del .docx. Si la corrección
   de §2.2.6 → "Representación basada en grafos en ajedrez" ya se aplicó en otra
   copia, ignorar; si no, aplicarla.
3. En §4 hay **dos subsecciones con el título "Resultados experimentales"** (en
   §4.7) y una numeración irregular; conviene revisarla al integrar el nuevo §4.
