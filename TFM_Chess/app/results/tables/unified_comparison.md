# Comparación en igualdad de condiciones (mismo split, mismas partidas de test)

Todas las filas comparten el MISMO conjunto de test (1965 partidas) y el mismo 60% de entrenamiento.

| modelo                               |   test_f1_macro |   f1_macro_std |   test_accuracy |   test_f1_weighted |   n_test |
|:-------------------------------------|----------------:|---------------:|----------------:|-------------------:|---------:|
| GNN (estática + agregación) · grafos |          0.371  |         0.008  |          0.3876 |             0.3887 |     1965 |
| XGBoost · temporal                   |          0.3692 |         0      |          0.4102 |             0.3914 |     1965 |
| GNN temporal (GCN+GRU) · grafos      |          0.3692 |         0.0069 |          0.3885 |             0.3867 |     1965 |
| Random Forest · temporal             |          0.3647 |         0      |          0.3705 |             0.3733 |     1965 |
| Regresión logística · flat           |          0.3623 |         0      |          0.3746 |             0.3731 |     1965 |
| Random Forest · flat                 |          0.3529 |         0      |          0.3588 |             0.3596 |     1965 |
| XGBoost · flat                       |          0.3325 |         0      |          0.3791 |             0.3571 |     1965 |