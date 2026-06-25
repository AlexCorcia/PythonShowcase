# Comparativa global de modelos (ordenada por F1-macro de test)

| modelo              | features   | search_kind   |   cv_f1_macro |   test_accuracy |   test_f1_macro |   test_f1_weighted |   elapsed_s |
|:--------------------|:-----------|:--------------|--------------:|----------------:|----------------:|-------------------:|------------:|
| random_forest       | temporal   | random        |        0.3737 |          0.3888 |          0.3802 |             0.3901 |      272.43 |
| logistic_regression | flat       | grid          |        0.3463 |          0.3803 |          0.3656 |             0.3774 |       10.48 |
| xgboost             | temporal   | random        |        0.3559 |          0.412  |          0.3579 |             0.3856 |      271.88 |
| random_forest       | flat       | random        |        0.3531 |          0.3662 |          0.3569 |             0.367  |      263.99 |
| xgboost             | flat       | random        |        0.319  |          0.3888 |          0.3338 |             0.3617 |      217.49 |
| gnn                 | graphs     | random        |        0.2758 |          0.2798 |          0.268  |             0.268  |      182.65 |

## Mejores hiperparámetros por modelo

- **random_forest / temporal** (random, F1-macro test=0.3802): `n_estimators=300, min_samples_split=10, min_samples_leaf=8, max_features=log2, max_depth=None, class_weight=balanced`
- **logistic_regression / flat** (grid, F1-macro test=0.3656): `C=0.01, class_weight=balanced`
- **xgboost / temporal** (random, F1-macro test=0.3579): `subsample=0.7, reg_lambda=1.0, reg_alpha=0.1, n_estimators=600, min_child_weight=5, max_depth=3, learning_rate=0.05, gamma=0.1, colsample_bytree=0.9`
- **random_forest / flat** (random, F1-macro test=0.3569): `n_estimators=600, min_samples_split=2, min_samples_leaf=2, max_features=log2, max_depth=6, class_weight=balanced`
- **xgboost / flat** (random, F1-macro test=0.3338): `subsample=0.9, reg_lambda=2.0, reg_alpha=1.0, n_estimators=600, min_child_weight=5, max_depth=3, learning_rate=0.1, gamma=0.1, colsample_bytree=0.8`
- **gnn / graphs** (random, F1-macro test=0.2680): `hidden_dim=256, lr=0.001, weight_decay=1e-05, dropout=0.2, num_layers=3, batch_size=64`
