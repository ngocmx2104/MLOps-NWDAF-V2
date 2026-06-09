## Exp-1: E2E Quality (ROC-AUC / PR-AUC / F1)

| metric | n | mean | std | min | max |
| --- | --- | --- | --- | --- | --- |
| roc_auc | 10 | 0.9261 | 0.005519 | 0.9173 | 0.9323 |
| pr_auc | 10 | 0.6761 | 0.02481 | 0.6329 | 0.7119 |
| f1 | 10 | 0.09076 | 0.01864 | 0.0649 | 0.1203 |
| precision | 10 | 0.9721 | 0.03797 | 0.9091 | 1 |
| recall | 10 | 0.04771 | 0.01032 | 0.03364 | 0.06422 |


## Exp-2: C0 vs C1 Ablation

| metric | C0 | C1 | delta | p_value | significant |
| --- | --- | --- | --- | --- | --- |
| wall_s (mean) | 3.198 | 4.941 | 1.743 | 0.001953 | True |
| storage_delta_bytes | 0 | 745472 | 745472 | N/A | N/A |
| ml_test_score | 0 | 7 | 7 | N/A | N/A |
| traceable | False | True | N/A | N/A | N/A |


## Exp-3: Framework Overhead + Governance

| backend | wall_s_mean | wall_s_std | delta_wall_s | rss_mb_mean | delta_rss_mb | registry | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| noop | 2.291 | 0.3278 | 0 | 399.4 | 0 | False | False |
| mlflow | 3.388 | 0.2292 | 1.097 | 515 | 115.6 | True | True |
| clearml | 3.549 | 0.3389 | 1.258 | 484.5 | 85.1 | True | True |


## Exp-4: Drift Detection + Closed-Loop Retrain (ON vs OFF)

| scenario | drift_detected_any | detection_latency_steps | retrain_on | retrain_off |
| --- | --- | --- | --- | --- |
| sudden | True | 3 | 2 | 0 |
| gradual | True | 3 | 2 | 0 |
| recurring | False | -1 | 0 | 0 |


## Exp-5: Model-Swap IForest vs LSTM-AE

| model | roc_auc_mean | pr_auc_mean | f1_mean | train_wall_s_mean | n_ok | swap_core_changes |
| --- | --- | --- | --- | --- | --- | --- |
| iforest | 0.9261 | 0.6761 | 0.09076 | 1.885 | 10 | 0 |
| lstm_ae | 0.9478 | 0.7789 | 0.3325 | 3.925 | 10 | 0 |


## Exp-6: Modifiability (NWDAF Mods + Regression)

| mod_id | section | files_changed | lines_changed | regression_count | pass |
| --- | --- | --- | --- | --- | --- |
| add_feature | data | 1 | 1 | 0 | True |
| change_pingpong_rule | label | 1 | 4 | 0 | True |
| change_psi_threshold | monitoring | 1 | 2 | 0 | True |
