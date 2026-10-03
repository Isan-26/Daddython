# Notebooks sorted with their CSV outputs

Each folder holds a **copy** of one notebook and the CSV files it produced (or that it documents). The originals stay in the project root, because the notebooks read `train.csv`, `test.csv`, `candidate_v4.csv`, `experiments/` and the `.py` modules from there. Run the root versions; these folders are for browsing.

| Folder | Notebook | CSV files (public RMSE where known) | What it is |
|---|---|---|---|
| `01_linear_regression/` | `linearregression.ipynb` | `submission.csv`, `best.csv`, `best_v2.csv` (3.40), `best_v3.csv`, `gacor.csv` | First models (MissForest + Ridge / gradient boosting). The notebook now writes submission.csv; the best*.csv and gacor.csv files are earlier outputs from the first commits. |
| `02_eda/` | `eda_train_test.ipynb` | (none) | Exploratory analysis of train vs test (no CSV output). |
| `03_v4_pattern_models/` | `energy_model_review.ipynb` | `candidate_v4.csv` (3.37) | v4: one model per pattern of missing inputs + the v3 notebook model (code in energy_model.py). |
| `04_v5_catboost_imputed/` | `energy_model_v5.ipynb` | `candidate_v5.csv` | v5: CatBoost imputers + Ridge (code in energy_model_v5.py). |
| `05_v6/` | `energy_model_v6.ipynb` | `candidate_v6.csv` | v6 (code in energy_model_v6.py). |
| `06_best_model_v8/` | `best_model.ipynb` | `best_model.csv` (3.27026 (same as v8)), `candidate_v8.csv` (3.27026), `candidate_v7.csv` (3.27), `candidate_v5_lightgbm.csv` (3.29) | Best model before this session: 25% v4 + 75% LightGBM-imputed model (seed 527). best_model.csv is identical to candidate_v8.csv; v7 and v5_lightgbm are the same recipe at 35%/65% and 50%/50%. |
| `07_final_model_v9/` | `final_model.ipynb` | `candidate_v9.csv`, `candidate_v6_blend.csv` (3.32) | Explains the v8 recipe and writes candidate_v9.csv (code in final_model.py); candidate_v6_blend.csv (30% v4 + 40% new model + 30% v2) is documented here. |
| `08_candidate_final_10_seeds/` | `candidate_final_pipeline.ipynb` | `candidate_final.csv` (3.28) | 30% v4 + 70% LightGBM-imputed model averaged over 10 seeds (steadier for the hidden ranking). |
| `09_clean_model/` | `clean_model.ipynb` | `candidate_clean.csv` | Clean-model variant (code in clean_model.py). |
| `10_middle_model/` | `middle_model.ipynb` | `candidate_middle.csv` | Middle-model variant (code in middle_model.py). |
| `11_ensemble/` | `ensemble_model.ipynb` | `candidate_ensemble.csv`, `candidate_combined.csv` | Ensemble with CatBoost / friend-model predictions (cannot run on the platform). candidate_combined.csv was committed together with this notebook. |
| `12_neural_net/` | `neural_net_model.ipynb` | `candidate_nn.csv` (3.27148), `candidate_nn_w05.csv` (not submitted) | This session: best model + a scikit-learn neural network (15% -> candidate_nn.csv, 5% -> candidate_nn_w05.csv). |
| `13_deep_learning_realmlp/` | `deep_learning_model.ipynb` | `candidate_dl.csv` (3.26340), `candidate_gated.csv` (3.26880), `candidate_dl13.csv` (3.25670), `candidate_prevfix.csv` (3.25850), `candidate_dl13_prevfix.csv` (not submitted (predicted ~3.2512)) | This session: RealMLP (pytabkit) blended into candidate_nn_w05.csv, plus the previous_usage-blank correction. Needs the datathon-dl-env interpreter. |
| `other_outputs/` | (none) | `blend_v2_v4.csv`, `submission_v2_gated.csv`, `submission_platform_v2_gated.csv` | Outputs of scripts in experiments/ (second commit), not of a notebook. |

`platform_submission/` (prediction notebook + `model.pkl` for the platform run) was already its own folder and is unchanged.
