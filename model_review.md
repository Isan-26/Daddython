# Review of the energy-usage model

The latest approach is strong. The extra model I tested improved development cross-validation, but **the gain did not hold up on the reserved confirmation set**. Keep the current competition best as the default. `candidate_v4.csv` is an experimental submission, and a score of 3.20 on the hidden test labels has not been demonstrated.

## Results

I reserved 1,600 rows before selecting settings. The other 6,400 rows were used for four-fold model development. Both approaches used the same splits and masking trials. “v3” below reproduces the last section of `linearregression.ipynb`, with the imputer fitted only on training-fold rows.

| Evaluation | v3 | Experimental ensemble |
|---|---:|---:|
| Development: ordinary four-fold validation | 3.1052 | 3.0687 |
| Development: test missing-value patterns | 3.2498 | 3.2089 |
| Development: also reweighted for building/event mix | 3.5043 | 3.3955 |
| Reserved confirmation: ordinary validation | 3.1754 | 3.2029 |
| Reserved confirmation: test missing-value patterns | 3.2803 | 3.3026 |
| Reserved confirmation: also reweighted for building/event mix | 3.4407 | 3.4436 |

Lower is better. The event/building reweighting is a rough distribution-shift check, **not a leaderboard score prediction**. It uses feature-only event strata and smoothed density ratios. The confirmation test masks were sampled from actual test missing-value patterns and applied to the 1,506 complete confirmation rows, over ten trials. The same rows appear in each masking trial. A paired bootstrap resampled rows to account for that repetition; its gain interval included zero for both masked checks. Confirmation scores were not used to retune the candidate.

## Findings in your notebook

1. **The first models lose useful building information.** Cell 4 drops `building_id`. The same type contains buildings with different loads and responses. Your latest section already addresses this with building interactions; keep those.
2. **The first preprocessing step is fitted before the validation split, and is refitted on test.** Cell 13 calls `imputer.fit_transform` inside `clean_df`; cell 18 splits afterwards; cell 30 calls the same function on test and fits new means. Split raw rows first, fit preprocessing on the training fold, and call `transform` on validation/test using that fitted object. Keep the target outside preprocessing.
3. **The later imputer also uses prediction-set inputs.** Cell 35 calls `fit_imputer(pd.concat([train_df, pred_df]))`, despite its comment saying “TRAIN rows only.” The imputer excludes `energy_usage`, so this does not expose validation labels. It is transductive preprocessing: validation/test feature distributions influence the imputer. If you intend training-only validation, fit it on `train_df` alone. If competition rules allow using unlabelled test features, describe that choice accurately.
4. **The ElasticNet cell fails in a fresh run.** Cell 29 constructs an estimator and calls `predict` without `fit`. Add `elasticnet_baseline_model.fit(X_train, y_train)` before prediction, or remove that unfinished experiment.
5. **The original models used for `submission.csv` are still fitted on the training split.** Cell 30 uses the models from the holdout experiment. After choosing settings, refit on all labelled rows. Your later v3 submission section already does this.
6. **The current stress test adds too many blanks.** Cell 36 overlays copied test masks on validation rows that already have blanks. Missing values become the union of both masks, with higher rates than test. Score ordinary validation separately, then apply copied test masks to complete validation rows for an exact-pattern check.
7. **A random split misses part of the test difficulty.** Test contains more unusual rows and clustered blanks. A 10% split and one masking seed are unstable for comparing small improvements. Use multiple development folds, repeated masks, and a separate confirmation set.
8. **RMSE around 3.1 is not proof of a noise floor.** The EDA's linear-model RMSE is an observed model result. It does not establish an irreducible lower bound or make 3.20 impossible.

## What the experimental model changes

- Fits a supervised regression for each subset of observed numeric inputs. A row with missing `previous_usage`, for example, is predicted directly from its available features rather than a guessed previous value.
- Gives buildings their own effects and uses three daily sine/cosine harmonics.
- Adds modest residual tree corrections, with building identity treated as categorical.
- Partially weights training examples toward the unlabelled test building/event distribution.
- Blends 75% of those experts with 25% of the v3 architecture. Settings were fixed on development data before confirmation.

`energy_model.py` contains both architectures; `model_config.json` stores the selected settings. The candidate was fitted on all 8,000 labelled rows and writes the same `id,prediction` format and ID order as your existing submissions. The production fit uses no test targets. There are no additional dependencies.

## Practical recommendation

The experiment does not justify replacing your 3.4 competition best. The promising 3.21 development number alone would have been misleading. First make the validation and preprocessing corrections above so new experiments are measured consistently. Missing `previous_usage` and occupancy, plus the larger event share, remain useful targets for further work; the direct experts tested here did not produce a confirmed gain.

The exact submission behind your reported 3.4 score was not identified, so the comparison is against the latest v3 code rather than a verified leaderboard submission. The hidden test targets are unavailable here, so a 3.20 leaderboard score cannot be verified locally.

## Files

- `energy_model_review.ipynb`: readable results, reproducible confirmation, and candidate generation.
- `candidate_v4.csv`: 3,000 experimental predictions.
- `validation_results.json`: confirmation scores, uncertainty intervals, data hashes, and split details.
- `experiments/`: development scripts and results.
