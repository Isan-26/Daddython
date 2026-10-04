# V4 uncertainty averaging: implementation and checks

The requested multiple-imputation inference experiment is complete in platform_submission_v4. All existing v2 models remain frozen. No deep-learning, imputation, Ridge, HGB, pattern-expert or scaler model was fitted. The build completed in 15.6 seconds using the existing artifacts.

## Submission artifacts

- prediction_notebook.ipynb and model.pkl: standalone upload pair, default v4_uncertainty configuration.
- candidate_v4_uncertainty.csv: active 64-draw averaging experiment.
- candidate_v4_conservative.csv: fixed 25% application of that experiment's change.
- candidate_v2_fallback.csv: exact original platform v2 predictions.

All CSVs contain 3,000 rows, id,prediction and original test order. The notebook output uses a single prediction column, as required by the platform contract. The pickle is approximately 79.52 MB and contains the original v2 objects plus training-input uncertainty tables; no custom estimator classes are needed at upload time.

## Exact scope

The original v2 Ridge/HGB imputed energy branch is evaluated with 64 plausible fills for each incomplete row. Energy predictions are averaged **after** evaluating each draw. The branch's original final ensemble weight, 0.619875, remains fixed.

Donors are 7,537 complete training-input rows. Each missingness pattern has a joint empirical auxiliary residual table computed by inference with the existing v2 imputers. Nearby donors are selected using building, calendar and originally observed numeric values. Centered residual vectors are sampled jointly with positive/negative pairs and symmetric support bounds. This preserves each original imputed mean and correlations among perturbed inputs.

The expected-occupancy and expected-previous features stay based on the original raw row. Sampled fills are never passed into neural preprocessing or direct missing-pattern routing. All original ensemble weights, the MissForest branch, direct experts, neural predictions and missing-previous +0.8 remain unchanged.

Randomness is seeded by original row contents rather than ID, label, row position or batch. Finite sampled fills affect only original blanks.

## Measured changes, not accuracy gains

| Check | Result |
|---|---:|
| Models fitted | 0 |
| Complete test rows exactly unchanged | 2,453 |
| Incomplete test rows evaluated by averaging | 547 |
| RMS change on incomplete rows | 0.07999 |
| Maximum absolute change on incomplete rows | 0.45496 |
| Original-v2 fallback CSV maximum difference | 0.0 |

The 25% alternative reduces those prediction changes by a factor of four. It was specified as an optional configuration and was not selected using labels or leaderboard feedback.

The first 150 incomplete test rows were evaluated using 32, 64 and 128 draws. The RMS difference between 64 and 128 draws, measured after applying the branch's final ensemble weight, was 0.01194; maximum difference was 0.05227. These are Monte Carlo stability measurements, not energy RMSE measurements or proof that 128 draws is more accurate.

## Verification

Eight unit tests passed, including nonlinear prediction averaging, preservation of observed values and imputed means, joint sampling and physical bounds, row/order/batch independence, original expected features, correct branch weight, untouched baseline missingness, zero-update short circuit, and exclusion of self donors.

The standalone notebook passed isolated execution with only the artifact and input files available. Its output matches the main CSV exactly; repeat, shuffled, ID-bearing and separate-batch inputs agree. All 16 missingness patterns produce finite results, including every numeric input missing. Complete-row predictions and the fallback match original v2 exactly. The conservative configuration also matches its CSV.

The original model's serialized-state hash was identical before and after the build. Original model, notebook, train and test file hashes were unchanged. All 22 copied v2 inference functions/classes have exactly the same AST as the original notebook. Builder/runtime sources contain no estimator training calls; fit methods were blocked during the build.

## Validation limits

**No new honest OOF energy RMSE is reported.** The saved energy models were trained on all original training labels, and matching saved fold model objects are unavailable. Producing such a comparison would require refitting or access to genuinely unused labeled data, outside the user's no-retraining constraint.

The sampling table uses **in-sample auxiliary imputer residuals**. It is not an OOF-calibrated posterior. Symmetric bootstrapping also approximates potentially skewed or multimodal uncertainty. These limitations can reduce or misrepresent the spread of plausible missing values.

On 600 artificially masked training donors with self donors excluded, empirical coverage of nominal 90% intervals was 84.1% for temperature, 84.7% for humidity, 73.3% for occupancy and 85.3% for previous usage. The underlying imputers still saw these input values during their original fitting, so these diagnostic coverage figures are optimistic and cannot establish test coverage. Occupancy uncertainty is particularly undercovered in this diagnostic.

The main CSV actively implements the experiment, but no evidence yet establishes that it lowers the leaderboard score. Keep the original best submission available. The v2 fallback is a reproduction of the saved platform v2 artifact; it does not establish equivalence to differently generated historical root-level submissions.
