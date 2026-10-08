# Model Evaluation

Reproduce with `python train_and_evaluate.py` (all numbers below come from `reports/evaluation_results.json`).

## Models tested

| Model | Settings |
|---|---|
| Random Forest | 200 trees, `max_features="sqrt"`, seed 42 (same as the previously deployed model) |
| XGBoost | 300 trees, depth 6, learning rate 0.1, subsample/colsample 0.9, seed 42 |
| Logistic Regression (baseline) | L2, `max_iter=2000` |

No hyperparameter tuning was done; each model uses reasonable fixed settings.

## Features and target

- Features (order fixed, unchanged): `N, P, K, temperature, humidity, ph, rainfall`
- Target: 22 crop classes, 100 samples each (balanced)
- Data: `data/Crop_recommendation.csv` (2,200 rows, no missing values, no duplicate rows)

## Methodology

1. Stratified 80/20 train/test split (1,760 / 440), seed 42. The test set is not used for model selection.
2. 5-fold stratified cross-validation on the training split. `StandardScaler` sits inside a pipeline, so it is re-fit per fold (no leakage).
3. Metrics: accuracy, macro precision, macro recall, macro F1 (CV mean), plus held-out test metrics and a confusion matrix per model.
4. Selection rule: highest mean CV accuracy (tie-break: CV macro-F1).
5. The selected model is refit on the full dataset for deployment, as the previous pipeline did. The test numbers below come from the train-only fit, so they were not computed on data the model had seen.

## Results

5-fold CV on the training split (mean ± std of accuracy):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **0.9938 ± 0.0045** | 0.9943 | 0.9938 | 0.9937 |
| XGBoost | 0.9926 ± 0.0014 | 0.9930 | 0.9926 | 0.9926 |
| Logistic Regression | 0.9676 ± 0.0064 | 0.9703 | 0.9676 | 0.9673 |

Held-out test set (440 rows; precision/recall/F1 are macro-averaged):

| Model | Accuracy | Precision | Recall | F1 | Misclassified |
|---|---|---|---|---|---|
| Random Forest | 0.9955 | 0.9957 | 0.9955 | 0.9955 | 2 |
| XGBoost | 0.9909 | 0.9913 | 0.9909 | 0.9908 | 4 |
| Logistic Regression | 0.9727 | 0.9740 | 0.9727 | 0.9725 | 12 |

Confusion matrices: `reports/confusion_matrix_*.csv` (all three models) and `reports/confusion_matrix_selected.png`. Random Forest's two test errors were blackgram→maize and rice→jute; Logistic Regression's errors cluster in rice/jute and lentil/mothbeans.

## Selected model

**Random Forest**, chosen on CV accuracy. The gap to XGBoost (0.12 percentage points) is well inside one standard deviation of the fold-to-fold variation, so the two are effectively tied; Random Forest is also the architecture the app already used. Logistic Regression is clearly weaker, which indicates some non-linear structure in the data.

Saved artifacts (same format `app.py` already loads):
- `models/crop_model.pkl`: classifier trained on scaled features (supports `predict_proba`)
- `models/preprocessing.pkl`: `{"scaler", "label_encoder", "feature_names"}`

## Limitations

- **High dataset accuracy is not real-world farming accuracy.** These scores measure how well the model reproduces labels in this specific dataset. They say nothing about whether the recommended crop would actually be profitable or successful on a real farm.
- The dataset is small (100 samples per crop), balanced, and appears clean and idealised, with well-separated crop profiles. Real conditions overlap far more, and near-perfect scores on such data are a warning sign of an easy benchmark, not a guarantee.
- Many crops are labelled by the conditions they are typically grown in, not by yield or outcome data. The model learns "which crop profile do these numbers resemble", not "which crop will perform best".
- The random split may keep near-duplicate samples on both sides. There is no geographic, seasonal or temporal hold-out, so generalisation to new regions or years is untested.
- Inputs ignore soil type, season, irrigation, pests, market prices, crop rotation and farmer practice. N and K were rescaled in an earlier correction step (see `CHANGELOG_NPK_CORRECTION.md`); live inputs (weather API, user-entered soil values) may differ in distribution from the training data.
- Model probabilities are not calibrated and should not be read as the chance of success.
- Only 3 models with fixed hyperparameters were compared; no tuning or calibration was attempted.
- Differences between Random Forest and XGBoost are a handful of test samples and not statistically meaningful.
- The model's output is one signal in the app, combined with regional and weather rules, and should be treated as decision support, not a substitute for local agronomic advice or a soil test.
