"""
train_and_evaluate.py

ML experiment for the crop model: Random Forest vs XGBoost vs Logistic
Regression (baseline).

Method
------
1. Stratified 80/20 train/test split (random_state=42). The test set is
   never used for model selection.
2. 5-fold stratified cross-validation on the TRAIN split only. Scaling is
   inside a Pipeline so it is re-fit per fold (no leakage).
3. The best model is chosen by mean CV accuracy (tie-break: mean CV macro-F1).
4. Held-out test metrics (accuracy, macro precision/recall/F1, per-class
   report, confusion matrix) are reported for every model, from a fit on
   the train split only.
5. The selected model is then refit on the FULL dataset (same convention as
   the previous retrain_on_corrected_data.py) and saved in the exact format
   app.py already loads:
     models/crop_model.pkl       - bare classifier trained on SCALED features
     models/preprocessing.pkl    - {"scaler", "label_encoder", "feature_names"}

Usage:  python train_and_evaluate.py
Outputs: models/*.pkl, reports/evaluation_results.json,
         reports/confusion_matrix_<model>.csv, reports/confusion_matrix_selected.png
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = "data/Crop_recommendation.csv"
MODEL_OUT = "models/crop_model.pkl"
PREPROCESSING_OUT = "models/preprocessing.pkl"
REPORT_DIR = "reports"

# Must match the order app.py validates against preprocessing.pkl.
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=SEED, n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_ORDER]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    classes = list(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"Rows: {len(df)} | classes: {len(classes)} | train: {len(X_train)} | test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    results = {}
    for name, estimator in build_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        t0 = time.time()
        cvr = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_summary = {
            m: {"mean": float(np.mean(cvr[f"test_{m}"])), "std": float(np.std(cvr[f"test_{m}"]))}
            for m in scoring
        }

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, pred)
        results[name] = {
            "cv": cv_summary,
            "test": {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision_macro": float(p),
                "recall_macro": float(r),
                "f1_macro": float(f),
            },
            "per_class": classification_report(
                y_test, pred, target_names=classes, output_dict=True, zero_division=0
            ),
            "confusion_matrix": cm.tolist(),
        }
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            os.path.join(REPORT_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.csv")
        )
        print(
            f"\n{name}  ({time.time() - t0:.1f}s)\n"
            f"  CV   acc={cv_summary['accuracy']['mean']:.4f}±{cv_summary['accuracy']['std']:.4f} "
            f"P={cv_summary['precision_macro']['mean']:.4f} R={cv_summary['recall_macro']['mean']:.4f} "
            f"F1={cv_summary['f1_macro']['mean']:.4f}\n"
            f"  TEST acc={results[name]['test']['accuracy']:.4f} P={p:.4f} R={r:.4f} F1={f:.4f}"
        )

    # Selection: CV accuracy, tie-break CV macro-F1. Test set is NOT used.
    best = max(
        results,
        key=lambda n: (
            round(results[n]["cv"]["accuracy"]["mean"], 6),
            results[n]["cv"]["f1_macro"]["mean"],
        ),
    )
    print(f"\nSelected model (best 5-fold CV accuracy): {best}")

    # Confusion-matrix image for the selected model (optional dependency).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.array(results[best]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=90)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{best} - held-out test confusion matrix")
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix_selected.png"), dpi=130)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed - skipped confusion matrix PNG (CSV/JSON still written).")

    # Refit the selected model on the full dataset and save in the format
    # app.py expects (scaler + bare classifier on scaled features).
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X)
    final_model = build_models()[best]
    final_model.fit(X_full_scaled, y)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(final_model, MODEL_OUT)
    joblib.dump(
        {"scaler": scaler, "label_encoder": label_encoder, "feature_names": FEATURE_ORDER},
        PREPROCESSING_OUT,
    )
    assert hasattr(final_model, "predict_proba")  # app.py uses predict_proba

    with open(os.path.join(REPORT_DIR, "evaluation_results.json"), "w") as fh:
        json.dump(
            {
                "dataset": {"path": DATA_PATH, "rows": len(df), "classes": classes},
                "features": FEATURE_ORDER,
                "split": {"test_size": TEST_SIZE, "stratified": True, "seed": SEED},
                "cv_folds": CV_FOLDS,
                "selection_rule": "highest mean CV accuracy, tie-break mean CV macro-F1",
                "selected_model": best,
                "results": results,
            },
            fh,
            indent=2,
        )
    print(f"Saved model -> {MODEL_OUT}\nSaved preprocessing -> {PREPROCESSING_OUT}\nReports -> {REPORT_DIR}/")


if __name__ == "__main__":
    main()
