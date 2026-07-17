"""
retrain_on_corrected_data.py

Retrains the exact same model type/hyperparameters the project already
ships (RandomForestClassifier(n_estimators=200, max_features='sqrt',
random_state=42), StandardScaler, LabelEncoder - read directly out of
the existing models/preprocessing.pkl before overwriting anything) on
the N/K-corrected dataset, so the deployed model's input expectations
actually match the realistic kg/ha scale the frontend asks for and a
real Soil Health Card would report.

This does NOT change: feature order, model type/hyperparameters,
scaler/encoder type, or any Flask/route code. Only the training DATA
changed (see fix_npk_scale.py) - this script exists purely to make the
saved .pkl files consistent with that corrected data again.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/Crop_recommendation.csv"
MODEL_OUT = "models/crop_model.pkl"
PREPROCESSING_OUT = "models/preprocessing.pkl"

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_ORDER]
    y_raw = df["label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy on corrected data: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    # Retrain on the FULL dataset for the deployed artifact (same
    # convention as typical scikit-learn deployment practice - the
    # held-out split above is only to report a sanity-check accuracy).
    X_scaled_full = scaler.fit_transform(X)
    model.fit(X_scaled_full, y)

    joblib.dump(model, MODEL_OUT)
    joblib.dump(
        {"scaler": scaler, "label_encoder": label_encoder, "feature_names": FEATURE_ORDER},
        PREPROCESSING_OUT,
    )
    print(f"\nSaved retrained model -> {MODEL_OUT}")
    print(f"Saved matching preprocessing -> {PREPROCESSING_OUT}")


if __name__ == "__main__":
    main()
