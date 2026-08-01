"""
Trains 5 classification models on the Student Dropout & Academic Success dataset
(3-class target: Dropout / Enrolled / Graduate), using proper preprocessing:
- Numerical features -> StandardScaler
- Categorical (nominal-code) features -> OneHotEncoder
combined via ColumnTransformer and wrapped in a Pipeline per model, so each
saved .pkl file is a complete, ready-to-predict-on-raw-data pipeline.

Models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest

Saves:
- model/<safe_name>.pkl -> fitted pipeline (preprocessing + classifier)
- model/<safe_name>_classification_report.csv -> full per-class precision/recall/f1 report
- model/<safe_name>_confusion_matrix.csv -> confusion matrix (rows/cols = class names)
- model/metrics_comparison.csv -> comparison table across all 5 models, sorted by Accuracy
- model/best_model.json -> name of the top-accuracy model, for app.py to auto-select
- model/label_encoder.pkl -> fitted LabelEncoder for the target
- model/feature_names.json -> original (pre-encoding) feature columns
- model/categorical_columns.json -> the categorical columns fed to OneHotEncoder
- model/numerical_columns.json -> the numerical columns fed to StandardScaler
- model/class_names.json -> ordered class name list
- model/encoded_feature_names.json -> feature names AFTER one-hot encoding (236 total)
"""

import pandas as pd
from pathlib import Path
import json
import joblib
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

# ---- Path constants  -------
SCRIPT_DIR = Path(__file__).resolve().parent           # .../project/model
PROJECT_ROOT = SCRIPT_DIR.parent                        # .../project
MODEL_DIR = SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_ONLY_PATH = DATA_DIR / "train_only.csv"

RANDOM_STATE = 42


def safe_filename(name: str) -> str:
    """Convert a model display name into a filesystem-safe stem, e.g.
    'Random Forest' -> 'random_forest', 'k-NN (v2)' -> 'k_nn_v2'."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def main() -> None:
    # ---- Load training data (excludes the held-out test_data.csv used by the app) ----
    df = pd.read_csv(TRAIN_ONLY_PATH)
    X = df.drop("Target", axis=1)
    y_raw = df["Target"]

    categorical_columns = [
        "Marital status",
        "Application mode",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "International",
    ]

    missing = set(categorical_columns) - set(X.columns)
    if missing:
        raise ValueError(f"Missing expected categorical columns: {missing}")

    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    print("Categorical Features :", len(categorical_columns))
    print("Numerical Features :", len(numerical_columns))

    # ---- Encode target ----
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)
    print("\nTarget Classes")
    for i, label in enumerate(label_encoder.classes_):
        print(i, ":", label)

    # ---- Train/validation split (within the 80% training portion) ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print("\nTraining Set :", X_train.shape)
    print("Validation Set :", X_val.shape)

    # ---- Shared preprocessing: OneHotEncode categoricals, StandardScale numerics ----
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )

    # --- Define the 5 models to train ---- 
    model_definitions = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, clf in model_definitions.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)

        metrics = {
            "ML Model Name": name,
            "Accuracy": round(accuracy_score(y_val, preds), 4),
            "AUC": round(roc_auc_score(y_val, probs, multi_class="ovr", average="macro"), 4),
            "Precision": round(precision_score(y_val, preds, average="weighted", zero_division=0), 4),
            "Recall": round(recall_score(y_val, preds, average="weighted", zero_division=0), 4),
            "F1": round(f1_score(y_val, preds, average="weighted", zero_division=0), 4),
            "MCC": round(matthews_corrcoef(y_val, preds), 4),
        }
        results.append(metrics)
        fitted_pipelines[name] = pipeline

        print(f"\n{name}")
        for metric, value in metrics.items():
            if metric != "ML Model Name":
                print(f"{metric:<12}: {value}")

        safe_name = safe_filename(name)

        # ---- Per-model classification report ----
        report = classification_report(
            y_val, preds, target_names=class_names, output_dict=True, zero_division=0
        )
        pd.DataFrame(report).transpose().to_csv(
            MODEL_DIR / f"{safe_name}_classification_report.csv"
        )

        # ---- Per-model confusion matrix ----
        cm = confusion_matrix(y_val, preds)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            MODEL_DIR / f"{safe_name}_confusion_matrix.csv"
        )

    # ---- Save comparison table, sorted by Accuracy (best first) ----
    results_df = (
        pd.DataFrame(results)
        .sort_values(by="Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    results_df.to_csv(MODEL_DIR / "metrics_comparison.csv", index=False)
    print("\nComparison table (sorted by Accuracy):\n", results_df)

    # ---- Save the best-performing model's name, so app.py can default to it ----
    best_model_name = results_df.iloc[0]["ML Model Name"]
    with open(MODEL_DIR / "best_model.json", "w") as f:
        json.dump({"best_model": best_model_name}, f)
    print(f"\nBest model by Accuracy: {best_model_name}")

    # ---- Save fitted pipelines (preprocessing + model bundled together) ----
    for name, pipeline in fitted_pipelines.items():
        safe_name = safe_filename(name)
        joblib.dump(pipeline, MODEL_DIR / f"{safe_name}.pkl")

    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(list(X.columns), f)
    with open(MODEL_DIR / "categorical_columns.json", "w") as f:
        json.dump(categorical_columns, f)
    with open(MODEL_DIR / "numerical_columns.json", "w") as f:
        json.dump(numerical_columns, f)
    with open(MODEL_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    # ---- Save post-one-hot-encoding feature names (for interpretation / feature importance) ----
    # All 5 pipelines share the same ColumnTransformer definition, so any one of them
    # gives the representative expanded feature list.
    sample_pipeline = next(iter(fitted_pipelines.values()))
    encoded_feature_names = sample_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    with open(MODEL_DIR / "encoded_feature_names.json", "w") as f:
        json.dump(encoded_feature_names, f)
    print(f"\nEncoded feature count after preprocessing: {len(encoded_feature_names)}")

    # ---- Save per-feature default values (median for numeric, mode for categorical) ----
    # Used by the Streamlit app's single-student prediction form, so the user only
    # has to fill in a handful of the most relevant fields and everything else
    # defaults to a "typical student" value instead of zeros/blanks.
    default_values = {}
    for col in numerical_columns:
        default_values[col] = float(X[col].median())
    for col in categorical_columns:
        default_values[col] = X[col].mode(dropna=True).iloc[0]

        if hasattr(default_values[col], "item"):
            default_values[col] = default_values[col].item()
    with open(MODEL_DIR / "default_values.json", "w") as f:
        json.dump(default_values, f)
    print("Default (median/mode) feature values saved for the single-student form.")

    print("\nAll pipelines, per-model reports/confusion matrices, and metadata saved to /model")
    print("Class names:", class_names)


if __name__ == "__main__":
    main()
