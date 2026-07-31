"""
Trains 5 classification models on the Student Dropout & Academic Success dataset
(3-class target: Dropout / Enrolled / Graduate), using proper preprocessing:
- Numerical features -> StandardScaler
- Categorical (nominal-code) features -> OneHotEncoder
combined via ColumnTransformer and wrapped in a Pipeline per model, so each
saved .pkl file is a complete, ready-to-predict-on-raw-data pipeline.
Logistic Regression and Decision Tree — with basic evaluation metrics printed 
to console.

Models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest
"""

import pandas as pd
from pathlib import Path
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)
 
# ---- Path constants (portable: relative to this file, not the OS) -------
SCRIPT_DIR = Path(__file__).resolve().parent           # .../project/model
PROJECT_ROOT = SCRIPT_DIR.parent                        # .../project
MODEL_DIR = SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_ONLY_PATH = DATA_DIR / "train_only.csv"
 
RANDOM_STATE = 42
 
 
def main() -> None:
    # ---- Load training data (excludes the held-out test_data.csv used by the app) ----
    df = pd.read_csv(TRAIN_ONLY_PATH)
    X = df.drop("Target", axis=1)
    y_raw = df["Target"]
 
    categorical_columns = [
        "Marital status",
        "Application mode",
        "Application order",
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
 
    model_definitions = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
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
            "Precision": round(precision_score(y_val, preds, average="weighted"), 4),
            "Recall": round(recall_score(y_val, preds, average="weighted"), 4),
            "F1": round(f1_score(y_val, preds, average="weighted"), 4),
            "MCC": round(matthews_corrcoef(y_val, preds), 4),
        }
        results.append(metrics)
        fitted_pipelines[name] = pipeline
        print(name, metrics)
 
    results_df = pd.DataFrame(results)
    print("\nComparison table so far:\n", results_df)
 
    # TODO: add kNN, Naive Bayes, Random Forest; 
    # save per-model classification reports and confusion matrices;
    # sort the comparison table and save the fitted pipelines + best model.
 
if __name__ == "__main__":
    main()