import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Student Dropout Classifier", page_icon="🎓", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


@st.cache_resource
def load_artifacts():
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    with open(MODEL_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(MODEL_DIR / "class_names.json") as f:
        class_names = json.load(f)
    # Each .pkl is a full sklearn Pipeline: ColumnTransformer (OneHotEncoder + StandardScaler) -> classifier
    models = {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}

    best_model_path = MODEL_DIR / "best_model.json"
    if best_model_path.exists():
        with open(best_model_path) as f:
            best_model_name = json.load(f)["best_model"]
    else:
        best_model_name = list(models.keys())[0]

    return label_encoder, feature_names, class_names, models, best_model_name


label_encoder, feature_names, class_names, models, best_model_name = load_artifacts()

st.title("🎓 Student Dropout & Academic Success — Model Comparison App")
st.markdown(
    "This app demonstrates **5 classification models** trained on the UCI "
    "**Predict Students' Dropout and Academic Success** dataset. It predicts "
    "whether a student is likely to **Dropout**, remain **Enrolled**, or **Graduate**. "
    "Upload the provided `test_data.csv` (or any CSV with the same columns) to see "
    "predictions and evaluation metrics for the model of your choice."
)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
model_names = list(models.keys())
default_index = model_names.index(best_model_name) if best_model_name in model_names else 0
selected_model_name = st.sidebar.selectbox(
    "Select a model", model_names, index=default_index,
    help=f"Defaults to '{best_model_name}', the top-Accuracy model on the validation split.",
)
selected_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Expected columns: 36 student demographic/academic features plus a "
    "`Target` column with values: Dropout, Enrolled, Graduate."
)

if uploaded_file is None:
    st.info("Upload `test_data.csv` from the sidebar to get started.")
    st.stop()

# ---------------- Load and validate uploaded data ----------------
try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

if "Target" not in data.columns:
    st.error("Uploaded CSV must include a 'Target' column (ground-truth label).")
    st.stop()

missing_cols = [c for c in feature_names if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
    st.stop()

X = data[feature_names]
y_true_raw = data["Target"]
y_true = label_encoder.transform(y_true_raw)

st.subheader("Uploaded Data Preview")
st.dataframe(data.head(10), use_container_width=True)
st.caption(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

st.markdown("**Class distribution in uploaded data**")
st.bar_chart(y_true_raw.value_counts())

# ---------------- Run selected model ----------------
y_pred = selected_model.predict(X)
y_proba = selected_model.predict_proba(X)

st.subheader(f"📊 Evaluation Metrics — {selected_model_name}")

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC Score (OvR, macro)": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
    "Precision (weighted)": precision_score(y_true, y_pred, average="weighted"),
    "Recall (weighted)": recall_score(y_true, y_pred, average="weighted"),
    "F1 Score (weighted)": f1_score(y_true, y_pred, average="weighted"),
    "MCC Score": matthews_corrcoef(y_true, y_pred),
}

cols = st.columns(len(metrics))
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

# ---------------- Confusion matrix + classification report ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    st.markdown("**Classification Report**")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

# ---------------- Compare all models ----------------
st.subheader("🔁 Compare All Models on This Uploaded Data")

comparison_rows = []
for name, m in models.items():
    preds = m.predict(X)
    probs = m.predict_proba(X)
    comparison_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, preds),
        "AUC (OvR macro)": roc_auc_score(y_true, probs, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true, preds, average="weighted"),
        "Recall": recall_score(y_true, preds, average="weighted"),
        "F1": f1_score(y_true, preds, average="weighted"),
        "MCC": matthews_corrcoef(y_true, preds),
    })

comparison_df = pd.DataFrame(comparison_rows).round(4)
st.dataframe(comparison_df, use_container_width=True)
st.bar_chart(comparison_df.set_index("Model")[["Accuracy", "AUC (OvR macro)", "F1"]])

st.markdown("---")
st.caption("Built for ML Assignment 2 — Streamlit Community Cloud deployment demo.")

