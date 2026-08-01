import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Students' Dropout and Academic Success Classifier", page_icon="🎓", layout="wide")

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
    return label_encoder, feature_names, class_names, models


label_encoder, feature_names, class_names, models = load_artifacts()

st.title("Student Dropout and Academic Success — Model Comparison App")
st.markdown(
    "This app demonstrates **5 classification models** trained on the UCI "
    "**Predict Students' Dropout and Academic Success** dataset. It predicts "
    "whether a student is likely to **Dropout**, remain **Enrolled**, or **Graduate**."
)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
model_names = list(models.keys())
selected_model_name = st.sidebar.selectbox("Select a model", model_names)
selected_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Expected columns: 36 student demographic/academic features plus a "
    "`Target` column with values: Dropout, Enrolled, Graduate."
)

# ---------------- Load and preview uploaded data ----------------
if uploaded_file is None:
    st.info("Upload `test_data.csv` from the sidebar to get started.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

if "Target" not in data.columns:
    st.error("Uploaded CSV must include a 'Target' column.")
    st.stop()

missing_cols = [c for c in feature_names if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
    st.stop()

st.subheader("Uploaded Data Preview")
st.dataframe(data.head(10), width="stretch")
st.caption(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

st.markdown("**Class distribution in uploaded data**")
st.bar_chart(data["Target"].value_counts())

# TODO: run the selected model, compute and display evaluation metrics, confusion matrix, classification report, and an all-model comparison table.
