import json
import joblib
import numpy as np
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

# Portable path: relative to this file's location, not the OS or working directory
MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

# Course codes -> readable names, per the official UCI variable documentation
# (https://archive.ics.uci.edu/dataset/697). Used only for the single-student
# form's dropdown; the model itself still sees the raw numeric code.
COURSE_NAMES = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)",
}


@st.cache_resource
def load_artifacts():
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    with open(MODEL_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(MODEL_DIR / "class_names.json") as f:
        class_names = json.load(f)
    with open(MODEL_DIR / "categorical_columns.json") as f:
        categorical_columns = json.load(f)
    with open(MODEL_DIR / "numerical_columns.json") as f:
        numerical_columns = json.load(f)
    with open(MODEL_DIR / "default_values.json") as f:
        default_values = json.load(f)
    # Each .pkl is a full sklearn Pipeline: ColumnTransformer (OneHotEncoder + StandardScaler) -> classifier
    models = {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}

    # Best model (by validation Accuracy) is saved by train_models.py; fall back
    # to the first model in the dict if the file is missing for any reason.
    best_model_path = MODEL_DIR / "best_model.json"
    if best_model_path.exists():
        with open(best_model_path) as f:
            best_model_name = json.load(f)["best_model"]
    else:
        best_model_name = list(models.keys())[0]

    return (
        label_encoder, feature_names, class_names, categorical_columns,
        numerical_columns, default_values, models, best_model_name,
    )


(
    label_encoder, feature_names, class_names, categorical_columns,
    numerical_columns, default_values, models, best_model_name,
) = load_artifacts()

st.title("Student Dropout & Academic Success — Model Comparison App")
st.markdown(
    "This app demonstrates **5 classification models** trained on the UCI "
    "**Predict Students' Dropout and Academic Success** dataset. It predicts "
    "whether a student is likely to **Dropout**, remain **Enrolled**, or **Graduate**.\n\n"
    "Each model is a complete scikit-learn `Pipeline`: categorical features "
    "(course, application mode, parents' occupation/qualification, etc.) are "
    "one-hot encoded, numerical features (grades, curricular units, macro-economic "
    "indicators) are standardized — so raw student data is handled automatically."
)

# ---------------- Model selector (shared across both tabs) ----------------
st.sidebar.header("Controls")
model_names = list(models.keys())
default_index = model_names.index(best_model_name) if best_model_name in model_names else 0
selected_model_name = st.sidebar.selectbox(
    "Select a model",
    model_names,
    index=default_index,
    help=f"Defaults to '{best_model_name}', the top-Accuracy model on the validation split.",
)
selected_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Bulk Evaluation** tab: upload a CSV with 36 feature columns + `Target` "
    "to evaluate a model against known outcomes.\n\n"
    "**Single Student Prediction** tab: enter a handful of key details for one "
    "student and get a live prediction — the rest of the 36 features default "
    "to the median/mode values from the training data."
)

tab_bulk, tab_single = st.tabs(["Bulk Evaluation (CSV Upload)", "Single Student Prediction"])

# =====================================================================
# TAB 1: Bulk evaluation against a CSV with known outcomes
# =====================================================================
with tab_bulk:
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"], key="bulk_uploader")

    if uploaded_file is None:
        st.info("👆 Upload `test_data.csv` to get started.")
    else:
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

        st.subheader("📄 Uploaded Data Preview")
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

        # ---------------- Domain-specific interpretation ----------------
        # For a student-support use case, not all misclassifications carry equal
        # weight: predicting "Graduate" for a student who actually drops out is
        # the costliest error, since it's the exact student an early-intervention
        # program most needs to flag. We surface that count explicitly rather
        # than leaving it buried in the confusion matrix.
        if "Dropout" in class_names and "Graduate" in class_names:
            dropout_idx = class_names.index("Dropout")
            graduate_idx = class_names.index("Graduate")
            missed_dropouts = cm[dropout_idx, graduate_idx]
            total_actual_dropouts = cm[dropout_idx, :].sum()
            if total_actual_dropouts > 0:
                miss_rate = missed_dropouts / total_actual_dropouts
                st.warning(
                    f"⚠️ **At-risk-student blind spot:** of {int(total_actual_dropouts)} students who "
                    f"actually dropped out, **{selected_model_name}** predicted **{int(missed_dropouts)} "
                    f"({miss_rate:.1%})** of them as 'Graduate' — the highest-cost error for an early-"
                    f"intervention program, since these are exactly the students who wouldn't get flagged "
                    f"for support."
                )

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

# =====================================================================
# TAB 2: Single-student prediction (manual entry form)
# =====================================================================
with tab_single:
    st.markdown(
        "Fill in the details that matter most for this student — everything "
        "else defaults to the **median/mode value from the training data** "
        "(a 'typical' student), so you don't need to specify all 36 fields."
    )

    with st.form("single_student_form"):
        st.markdown("**Academic path**")
        f_course = st.selectbox(
            "Course", options=list(COURSE_NAMES.keys()),
            format_func=lambda code: f"{COURSE_NAMES[code]} ({code})",
            index=list(COURSE_NAMES.keys()).index(9500) if 9500 in COURSE_NAMES else 0,
        )
        f_daytime = st.radio("Attendance", options=[1, 0], format_func=lambda v: "Daytime" if v == 1 else "Evening", horizontal=True)
        f_age = st.slider("Age at enrollment", min_value=17, max_value=70, value=int(default_values.get("Age at enrollment", 20)))

        st.markdown("**Financial status**")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            f_scholarship = st.radio("Scholarship holder?", options=[1, 0], format_func=lambda v: "Yes" if v == 1 else "No")
        with col_f2:
            f_tuition_ok = st.radio("Tuition fees up to date?", options=[1, 0], format_func=lambda v: "Yes" if v == 1 else "No")
        with col_f3:
            f_debtor = st.radio("Debtor?", options=[1, 0], format_func=lambda v: "Yes" if v == 1 else "No")

        st.markdown("**Academic performance so far**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            f_units_1st_approved = st.slider("Curricular units approved — 1st sem", 0, 26, int(default_values.get("Curricular units 1st sem (approved)", 5)))
            f_grade_1st = st.slider("Average grade — 1st sem (0-20 scale)", 0.0, 20.0, float(default_values.get("Curricular units 1st sem (grade)", 12.0)))
        with col_p2:
            f_units_2nd_approved = st.slider("Curricular units approved — 2nd sem", 0, 26, int(default_values.get("Curricular units 2nd sem (approved)", 5)))
            f_grade_2nd = st.slider("Average grade — 2nd sem (0-20 scale)", 0.0, 20.0, float(default_values.get("Curricular units 2nd sem (grade)", 12.0)))

        submitted = st.form_submit_button("Predict Outcome")

    if submitted:
        # Start from the "typical student" defaults, then overlay the fields
        # the user actually customized above.
        student = dict(default_values)
        student.update({
            "Course": f_course,
            "Daytime/evening attendance": f_daytime,
            "Age at enrollment": f_age,
            "Scholarship holder": f_scholarship,
            "Tuition fees up to date": f_tuition_ok,
            "Debtor": f_debtor,
            "Curricular units 1st sem (approved)": f_units_1st_approved,
            "Curricular units 1st sem (grade)": f_grade_1st,
            "Curricular units 2nd sem (approved)": f_units_2nd_approved,
            "Curricular units 2nd sem (grade)": f_grade_2nd,
        })
        student_df = pd.DataFrame([student])[feature_names]

        pred_encoded = selected_model.predict(student_df)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        pred_proba = selected_model.predict_proba(student_df)[0]

        st.markdown("---")
        st.subheader(f"Prediction — {selected_model_name}")

        outcome_style = {"Dropout": "🔴", "Enrolled": "🟡", "Graduate": "🟢"}
        st.markdown(f"### {outcome_style.get(pred_label, '')} Predicted outcome: **{pred_label}**")

        proba_df = pd.DataFrame({"Class": class_names, "Probability": pred_proba}).set_index("Class")
        st.bar_chart(proba_df)

        if pred_label == "Dropout":
            st.warning(
                "This profile resembles historical dropout cases. In a real "
                "advising workflow, this is the kind of prediction that would "
                "trigger early outreach — e.g. a check-in about workload, "
                "finances, or academic support resources."
            )

st.markdown("---")
st.caption(
    "Built for ML Assignment 2 — Streamlit Community Cloud deployment demo. "
    "Dataset: UCI Predict Students' Dropout and Academic Success."
)