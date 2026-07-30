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
"""
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent           # .../project/model
PROJECT_ROOT = SCRIPT_DIR.parent                        # .../project
MODEL_DIR = SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_ONLY_PATH = DATA_DIR / "train_only.csv"

RANDOM_STATE = 42


def main() -> None:
    # ---- Load training data (excludes test_data.csv) ----
    df = pd.read_csv(TRAIN_ONLY_PATH)
    X = df.drop("Target", axis=1)
    y_raw = df["Target"]

    categorical_columns = [
        "Marital status",
        "Application mode",
        "Application order",
        "Course",
        'Daytime/evening attendance',
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
        "Target"
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

    print("\nClass balance (training portion):")
    print(y_raw.value_counts())

    # TODO build the ColumnTransformer preprocessing pipeline and start training the models.


if __name__ == "__main__":
    main()