# ML Assignment 2

Machine Learning Classification Models

Author: RAVIVARMAN S

BITS ID: 2025DA04030

## a. Problem Statement

The ability to predict students' academic outcomes is essential for educational institutions to improve student retention, enhance academic performance, and provide timely support to students who are at risk of dropping out. Early prediction enables institutions to implement appropriate intervention strategies, thereby improving graduation rates and reducing student attrition.

The objective of this project is to develop and compare multiple machine learning classification models capable of predicting whether a student will **Dropout**, remain **Enrolled**, or **Graduate** based on demographic, socioeconomic, academic, and institutional factors. The trained models will be deployed through an interactive **Streamlit** web application, allowing users to predict student academic outcomes and compare the performance of different machine learning algorithms.

---

## b. Dataset Description

- **Dataset Name:** Predict Students' Dropout and Academic Success
- **Source:** UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **Number of Instances:** 4,424
- **Number of Features:**
  - **36 input features** consisting of demographic, academic, socioeconomic, and macroeconomic attributes.
  - **1 target variable** (`Target`) representing the student's academic outcome.
- **Classification Type:** Multiclass Classification
- **Target Classes:**
  - Dropout
  - Enrolled
  - Graduate
- **Missing Values:** None
- **Data Type:** Mixed (Numerical and Encoded Categorical Features)

### EDA Summary

- The dataset contains **4,424 records**, **36 input features**, and **one target variable (`Target`)**.
- The dataset contains **no missing values** and **no duplicate records**, indicating that the data is clean and suitable for machine learning analysis.
- The target variable consists of **three classes**: **Graduate**, **Dropout**, and **Enrolled**.
- The class distribution is reasonably balanced:
  - **Graduate:** 2,209 records (49.93%)
  - **Dropout:** 1,421 records (32.12%)
  - **Enrolled:** 794 records (17.95%)
- The dataset contains a combination of continuous numerical variables (such as grades, age, GDP, unemployment rate, and inflation rate) and categorical variables encoded as integers (such as gender, marital status, course, scholarship holder, and debtor status).
- Numerical features exhibit varying distributions, with several variables showing skewness and the presence of outliers.
- Correlation analysis indicates relationships among academic performance variables, while no severe multicollinearity is observed across the numerical features.
- Based on the exploratory data analysis, the dataset is clean, well-structured, and suitable for multiclass classification. Appropriate preprocessing, including handling encoded categorical variables, feature scaling for distance-based algorithms, and a stratified train-test split, will be performed before model development.

## c. Github Repository Link
'https://github.com/sravivarman/ml_assignment_2_2025da04030'

## d. Models used
All 5 models below were trained on the same 80/20 train/validation split of
the training portion of the dataset, using the shared preprocessing pipeline
described above and a label-encoded target (Dropout=0, Enrolled=1, Graduate=2).
 
Model configuration notes:
- **Logistic Regression**: `solver="lbfgs"`, `max_iter=5000`, `random_state=42`.  
- **Decision Tree**: `random_state=42`; all other hyperparameters (e.g. `max_depth`,
  `min_samples_split`, `criterion="gini"`) left at scikit-learn defaults, so
  the tree grows until its leaves are pure.
- **kNN**: `n_neighbors=5`, `weights="distance"` — closer neighbors get more
  influence, which improved results slightly over uniform weighting.
- **Naive Bayes**: `GaussianNB()` with default hyperparameters (no tuning applied).
- **Random Forest**: `n_estimators=300`, `random_state=42`, `n_jobs=-1` (uses
  all CPU cores).

### Comparison Table
*(sorted by Accuracy, best first)*
 
| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.7782   | 0.8928 | 0.7674    | 0.7782 | 0.7701 | 0.6331 |
| Random Forest       | 0.7726   | 0.8759 | 0.7519    | 0.7726 | 0.7533 | 0.6219 |
| kNN                 | 0.7316   | 0.8229 | 0.7181    | 0.7316 | 0.7143 | 0.5515 |
| Decision Tree       | 0.6949   | 0.7358 | 0.6925    | 0.6949 | 0.6929 | 0.5049 |
| Naive Bayes         | 0.2556   | 0.6883 | 0.2966    | 0.2556 | 0.1846 | 0.1284 |
 
*(AUC = One-vs-Rest, macro-averaged; Precision/Recall/F1 = weighted average across the 3 classes)*
 
### Observations
 
| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Best overall performer. The linear decision boundary generalizes well once categorical codes are properly one-hot encoded instead of being treated as ordinal numbers. |
| Random Forest        | Second-best, close behind Logistic Regression. Averaging many trees over the one-hot + scaled feature set captures non-linear interactions (e.g. how curricular performance combines with socio-economic factors) while staying robust to the high-dimensional categorical space. |
| kNN                  | Distance-weighted voting (`weights="distance"`) gave a small but real improvement over uniform weighting. Still capped by the high-dimensional, mostly-binary one-hot feature space, which dilutes the meaning of Euclidean distance. |
| Decision Tree        | A single tree can now split cleanly on individual one-hot category flags (e.g. "Course = Nursing?") instead of arbitrary numeric thresholds on category codes, but it still overfits to the training split more than the ensemble model. |
| Naive Bayes          | **Collapsed sharply after one-hot encoding** (25.6% accuracy). This is a well-known Naive Bayes failure mode: `GaussianNB` assumes every feature is continuous and normally distributed, but one-hot encoding expanded the feature space to 232 mostly-binary (0/1) columns. Many of these have near-zero variance for rare categories, which distorts `GaussianNB`'s per-feature Gaussian likelihood estimates and causes it to over-predict a single class. A `CategoricalNB`/`BernoulliNB` treatment of the categorical block would likely fix this — it demonstrates why the choice of Naive Bayes variant must match the data type. |
| **Overall Winner**   | **Logistic Regression** — highest Accuracy, AUC, Precision, Recall, F1, and MCC among all 5 models on this dataset. |

## How to Run Locally
```bash
pip install -r requirements.txt
python data/prepare_data.py       # generates the CSVs (already included in repo)
python model/train_models.py      # trains models (already included in repo)
streamlit run streamlit_app.py
```

## Deployment on Streamlit Community Cloud

