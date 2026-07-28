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
Comparison Table with the evaluation metrics calculated for all the 6 models as below:
