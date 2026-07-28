# ML Assignment 2

Machine Learning Classification Models

Author: RAVIVARMAN S
BITS ID: 2025DA04030

## a. Problem Statement
The rapid growth of Internet of Things (IoT) devices has increased the risk of cyberattacks targeting interconnected systems and smart environments. Early and accurate detection of malicious network traffic is essential to ensure the security and reliability of IoT networks. The goal of this project is to develop and compare multiple machine learning classification models capable of identifying different types of network attacks from normal IoT traffic using the RT-IoT2022 dataset. The trained models will be deployed through an interactive Streamlit web application, allowing users to classify network traffic based on input features and evaluate the performance of different machine learning algorithms.

## b. Dataset Description
- Dataset Name: RT-IoT2022 Dataset
- Sourse: UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/942/rt-iot2022
- No. of Instances: 123,117
- No. of Features:
    83 input features (network traffic and flow characteristics)
    1 target variable (Attack_type) containing different IoT traffic and attack categories
- Classification Type: Multiclass Classification
- Number of Target Classes: 12 (in the downloaded dataset version)
- Missing Values: None
- Data Type: Mixed (Numerical and Categorical)

### EDA Summary
- The RT-IoT2022 dataset contains 123,117 records with 83 input features and one target variable (Attack_type).
- The dataset contains no missing values and no duplicate records (assuming your checks confirmed this).
- There are 12 target classes in the current dataset version.
- The target variable is highly imbalanced, with DOS_SYN_Hping accounting for 76.89% of all records, while NMAP_FIN_SCAN is the least represented class (0.02%).
- Numerical features exhibit varying distributions, with several showing skewness and outliers.
- Correlation analysis identified relationships among some numerical features, but no severe multicollinearity affecting the entire dataset.
- Based on the EDA, the dataset is clean and suitable for multiclass classification after appropriate preprocessing, including encoding categorical variables and using a stratified train-test split to preserve class proportions.

## c. Github Repository Link
'https://github.com/sravivarman/ml_assignment_2_2025da04030'

## d. Models used
Make a Comparison Table with the evaluation metrics calculated for all the 6 models as below:
