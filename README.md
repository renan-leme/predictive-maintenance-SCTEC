# Predictive Maintenance using Machine Learning

## Overview
Predictive maintenance has become one of the most important applications of Artificial
Intelligence in Industry 4.0. By analyzing historical sensor data, machine learning 
models can identify patterns associated with equipment failures before they occur, 
reducing unexpected downtime and maintenance costs.
This project implements a complete **Machine Learning pipeline** to predict machine
failures based on operational sensor data. The workflow includes data exploration,
preprocessing, feature engineering, model training, hyperparameter tuning, and
performance evaluation.

## Project Objectives
-   Perform Exploratory Data Analysis (EDA)
-   Clean and preprocess industrial sensor data
-   Engineer new predictive features
-   Handle class imbalance using resampling techniques
-   Compare machine learning algorithms
-   Analyze overfitting
-   Select the best predictive model

## Dataset
The dataset contains historical information collected from industrial equipment, including:
|        Feature          |          Description         |
|-------------------------|------------------------------|
| `product_id`            | Product identifier           |
| `machine_type`          | Machine category             |
| `air_temperature_k`     | Air temperature (Kelvin)     |
| `process_temperature_k` | Process temperature (Kelvin) |
| `rotational_speed_rpm`  | Rotational speed             |
| `torque_nm`             | Motor torque                 |
| `tool_wear_min`         | Tool wear (minutes)          |
| `machine_failure`       | Target variable              |
Additional failure-type columns are included in the original dataset but were intentionally
excluded from model training to prevent data leakage.

# Machine Learning Pipeline
1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Missing Value Imputation
5. Outlier Analysis
6. Feature Engineering
7. Data Encoding
8. Train/Test Split
9. Data Balancing (SMOTE)
10. Feature Scaling
11. Model Training
12. Hyperparameter Tuning
13. Performance Evaluation
14. Model Selection

# Predictive Maintenance Classification Pipeline
This notebook implements a leakage-aware Machine Learning pipeline for predictive maintenance.

```text
Data cleaning
        ↓
Train/Test Split
        ↓
Imputer.fit(X_train)
        ↓
Imputer.transform(train/test)
        ↓
Feature Engineering
        ↓
OneHotEncoder.fit(X_train)
        ↓
OneHotEncoder.transform(train/test)
        ├──────────────► Decision Tree → SMOTE → Decision Tree
        ↓
StandardScaler.fit(X_train)
        ↓
StandardScaler.transform(train/test)
        ↓
SMOTE
        ↓
KNN

## Feature Engineering
Additional engineered features:
-   **Power** = Rotational Speed × Torque -> Represents the mechanical power demand of the machine.
-   **Temperature Difference** = Process Temperature − Air Temperature -> Represents the thermal operating condition of the equipment.
The categorical feature **machine_type** was transformed using **One-Hot
Encoding**, creating binary variables suitable for machine learning
algorithms.

## Machine Learning Models
The following models were evaluated:
-   K-Nearest Neighbors (KNN)
-   Decision Tree
Hyperparameter tuning was performed for both models in order to reduce overfitting and improve generalization.

## Technologies Used
-   Python 3
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-Learn
-   Imbalanced-Learn (SMOTE)
-   Jupyter Notebook

## Project Structure
```text
predictive-maintenance/
│
├── predictive_maintenance.csv
├── predictive_maintenance.ipynb
├── requirements.txt
├── README.md
└── outputs/
```

## Github Structure
```text
main/
│
└── develop
     ├── feature/eda
     ├── feature/data-prep
     ├── feature/feature-engineering
     ├── feature/modeling
     ├── feature/evaluation
     └── docs/readme
```

## Installation
-   Clone the repository: git clone https://github.com/renan-leme/predictive-maintenance-SCTEC.git
-   Create a virtual environment: python -m venv .venv
-   Activate the environment.
-   Windows: .venv\Scripts\activate
-   Linux / macOS: source .venv/bin/activate
-   Install the dependencies: pip install -r requirements.txt

## Running the Project
-   Launch Jupyter Notebook: jupyter notebook
-   Open: Predictive_Maintenance.ipynb
-   Execute all cells sequentially.

## Results
The project compares different classification models by evaluating:
-   Accuracy
-   Confusion Matrix
-   Classification Report
-   Overfitting Analysis
The final model is selected according to its performance on the test dataset.

## Future Improvements
Possible enhancements include:
-   Random Forest
-   XGBoost
-   LightGBM
-   Support Vector Machine (SVM)
-   Neural Networks
-   Cross Validation
-   SHAP Explainability
-   ROC Curve
-   Precision-Recall Curve
-   Model Deployment using Flask or FastAPI

## Demonstration Video
Video link: https://youtu.be/JF6jW9guI2Q


## Author
**Renan de Brito Leme**
Electrical Engineer \| Industrial Automation \| Machine Learning \| Data Science