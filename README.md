# WorkforcePulse — Employee Attrition Analytics & Prediction

WorkforcePulse is a Data Analytics and Machine Learning project that analyzes employee attrition patterns and builds predictive models to identify factors associated with employee turnover.

## Problem Statement

Employee attrition can affect productivity, hiring costs, team stability, and organizational performance.

This project uses the IBM HR Analytics Employee Attrition & Performance dataset to:

* Analyze employee demographics and workplace characteristics
* Identify patterns associated with employee attrition
* Explore relationships between overtime, job satisfaction, income, tenure, and attrition
* Engineer additional features related to employee tenure and overtime
* Build machine learning models for attrition prediction
* Evaluate models using multiple classification metrics

## Dataset

* **Records:** 1,470 employees
* **Original features:** 35
* **Target:** `Attrition`
* **Target classes:**

  * `0` = No Attrition
  * `1` = Attrition
* **Source:** IBM HR Analytics Employee Attrition & Performance dataset

Dataset source: Kaggle — IBM HR Analytics Employee Attrition & Performance

The original raw dataset is preserved unchanged in the `data/raw/` directory.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Joblib

## Project Workflow

```text
Raw Dataset
    ↓
Data Quality Analysis
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Data Preprocessing
    ↓
Machine Learning
    ↓
Model Evaluation
    ↓
Insights & Conclusion
```

## Data Cleaning

The preprocessing stage includes:

* Checking missing values and duplicate records
* Removing identifier and constant columns
* Converting the `Attrition` target into binary values
* Cleaning categorical values
* Preparing numerical and categorical variables for machine learning

## Exploratory Data Analysis

The project explores:

* Employee attrition distribution
* Age distribution
* Monthly income
* Years at company
* Overtime and attrition
* Job role and attrition
* Job satisfaction and attrition
* Department and attrition

## Feature Engineering

Additional features were created to support analysis and prediction:

* `OverTime_Flag`
* `Gender_Flag`
* `PromotionGap`
* `RoleTenureRatio`
* `ManagerTenureRatio`
* `IncomePerWorkingYear`

## Machine Learning Models

Three classification models were trained:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The dataset was split into training and testing sets using an 80:20 stratified split.

Categorical variables were one-hot encoded, while numerical variables were imputed and standardized through a preprocessing pipeline.

## Model Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.7687 |    0.3765 | 0.6809 |   0.4848 |  0.8145 |
| Random Forest       |   0.8265 |    0.4412 | 0.3191 |   0.3704 |  0.7707 |
| Gradient Boosting   |   0.8469 |    0.5714 | 0.1702 |   0.2623 |  0.8082 |

The models show different performance characteristics across accuracy, precision, recall, F1 score, and ROC-AUC. Logistic Regression achieved the highest recall and F1 score among the evaluated models, while Gradient Boosting achieved the highest accuracy and precision.

## Key Analytical Findings

The exploratory analysis examines the association of factors such as:

* Overtime
* Job role
* Job satisfaction
* Income
* Employee tenure
* Work-life balance

with employee attrition in the dataset.

These findings represent patterns in the provided dataset and should not be interpreted as proof of causal relationships.

## Project Structure

```text
WorkforcePulse/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── notebooks/
│   └── ChayanshiJain_WorkforcePulse.ipynb
│
├── outputs/
│   └── model/
│
├── reports/
│
├── src/
│   ├── data_cleaning.py
│   ├── eda_analysis.py
│   ├── features.py
│   ├── generate_eda_plots.py
│   ├── model_evaluation.py
│   └── train_model.py
│
├── tests/
│
├── README.md
└── requirements.txt
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/chayanshijain23/WorkforcePulse.git
cd WorkforcePulse
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/ChayanshiJain_WorkforcePulse.ipynb
```

## Limitations

* The dataset contains 1,470 records and is synthetic in nature.
* The attrition class is smaller than the non-attrition class, creating class imbalance.
* The dataset represents a fixed set of employee records and may not generalize to real organizations.
* Model predictions should not be used as the sole basis for real-world employee or HR decisions.

## Conclusion

WorkforcePulse demonstrates an end-to-end data analytics and machine learning workflow for employee attrition analysis, covering data cleaning, exploratory analysis, feature engineering, preprocessing, model training, and evaluation.

The project is intended for academic and analytical purposes and demonstrates how machine learning techniques can be applied to study patterns associated with employee attrition.

## Author

**Chayanshi Jain**
B.Tech — Computer Science & Engineering (Data Science)
ABES Engineering College, Ghaziabad

## GitHub Repository

https://github.com/chayanshijain23/WorkforcePulse
