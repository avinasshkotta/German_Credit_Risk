# German Credit Risk — Accuracy vs Recall

## 1. Project overview

This project uses the **Statlog (German Credit Data)** from the UCI Machine Learning Repository to demonstrate an important credit-risk modeling principle:

> The right evaluation metric is determined by the business cost of errors, not by the metric that looks best in a report.

The project trains two Logistic Regression model-selection pipelines:

1. **Accuracy-optimized model** — hyperparameters selected using cross-validated accuracy.
2. **Recall-optimized model** — hyperparameters selected using cross-validated recall for the **Bad** credit-risk class.

The final test-set comparison measures how many genuinely Bad applicants are missed by each model. The difference is translated into an estimated financial impact using a configurable average default-loss assumption.

## 2. Dataset

Source: UCI Machine Learning Repository — Statlog (German Credit Data)

Official source:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The dataset contains:

- 1,000 applicants
- 20 attributes
- Categorical and integer features
- Binary credit-risk target: Good / Bad
- No missing values in the original UCI dataset

UCI also documents a cost matrix where:
- Actual Good → Predicted Bad = cost 1
- Actual Bad → Predicted Good = cost 5

The notebook downloads the data directly using:

```python
from ucimlrepo import fetch_ucirepo
credit = fetch_ucirepo(id=144)
```

Therefore, a local dataset file is not required.

## 3. Business problem

A credit-risk team wants to identify applicants who are likely to represent bad credit risk.

Two types of mistakes are possible:

- **False positive:** a Good applicant is classified as Bad.
- **False negative:** a Bad applicant is classified as Good.

For this case study, missing a Bad applicant is treated as especially important because the applicant may be approved despite having elevated risk.

## 4. Project objective

Build and compare two predictive approaches:

### Model A — Accuracy optimized

Select model hyperparameters using cross-validated **accuracy**.

### Model B — Recall optimized

Select model hyperparameters using cross-validated **recall for the Bad class**.

Then:

- compare accuracy
- compare precision for Bad
- compare recall for Bad
- compare F1 for Bad
- compare ROC-AUC
- compare confusion matrices
- count missed Bad applicants
- calculate the additional Bad applicants caught by the recall-optimized approach
- estimate avoided default loss

## 5. Analytics included

### Descriptive analytics

The notebook performs:

- dataset shape and structure
- data types
- missing-value check
- Good/Bad class distribution
- percentages of Good and Bad applicants
- descriptive statistics
- numeric feature distributions

### Diagnostic analytics

The notebook examines:

- average numeric characteristics by Good/Bad class
- correlations among numeric variables
- class differences
- confusion matrices after model training

### Predictive analytics

The notebook:

- creates a stratified train/test split
- preprocesses categorical and numeric features
- uses Logistic Regression
- uses 5-fold cross-validation
- optimizes one model for accuracy
- optimizes another model for Bad-class recall
- evaluates both on an untouched test set

### Visualization

The notebook generates:

- credit-risk class distribution
- numeric feature histograms
- correlation heatmap
- confusion matrices
- model metric comparison chart

## 6. Business impact calculation

The key business calculation is:

```text
Additional Bad applicants caught
    = Bad applicants missed by accuracy model
      - Bad applicants missed by recall model
```

Then:

```text
Estimated avoided default loss
    = Additional Bad applicants caught
      × Average default loss per default
```

The notebook uses:

```python
AVERAGE_DEFAULT_LOSS = 5000.0
```

as an example placeholder.

**Replace this value with the organization's actual average loss per default before presenting financial results.**

## 7. Project structure

```text
german_credit_risk_case_study/
│
├── German_Credit_Risk_Accuracy_vs_Recall.ipynb
└── README.md
```

## 8. How to run

### Option A — Jupyter Notebook

Open:

```text
German_Credit_Risk_Accuracy_vs_Recall.ipynb
```

Run the cells from top to bottom.

### Option B — Google Colab

Upload the `.ipynb` file to Google Colab and run the cells.

The notebook installs `ucimlrepo` and retrieves the dataset from UCI.

### Required packages

```bash
pip install ucimlrepo pandas numpy scikit-learn matplotlib seaborn
```

## 9. Expected workflow

```text
UCI German Credit Dataset
          ↓
Data Loading
          ↓
Descriptive Analytics
          ↓
Diagnostic Analytics
          ↓
Train/Test Split
          ↓
Preprocessing
          ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Accuracy Optimization   Recall Optimization
 │                       │
 ▼                       ▼
Accuracy Model          Recall Model
 └──────────────┬────────┘
                ↓
        Test-set comparison
                ↓
       Confusion matrices
                ↓
       Business impact
                ↓
      Estimated default loss
```

## 10. Important interpretation

Accuracy should not automatically be treated as the most appropriate metric.

For this business problem, the important question is:

> How many genuinely Bad applicants are being missed?

Recall for the Bad class directly addresses that question.

However, increasing recall can also increase the number of Good applicants incorrectly classified as Bad. Therefore, the final production decision should consider the organization's actual costs, approval policy, risk appetite, fairness requirements, regulatory requirements, and human-review process.

## 11. Limitations

- The dataset contains only 1,000 observations.
- It is a historical dataset and may not represent current lending populations.
- Model results can change with different data splits.
- The average default loss is a business assumption unless replaced with verified company data.
- This is an educational/business-practice example, not a production lending-decision system.
- Production deployment would require additional work around calibration, explainability, monitoring, fairness, governance, regulatory compliance, and human oversight.

## 12. Citation

Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.

DOI: https://doi.org/10.24432/C5NC77

UCI dataset page:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
