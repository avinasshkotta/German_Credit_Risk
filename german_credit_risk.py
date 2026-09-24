# ============================================================
# GERMAN CREDIT RISK — ACCURACY VS RECALL
# ============================================================
# Business Case:
# Compare an accuracy-optimized model with a recall-optimized
# model for detecting BAD credit-risk applicants.
#
# Dataset:
# UCI Statlog German Credit Data
#
# 1 = Good Credit
# 2 = Bad Credit
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED LIBRARIES
# ============================================================

# Run this cell only if packages are not installed.
# Uncomment if required.

# !pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    make_scorer
)


# ============================================================
# 3. SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.30

# UCI target coding:
# 1 = Good
# 2 = Bad

GOOD_CLASS = 1
BAD_CLASS = 2

# Business assumption
# Change this to the actual average loss supplied by the business.
AVERAGE_DEFAULT_LOSS = 5000.0


# ============================================================
# 4. LOAD DATA FROM UCI
# ============================================================

print("=" * 70)
print("LOADING GERMAN CREDIT DATASET")
print("=" * 70)

credit = fetch_ucirepo(id=144)

X = credit.data.features.copy()

y = credit.data.targets.copy()

# Convert target DataFrame into Series
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Convert target to integer
y = pd.to_numeric(y, errors="coerce").astype(int)

print("\nDataset loaded successfully.")

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nFirst 5 rows:")
display(X.head())

print("\nTarget values:")
print(y.value_counts().sort_index())


# ============================================================
# 5. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATA INFORMATION")
print("=" * 70)

print("\nDataset dimensions:")
print("Rows:", X.shape[0])
print("Columns:", X.shape[1])

print("\nData types:")
print(X.dtypes)

print("\nMissing values:")
print(X.isnull().sum())

print("\nTotal missing values:")
print(X.isnull().sum().sum())


# ============================================================
# 6. DESCRIPTIVE ANALYTICS
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE ANALYTICS")
print("=" * 70)


# ------------------------------------------------------------
# Target distribution
# ------------------------------------------------------------

target_summary = pd.DataFrame({
    "Count": y.value_counts().sort_index(),
    "Percentage": (
        y.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )
})

target_summary.index = target_summary.index.map({
    GOOD_CLASS: "Good",
    BAD_CLASS: "Bad"
})

print("\nCredit Risk Distribution:")
display(target_summary)


# ------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------

print("\nDescriptive Statistics:")

display(
    X.describe(include="all").T
)


# ============================================================
# 7. CREDIT RISK DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x=y.map({
        GOOD_CLASS: "Good",
        BAD_CLASS: "Bad"
    })
)

plt.title("German Credit Risk Distribution")

plt.xlabel("Credit Risk")

plt.ylabel("Number of Applicants")

plt.tight_layout()

plt.show()


# ============================================================
# 8. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ============================================================

numeric_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumeric columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 9. NUMERIC FEATURE DISTRIBUTIONS
# ============================================================

if len(numeric_columns) > 0:

    X[numeric_columns].hist(
        figsize=(15, 12),
        bins=20
    )

    plt.suptitle(
        "Numeric Feature Distributions",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 10. DIAGNOSTIC ANALYTICS
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC ANALYTICS")
print("=" * 70)


# Create diagnostic dataframe

diagnostic_df = X.copy()

diagnostic_df["CreditRisk"] = y.map({
    GOOD_CLASS: "Good",
    BAD_CLASS: "Bad"
})


# ------------------------------------------------------------
# Compare numeric variables by credit risk
# ------------------------------------------------------------

if len(numeric_columns) > 0:

    group_statistics = (
        diagnostic_df
        .groupby("CreditRisk")[numeric_columns]
        .mean()
        .T
    )

    print("\nAverage Numeric Values by Credit Risk:")
    display(group_statistics)


# ============================================================
# 11. CORRELATION MATRIX
# ============================================================

if len(numeric_columns) > 1:

    correlation_matrix = (
        diagnostic_df[numeric_columns]
        .corr()
    )

    plt.figure(
        figsize=(12, 9)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Correlation Matrix — Numeric Features"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


print("\nTraining target distribution:")

print(
    y_train
    .value_counts(normalize=True)
    .sort_index()
)


print("\nTesting target distribution:")

print(
    y_test
    .value_counts(normalize=True)
    .sort_index()
)


# ============================================================
# 13. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("CREATING PREPROCESSING PIPELINE")
print("=" * 70)


# Numeric preprocessing

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Combine preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# 14. BASE LOGISTIC REGRESSION PIPELINE
# ============================================================

base_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=3000,
                solver="liblinear",
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ============================================================
# 15. HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "model__class_weight": [
        None,
        "balanced"
    ]
}


# ============================================================
# 16. CREATE BAD-CLASS RECALL SCORER
# ============================================================

print("\n" + "=" * 70)
print("CREATING BAD-CLASS RECALL SCORER")
print("=" * 70)


# IMPORTANT:
#
# GridSearchCV does NOT accept:
#
# pos_label=BAD_CLASS
#
# directly.
#
# Instead, make_scorer() is used.

bad_recall_scorer = make_scorer(
    recall_score,
    pos_label=BAD_CLASS
)


# ============================================================
# 17. ACCURACY-OPTIMIZED GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ACCURACY-OPTIMIZED MODEL")
print("=" * 70)


accuracy_search = GridSearchCV(

    estimator=base_pipeline,

    param_grid=param_grid,

    scoring="accuracy",

    cv=5,

    n_jobs=-1,

    refit=True

)


accuracy_search.fit(
    X_train,
    y_train
)


# Get best accuracy model

accuracy_model = (
    accuracy_search.best_estimator_
)


print("\nBest Accuracy Parameters:")

print(
    accuracy_search.best_params_
)


print("\nBest Cross-Validation Accuracy:")

print(
    round(
        accuracy_search.best_score_,
        4
    )
)


# ============================================================
# 18. RECALL-OPTIMIZED GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BAD-CLASS RECALL-OPTIMIZED MODEL")
print("=" * 70)


recall_search = GridSearchCV(

    estimator=base_pipeline,

    param_grid=param_grid,

    scoring=bad_recall_scorer,

    cv=5,

    n_jobs=-1,

    refit=True

)


recall_search.fit(
    X_train,
    y_train
)


# Get best recall model

recall_model = (
    recall_search.best_estimator_
)


print("\nBest Recall Parameters:")

print(
    recall_search.best_params_
)


print("\nBest Cross-Validation Bad-Class Recall:")

print(
    round(
        recall_search.best_score_,
        4
    )
)


# ============================================================
# 19. PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST SET PREDICTIONS")
print("=" * 70)


accuracy_predictions = (
    accuracy_model.predict(X_test)
)


recall_predictions = (
    recall_model.predict(X_test)
)


# ============================================================
# 20. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    # Probability of Bad class

    class_index = list(
        model.classes_
    ).index(
        BAD_CLASS
    )

    probabilities = (
        model.predict_proba(X_test)
        [:, class_index]
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision_bad = precision_score(
        y_test,
        predictions,
        pos_label=BAD_CLASS,
        zero_division=0
    )

    recall_bad = recall_score(
        y_test,
        predictions,
        pos_label=BAD_CLASS,
        zero_division=0
    )

    f1_bad = f1_score(
        y_test,
        predictions,
        pos_label=BAD_CLASS,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        (y_test == BAD_CLASS).astype(int),
        probabilities
    )

    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision_Bad": precision_bad,

        "Recall_Bad": recall_bad,

        "F1_Bad": f1_bad,

        "ROC_AUC": roc_auc
    }


# ============================================================
# 21. EVALUATE BOTH MODELS
# ============================================================

accuracy_results = evaluate_model(

    "Accuracy-Optimized",

    accuracy_model,

    X_test,

    y_test
)


recall_results = evaluate_model(

    "Recall-Optimized",

    recall_model,

    X_test,

    y_test
)


results = pd.DataFrame(
    [
        accuracy_results,
        recall_results
    ]
)


print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

display(
    results.round(4)
)


# ============================================================
# 22. CLASSIFICATION REPORT — ACCURACY MODEL
# ============================================================

print("\n" + "=" * 70)
print("ACCURACY-OPTIMIZED MODEL")
print("=" * 70)


print(
    classification_report(
        y_test,
        accuracy_predictions,
        labels=[
            GOOD_CLASS,
            BAD_CLASS
        ],
        target_names=[
            "Good",
            "Bad"
        ],
        zero_division=0
    )
)


# ============================================================
# 23. CLASSIFICATION REPORT — RECALL MODEL
# ============================================================

print("\n" + "=" * 70)
print("RECALL-OPTIMIZED MODEL")
print("=" * 70)


print(
    classification_report(
        y_test,
        recall_predictions,
        labels=[
            GOOD_CLASS,
            BAD_CLASS
        ],
        target_names=[
            "Good",
            "Bad"
        ],
        zero_division=0
    )
)


# ============================================================
# 24. CONFUSION MATRICES
# ============================================================

cm_accuracy = confusion_matrix(

    y_test,

    accuracy_predictions,

    labels=[
        GOOD_CLASS,
        BAD_CLASS
    ]
)


cm_recall = confusion_matrix(

    y_test,

    recall_predictions,

    labels=[
        GOOD_CLASS,
        BAD_CLASS
    ]
)


# ============================================================
# 25. VISUALIZE CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)


sns.heatmap(

    cm_accuracy,

    annot=True,

    fmt="d",

    cmap="Blues",

    cbar=False,

    xticklabels=[
        "Good",
        "Bad"
    ],

    yticklabels=[
        "Good",
        "Bad"
    ],

    ax=axes[0]
)


axes[0].set_title(
    "Accuracy-Optimized Model"
)

axes[0].set_xlabel(
    "Predicted"
)

axes[0].set_ylabel(
    "Actual"
)


sns.heatmap(

    cm_recall,

    annot=True,

    fmt="d",

    cmap="Greens",

    cbar=False,

    xticklabels=[
        "Good",
        "Bad"
    ],

    yticklabels=[
        "Good",
        "Bad"
    ],

    ax=axes[1]
)


axes[1].set_title(
    "Recall-Optimized Model"
)

axes[1].set_xlabel(
    "Predicted"
)

axes[1].set_ylabel(
    "Actual"
)


plt.tight_layout()

plt.show()


# ============================================================
# 26. METRIC COMPARISON VISUALIZATION
# ============================================================

metrics_to_plot = [

    "Accuracy",

    "Precision_Bad",

    "Recall_Bad",

    "F1_Bad",

    "ROC_AUC"

]


metric_plot = (
    results
    .set_index("Model")
    [metrics_to_plot]
)


metric_plot.plot(
    kind="bar",
    figsize=(13, 6)
)


plt.title(
    "Accuracy vs Recall Optimized Models"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.show()


# ============================================================
# 27. ROC CURVE
# ============================================================

accuracy_probabilities = (

    accuracy_model
    .predict_proba(X_test)
    [:, list(
        accuracy_model.classes_
    ).index(BAD_CLASS)]

)


recall_probabilities = (

    recall_model
    .predict_proba(X_test)
    [:, list(
        recall_model.classes_
    ).index(BAD_CLASS)]

)


y_test_binary = (
    y_test == BAD_CLASS
).astype(int)


fpr_accuracy, tpr_accuracy, _ = roc_curve(
    y_test_binary,
    accuracy_probabilities
)


fpr_recall, tpr_recall, _ = roc_curve(
    y_test_binary,
    recall_probabilities
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    fpr_accuracy,
    tpr_accuracy,
    label="Accuracy-Optimized"
)


plt.plot(
    fpr_recall,
    tpr_recall,
    label="Recall-Optimized"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve — German Credit Risk"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 28. COUNT MISSED BAD APPLICANTS
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS IMPACT ANALYSIS")
print("=" * 70)


def count_bad_false_negatives(
    model,
    X_data,
    y_data
):

    predictions = model.predict(
        X_data
    )

    # Actual Bad = 2
    # Predicted Good = 1
    #
    # These are the risky applicants
    # that the model failed to identify.

    missed_bad = (
        (y_data == BAD_CLASS)
        &
        (predictions == GOOD_CLASS)
    )

    return int(
        missed_bad.sum()
    )


accuracy_missed_bad = (
    count_bad_false_negatives(
        accuracy_model,
        X_test,
        y_test
    )
)


recall_missed_bad = (
    count_bad_false_negatives(
        recall_model,
        X_test,
        y_test
    )
)


print(
    "Bad applicants missed by Accuracy Model:",
    accuracy_missed_bad
)


print(
    "Bad applicants missed by Recall Model:",
    recall_missed_bad
)


# ============================================================
# 29. ADDITIONAL BAD APPLICANTS CAUGHT
# ============================================================

additional_bad_caught = (

    accuracy_missed_bad
    -
    recall_missed_bad

)


print(
    "\nAdditional Bad applicants caught "
    "by Recall Model:",
    additional_bad_caught
)


# ============================================================
# 30. ESTIMATED DEFAULT LOSS
# ============================================================

estimated_avoided_loss = (

    additional_bad_caught
    *
    AVERAGE_DEFAULT_LOSS

)


print(
    "\nAverage Default Loss Assumption:",
    f"${AVERAGE_DEFAULT_LOSS:,.2f}"
)


print(
    "Estimated Avoided Default Loss:",
    f"${estimated_avoided_loss:,.2f}"
)


# ============================================================
# 31. BUSINESS CASE SUMMARY
# ============================================================

business_summary = pd.DataFrame({

    "Measure": [

        "Bad Applicants Missed - Accuracy Model",

        "Bad Applicants Missed - Recall Model",

        "Additional Bad Applicants Caught",

        "Average Default Loss",

        "Estimated Avoided Default Loss"

    ],

    "Value": [

        accuracy_missed_bad,

        recall_missed_bad,

        additional_bad_caught,

        AVERAGE_DEFAULT_LOSS,

        estimated_avoided_loss

    ]

})


print("\n" + "=" * 70)
print("BUSINESS CASE SUMMARY")
print("=" * 70)


display(
    business_summary
)


# ============================================================
# 32. UCI COST MATRIX ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("UCI COST MATRIX ANALYSIS")
print("=" * 70)


# UCI documented cost matrix:
#
# Actual Good -> Predicted Bad = 1
# Actual Bad  -> Predicted Good = 5


UCI_COST_GOOD_AS_BAD = 1

UCI_COST_BAD_AS_GOOD = 5


def calculate_uci_cost(
    model,
    X_data,
    y_data
):

    predictions = model.predict(
        X_data
    )

    actual_good_predicted_bad = (

        (y_data == GOOD_CLASS)
        &
        (predictions == BAD_CLASS)

    ).sum()


    actual_bad_predicted_good = (

        (y_data == BAD_CLASS)
        &
        (predictions == GOOD_CLASS)

    ).sum()


    total_cost = (

        actual_good_predicted_bad
        *
        UCI_COST_GOOD_AS_BAD

        +

        actual_bad_predicted_good
        *
        UCI_COST_BAD_AS_GOOD

    )


    return {

        "Good_as_Bad": int(
            actual_good_predicted_bad
        ),

        "Bad_as_Good": int(
            actual_bad_predicted_good
        ),

        "Total_Cost": int(
            total_cost
        )

    }


accuracy_uci_cost = calculate_uci_cost(
    accuracy_model,
    X_test,
    y_test
)


recall_uci_cost = calculate_uci_cost(
    recall_model,
    X_test,
    y_test
)


uci_cost_comparison = pd.DataFrame({

    "Model": [

        "Accuracy-Optimized",

        "Recall-Optimized"

    ],

    "Good predicted as Bad": [

        accuracy_uci_cost[
            "Good_as_Bad"
        ],

        recall_uci_cost[
            "Good_as_Bad"
        ]

    ],

    "Bad predicted as Good": [

        accuracy_uci_cost[
            "Bad_as_Good"
        ],

        recall_uci_cost[
            "Bad_as_Good"
        ]

    ],

    "UCI Cost": [

        accuracy_uci_cost[
            "Total_Cost"
        ],

        recall_uci_cost[
            "Total_Cost"
        ]

    ]

})


display(
    uci_cost_comparison
)


# ============================================================
# 33. FINAL MODEL COMPARISON
# ============================================================

final_comparison = results.copy()


final_comparison[
    "Missed_Bad_Applicants"
] = [

    accuracy_missed_bad,

    recall_missed_bad

]


print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)


display(
    final_comparison.round(4)
)


# ============================================================
# 34. BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS INTERPRETATION")
print("=" * 70)


print(
    """
1. Accuracy measures the percentage of all applicants
   classified correctly.

2. Bad-class recall measures the percentage of genuinely
   risky applicants that the model successfully identifies.

3. A credit-risk team should not select a model based only
   on overall accuracy.

4. Missing a Bad applicant means the model predicted
   Good when the applicant was actually Bad.

5. The business impact depends on the financial cost
   associated with such missed risky applicants.

6. The recall-optimized model is specifically trained to
   improve detection of the Bad class.

7. Increasing Bad-class recall can come with a trade-off:
   more Good applicants may also be classified as Bad.

8. Therefore, the final production decision should consider
   both model performance and the organization's actual
   cost of each type of classification error.
"""
)


# ============================================================
# 35. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)


print("\nDataset:")
print("UCI Statlog German Credit Data")

print("\nNumber of Applicants:")
print(len(X))

print("\nNumber of Features:")
print(X.shape[1])

print("\nAccuracy-Optimized Model")
print(
    "Accuracy:",
    round(
        accuracy_results["Accuracy"],
        4
    )
)

print(
    "Bad Recall:",
    round(
        accuracy_results["Recall_Bad"],
        4
    )
)

print(
    "Missed Bad Applicants:",
    accuracy_missed_bad
)


print("\nRecall-Optimized Model")
print(
    "Accuracy:",
    round(
        recall_results["Accuracy"],
        4
    )
)

print(
    "Bad Recall:",
    round(
        recall_results["Recall_Bad"],
        4
    )
)

print(
    "Missed Bad Applicants:",
    recall_missed_bad
)


print(
    "\nAdditional Bad Applicants Caught:",
    additional_bad_caught
)


print(
    "Estimated Avoided Default Loss:",
    f"${estimated_avoided_loss:,.2f}"
)