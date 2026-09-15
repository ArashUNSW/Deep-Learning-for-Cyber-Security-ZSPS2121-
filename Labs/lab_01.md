# Week 1 Tutorial — Authentication Data and Baseline Model — Student Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

This tutorial introduces the BlueWave Sentinel case study. You will inspect authentication data, preserve the supplied train/validation/test split, and create a simple baseline model.



> **Version:** Student Version

## Model Used

**Model:** Logistic Regression Baseline  
**Model category:** Classical supervised baseline

Logistic regression learns a weighted relationship between authentication features and the probability of a suspicious session. It is intentionally simple, fast, and interpretable.

## Purpose in the Deep-Learning Workflow

A deep-learning project needs a credible baseline. The baseline shows whether later neural networks provide a real improvement over a simpler model.

## Model Input and Output

Input: scaled numerical authentication fields and one-hot encoded categorical fields. Output: a suspicious-session probability between 0 and 1.

## Important Limitation

It models a mostly linear decision boundary and may miss complex interactions. Its probability is a triage signal, not proof of malicious activity.

## Algorithm to Implement

1. Load the authentication dataset.
2. Use the supplied split to create training, validation, and test tables.
3. Scale numerical features and one-hot encode categorical features.
4. Fit logistic regression using training data only.
5. Convert test probabilities into decisions using a threshold.
6. Evaluate precision, recall, F1-score, and the confusion matrix.

The code contains guided `TODO` sections. Complete each section using the algorithm above before running the complete workflow.

## Dataset

```text
C:\BlueWave\data\authentication_events.csv
```

Main target:

```text
label_suspicious_session
```

## Steps

1. Open Jupyter Notebook.
2. Create a notebook named `Week01_Authentication_Baseline.ipynb`.
3. Copy and run the code below.
4. Review the dataset shape, missing values, class balance, and model results.
5. Save the generated CSV and image.

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(r"C:\BlueWave\data")
OUTPUT_DIR = Path(r"C:\BlueWave\outputs\week01")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA_DIR / "authentication_events.csv")

print("Shape:", data.shape)
print("\nMissing values:")
print(data.isna().sum())
print("\nClass balance:")
print(data["label_suspicious_session"].value_counts())

numeric_features = [
    "login_hour",
    "failed_attempts_24h",
    "impossible_travel_score",
    "device_trust_score",
    "mfa_used",
    "privileged_account",
    "new_device",
    "vpn_used",
    "login_success"
]

categorical_features = [
    "role",
    "department",
    "source_country"
]

features = numeric_features + categorical_features
target = "label_suspicious_session"

# TODO 1: Create the three supplied splits.
# Algorithm:
# - Select rows where split equals "train".
# - Select rows where split equals "validation".
# - Select rows where split equals "test".
train = None
validation = None
test = None

# TODO 2: Build the preprocessing object.
# Numerical fields require StandardScaler.
# Categorical fields require OneHotEncoder(handle_unknown="ignore").
preprocessor = None

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# TODO 3: Fit the complete pipeline using training features and labels.
raise NotImplementedError("Fit the baseline model.")

test_probability = model.predict_proba(test[features])[:, 1]
test_prediction = (test_probability >= 0.50).astype(int)

print(classification_report(
    test[target],
    test_prediction,
    digits=3
))

results = pd.DataFrame({
    "session_id": test["session_id"],
    "true_label": test[target],
    "probability": test_probability,
    "prediction": test_prediction
})

results.to_csv(
    OUTPUT_DIR / "week01_authentication_predictions.csv",
    index=False
)

ConfusionMatrixDisplay.from_predictions(
    test[target],
    test_prediction
)
plt.title("Week 1 Authentication Baseline")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "week01_confusion_matrix.png",
    dpi=150
)
plt.show()
```

## Expected Output

```text
C:\BlueWave\outputs\week01\week01_authentication_predictions.csv
C:\BlueWave\outputs\week01\week01_confusion_matrix.png
```

## Assessed Challenge — Individual Submission

This challenge is part of the tutorial submission. Your answer must be based on **your own executed notebook and your own observed results**.

### Your Individual Challenge Settings

Use only the **last three digits** of your student number.

```python
STUDENT_LAST3 = 123  # replace 123 with your own last three digits
STUDENT_SEED = 1000 + STUDENT_LAST3
CHALLENGE_GROUP = STUDENT_LAST3 % 10

if CHALLENGE_GROUP <= 3:
    CHALLENGE_VARIANT = "A"
elif CHALLENGE_GROUP <= 6:
    CHALLENGE_VARIANT = "B"
else:
    CHALLENGE_VARIANT = "C"

print("Student seed:", STUDENT_SEED)
print("Challenge variant:", CHALLENGE_VARIANT)
```

Where a PyTorch or NumPy random seed is used in the challenge, use `STUDENT_SEED`.

### What You Must Submit for the Challenge

Submit all of the following:

1. the completed challenge code in your notebook;
2. the executed output showing your assigned challenge variant;
3. an **Evidence Table** containing the exact numerical values requested below;
4. the requested plot or CSV output;
5. a **150–250 word interpretation** that refers to your actual measured values;
6. a short **Code Explanation** answering the tutorial-specific question; and
7. the declaration:

> I generated the reported results from my own Skillable lab run and can explain the code and results.

Follow the course rules for the use of generative AI or other assistance. Generic explanations that are not supported by the submitted notebook outputs do not satisfy this challenge.



### Challenge — Decision Threshold and SOC Workload

Your assigned threshold pair is:

| Variant | Threshold 1 | Threshold 2 |
|---|---:|---:|
| A | 0.25 | 0.55 |
| B | 0.35 | 0.65 |
| C | 0.45 | 0.75 |

Use the probabilities already produced by the baseline model.

#### Algorithm

For each assigned threshold:

1. convert probability to a binary prediction;
2. calculate precision, recall, and F1-score;
3. calculate the number of predicted alerts;
4. calculate the number of false positives and false negatives;
5. compare the two operating points.

Complete:

```python
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

threshold_map = {
    "A": [0.25, 0.55],
    "B": [0.35, 0.65],
    "C": [0.45, 0.75]
}

assigned_thresholds = threshold_map[CHALLENGE_VARIANT]

challenge_rows = []

for threshold in assigned_thresholds:
    # TODO: convert test_probability to predictions.
    prediction = None

    # TODO: calculate TN, FP, FN, TP.
    tn, fp, fn, tp = 0, 0, 0, 0

    # TODO: append threshold, precision, recall, F1,
    # alert count, FP and FN to challenge_rows.
    raise NotImplementedError("Complete the threshold challenge.")
```

#### Evidence Table

| Threshold | Precision | Recall | F1 | Alerts | False Positives | False Negatives |
|---:|---:|---:|---:|---:|---:|---:|
| assigned 1 | | | | | | |
| assigned 2 | | | | | | |

#### Code Explanation

Why does changing the threshold alter the SOC workload even though the trained model has not changed?

#### Interpretation

State which of your two thresholds you would choose **for this synthetic exercise** and justify the decision using at least three values from your Evidence Table.

## Practice Challenge

Change the decision threshold from `0.50` to `0.30` and `0.70`.

Explain:

- Which threshold produces more alerts?
- Which threshold misses more suspicious sessions?
- Why is accuracy alone insufficient for this dataset?

## Student Check Before Submission

- All `TODO` sections have been completed.
- The notebook runs from the first cell to the final cell.
- Generated outputs have been checked.
- Challenge questions have been answered using observed results.

## Submission

Submit the notebook, prediction CSV, confusion matrix, and a short paragraph answering the challenge.
