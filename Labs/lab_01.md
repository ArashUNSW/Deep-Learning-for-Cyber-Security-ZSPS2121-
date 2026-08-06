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

## Challenge

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
