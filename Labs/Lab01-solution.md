# Week 1 Tutorial — Authentication Data and Baseline Model — Tutor Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

This tutorial introduces the BlueWave Sentinel case study. You will inspect authentication data, preserve the supplied train/validation/test split, and create a simple baseline model.



> **Version:** Tutor Version — Complete Implementation

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

## Tutor Teaching Notes

- The training split is used to fit both preprocessing and the classifier.
- Lower thresholds normally increase recall and alert volume.
- Higher thresholds normally reduce alert volume but can increase false negatives.
- This model establishes the performance level that deep models should exceed.

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

train = data[data["split"] == "train"]
validation = data[data["split"] == "validation"]
test = data[data["split"] == "test"]

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(train[features], train[target])

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

## Tutor Marking Guide — Individual Assessed Challenge

The assessed challenge is deliberately evidence-dependent. It is not possible to guarantee that a student cannot consult ChatGPT or another tool, so marking should focus on **authentic execution evidence and demonstrated understanding** rather than generic prose.

### Individualisation

Students use:

```text
STUDENT_SEED = 1000 + last three digits of student number

last digit 0–3  → Variant A
last digit 4–6  → Variant B
last digit 7–9  → Variant C
```

The notebook must show the assigned seed and variant.

### Recommended Marking — 10 Marks

| Criterion | Marks |
|---|---:|
| Correct individual seed/variant and completed implementation | 2 |
| Executed evidence and requested output artefact | 3 |
| Numerical Evidence Table agrees with notebook output | 2 |
| Interpretation correctly refers to the student's measured values | 2 |
| Student can explain the designated code/algorithm step | 1 |

### Evidence Checks

Do not award full marks for a generic answer with no measured values. Check that:

- the challenge variant matches the student's last digit;
- reported numbers appear in the executed notebook;
- submitted plots/CSVs agree with those numbers;
- the student completed the required algorithmic section rather than pasting unexplained output; and
- the conclusion is bounded by the actual experiment.

A short tutor spot-check can ask the student to explain one line or predict what would happen if one parameter changed.



### Challenge Solution — Decision Threshold and SOC Workload

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
    prediction = (
        test_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        test[target],
        prediction,
        labels=[0, 1]
    ).ravel()

    challenge_rows.append({
        "threshold": threshold,
        "precision": precision_score(
            test[target], prediction, zero_division=0
        ),
        "recall": recall_score(
            test[target], prediction, zero_division=0
        ),
        "f1": f1_score(
            test[target], prediction, zero_division=0
        ),
        "alerts": int(prediction.sum()),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    })

challenge_results = pd.DataFrame(challenge_rows)
display(challenge_results)
```

**Expected reasoning:** lowering the threshold normally increases alerts and recall while often increasing false positives. Raising it normally reduces alerts but may increase false negatives. The preferred threshold is a risk/workload decision, not an accuracy-only decision.

## Practice Challenge

Change the decision threshold from `0.50` to `0.30` and `0.70`.

Explain:

- Which threshold produces more alerts?
- Which threshold misses more suspicious sessions?
- Why is accuracy alone insufficient for this dataset?

## Suggested Challenge Discussion

- The training split is used to fit both preprocessing and the classifier.
- Lower thresholds normally increase recall and alert volume.
- Higher thresholds normally reduce alert volume but can increase false negatives.
- This model establishes the performance level that deep models should exceed.

Tutors should require students to justify conclusions using their own outputs rather than assuming that a more complex model is automatically better.

## Submission

Submit the notebook, prediction CSV, confusion matrix, and a short paragraph answering the challenge.
