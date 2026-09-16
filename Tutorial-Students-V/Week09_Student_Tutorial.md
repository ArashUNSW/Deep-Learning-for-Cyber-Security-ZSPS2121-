# Week 9 Tutorial — Integrated Cybersecurity Decision Support — Student Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will combine four model scores into one session-level triage score and then review robustness, ethics, legal risk, and human oversight.



> **Version:** Student Version

## Model Used

**Model:** Logistic Stacking Fusion Model  
**Model category:** Model integration and decision-support layer

The stacking model learns how to combine probabilities or scores produced by the authentication, artefact, sequence, and flow models.

## Purpose in the Deep-Learning Workflow

Real deep-learning systems often contain several specialised models. A fusion layer creates one session-level decision while preserving missing-evidence information.

## Model Input and Output

Input: four model scores and four missingness indicators. Output: one integrated suspicious-session probability.

## Important Limitation

Stacking can amplify errors or hidden bias from upstream models. Missing evidence must not be silently converted into evidence of normality.

## Algorithm to Implement

1. Join all evidence sources using session_id.
2. Keep missing scores as missing values.
3. Create one missingness indicator per score.
4. Fit median imputation and feature scaling on training data.
5. Train logistic regression as the stacking layer.
6. Evaluate integrated predictions and test graceful degradation.

The code contains guided `TODO` sections. Complete each section using the algorithm above before running the complete workflow.

## Prepared Score Files

The instructor places these files in:

```text
C:\BlueWave\data
```

Required files:

```text
authentication_scores.csv
artifact_scores.csv
sequence_scores.csv
flow_scores.csv
session_split.csv
```

Each score file must contain:

```text
session_id
one score column between 0 and 1
```

## Steps

1. Create `Week09_Integration.ipynb`.
2. Join the score files using `session_id`.
3. Preserve missing values.
4. Calculate a transparent mean score.
5. Create a simple logistic stacking model.
6. Evaluate the test split.
7. Review subgroup and governance risks.

```python
from pathlib import Path
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(r"C:\BlueWave\data")
OUTPUT_DIR = Path(r"C:\BlueWave\outputs\week09")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sessions = pd.read_csv(DATA_DIR / "session_split.csv")
auth = pd.read_csv(DATA_DIR / "authentication_scores.csv")
artifact = pd.read_csv(DATA_DIR / "artifact_scores.csv")
sequence = pd.read_csv(DATA_DIR / "sequence_scores.csv")
flow = pd.read_csv(DATA_DIR / "flow_scores.csv")

# TODO 1: Join all score tables to sessions.
# Use session_id and left joins so missing evidence is preserved.
combined = None

score_columns = [
    "auth_score",
    "artifact_score",
    "sequence_score",
    "flow_score"
]

combined["available_evidence"] = (
    combined[score_columns]
    .notna()
    .sum(axis=1)
)

combined["mean_available_score"] = (
    combined[score_columns]
    .mean(axis=1, skipna=True)
)

# TODO 2: Create one integer missingness indicator
# for every score column.
raise NotImplementedError("Create missingness indicators.")

model_features = score_columns + [
    column + "_missing"
    for column in score_columns
]

train = combined[combined["split"] == "train"]
test = combined[combined["split"] == "test"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# TODO 3: Fit the stacking pipeline using
# training model features and training labels.
raise NotImplementedError("Fit the fusion model.")

test_probability = model.predict_proba(
    test[model_features]
)[:, 1]

test_prediction = (
    test_probability >= 0.50
).astype(int)

print(classification_report(
    test["label_suspicious_session"],
    test_prediction,
    digits=3
))

results = test[[
    "session_id",
    "label_suspicious_session",
    "available_evidence",
    "mean_available_score"
]].copy()

results["integrated_probability"] = test_probability
results["integrated_prediction"] = test_prediction

results.to_csv(
    OUTPUT_DIR / "week09_integrated_predictions.csv",
    index=False
)
```

## Governance Review

Answer these questions:

- What happens when one evidence source is missing?
- Can the model automatically block an account?
- Who reviews high-risk decisions?
- How can a person challenge an incorrect decision?
- What personal information is collected?
- How long should evidence and model outputs be retained?
- Which legal or policy questions require professional advice?

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



### Challenge — Missing Modality and Graceful Degradation

Your assigned evidence source to remove is:

| Variant | Remove |
|---|---|
| A | `auth_score` |
| B | `artifact_score` |
| C | `sequence_score` |

The flow score remains available in all variants.

#### Algorithm

1. evaluate the original integrated model;
2. make a copy of the test model features;
3. replace the assigned score with missing values;
4. set its missingness indicator to 1;
5. obtain new probabilities and predictions;
6. compare F1 and the percentage of decisions that changed.

Complete:

```python
remove_map = {
    "A": "auth_score",
    "B": "artifact_score",
    "C": "sequence_score"
}

removed_score = remove_map[CHALLENGE_VARIANT]

# TODO: create a modified copy of test[model_features].
# TODO: set removed_score to missing.
# TODO: set removed_score + "_missing" to 1.
# TODO: calculate new probability and prediction.
# TODO: compare with the original integrated result.
raise NotImplementedError("Complete the missing-modality challenge.")
```

#### Evidence Table

| Condition | F1 | Alerts | Changed Decisions |
|---|---:|---:|---:|
| All available evidence | | | |
| Assigned modality unavailable | | | |

#### Code Explanation

Why is replacing an unavailable score with `0` different from representing the score as missing?

#### Interpretation

Recommend a safe fallback for the SOC when your assigned evidence source is unavailable. Your answer must refer to the measured performance change.

## Practice Challenge

Remove one score source at a time and measure the change in test F1-score.

Recommend a safe fallback when one model is unavailable.

## Student Check Before Submission

- All `TODO` sections have been completed.
- The notebook runs from the first cell to the final cell.
- Generated outputs have been checked.
- Challenge questions have been answered using observed results.

## Submission

Submit the notebook, integrated prediction CSV, missing-modality comparison, and a short ethical and legal risk statement.
