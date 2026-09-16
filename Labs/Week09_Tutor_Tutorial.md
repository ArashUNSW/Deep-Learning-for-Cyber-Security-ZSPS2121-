# Week 9 Tutorial — Integrated Cybersecurity Decision Support — Tutor Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will combine four model scores into one session-level triage score and then review robustness, ethics, legal risk, and human oversight.



> **Version:** Tutor Version — Complete Implementation

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

## Tutor Teaching Notes

- Left joins preserve sessions when a modality is unavailable.
- Median imputation occurs inside the pipeline and is learned from training data.
- Missingness indicators allow the model to distinguish absent evidence from ordinary scores.
- The integrated probability remains decision support and cannot justify autonomous punitive action.

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

combined = (
    sessions
    .merge(auth, on="session_id", how="left")
    .merge(artifact, on="session_id", how="left")
    .merge(sequence, on="session_id", how="left")
    .merge(flow, on="session_id", how="left")
)

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

for column in score_columns:
    combined[column + "_missing"] = (
        combined[column].isna().astype(int)
    )

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

model.fit(
    train[model_features],
    train["label_suspicious_session"]
)

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



### Challenge Solution — Missing Modality

```python
from sklearn.metrics import f1_score

remove_map = {
    "A": "auth_score",
    "B": "artifact_score",
    "C": "sequence_score"
}

removed_score = remove_map[CHALLENGE_VARIANT]

original_prediction = test_prediction.copy()

modified_test = test[
    model_features
].copy()

modified_test[removed_score] = pd.NA
modified_test[
    removed_score + "_missing"
] = 1

modified_probability = model.predict_proba(
    modified_test
)[:, 1]

modified_prediction = (
    modified_probability >= 0.50
).astype(int)

original_f1 = f1_score(
    test["label_suspicious_session"],
    original_prediction
)

modified_f1 = f1_score(
    test["label_suspicious_session"],
    modified_prediction
)

changed_decisions = int(
    (modified_prediction != original_prediction).sum()
)

print("Original F1:", original_f1)
print("Modified F1:", modified_f1)
print("Changed decisions:", changed_decisions)
```

A safe answer may include analyst review, a transparent fallback score, or reduced automation. The correct recommendation must be tied to the measured degradation.

## Practice Challenge

Remove one score source at a time and measure the change in test F1-score.

Recommend a safe fallback when one model is unavailable.

## Suggested Challenge Discussion

- Left joins preserve sessions when a modality is unavailable.
- Median imputation occurs inside the pipeline and is learned from training data.
- Missingness indicators allow the model to distinguish absent evidence from ordinary scores.
- The integrated probability remains decision support and cannot justify autonomous punitive action.

Tutors should require students to justify conclusions using their own outputs rather than assuming that a more complex model is automatically better.

## Submission

Submit the notebook, integrated prediction CSV, missing-modality comparison, and a short ethical and legal risk statement.
