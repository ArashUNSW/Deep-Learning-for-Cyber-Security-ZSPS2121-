# Week 4 Tutorial — Evaluation, Regularisation, and Model Selection — Tutor Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will compare two neural networks, use dropout to reduce overfitting, select a threshold using validation data, and evaluate the test set once.



> **Version:** Tutor Version — Complete Implementation

## Model Used

**Model:** Regularised MLP with Early Model Selection  
**Model category:** Generalisation and model-selection workflow

Two MLP configurations are compared. Dropout randomly removes hidden activations during training, while weight decay discourages excessively large parameters.

## Purpose in the Deep-Learning Workflow

Deep models can fit training data without generalising. Regularisation, validation selection, and threshold selection create a more defensible final model.

## Model Input and Output

Input: the same authentication features. Output: a selected model, validation-selected threshold, and one final test evaluation.

## Important Limitation

The selected model may still be unstable across random seeds or future data. The test set must not become another tuning set.

## Tutor Teaching Notes

- Dropout is active in training mode and disabled in evaluation mode.
- The best model state is copied when validation loss improves.
- Threshold selection changes the balance between false positives and false negatives.
- The test set is evaluated only after model and threshold choices are fixed.

## Dataset

```text
C:\BlueWave\data\authentication_events.csv
```

## Steps

1. Create `Week04_Model_Selection.ipynb`.
2. Reuse the Week 3 preprocessing.
3. Train a model without dropout and a model with dropout.
4. Compare validation F1-score.
5. Select a threshold using validation data only.
6. Evaluate the test set once.

```python
from pathlib import Path
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    f1_score
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

DATA_DIR = Path(r"C:\BlueWave\data")
OUTPUT_DIR = Path(r"C:\BlueWave\outputs\week04")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

data = pd.read_csv(
    DATA_DIR / "authentication_events.csv"
)

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
    ("categorical", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), categorical_features)
])

X_train = torch.tensor(
    preprocessor.fit_transform(train[features]),
    dtype=torch.float32
)

X_validation = torch.tensor(
    preprocessor.transform(validation[features]),
    dtype=torch.float32
)

X_test = torch.tensor(
    preprocessor.transform(test[features]),
    dtype=torch.float32
)

y_train = torch.tensor(
    train[target].to_numpy().reshape(-1, 1),
    dtype=torch.float32
)

y_validation = torch.tensor(
    validation[target].to_numpy().reshape(-1, 1),
    dtype=torch.float32
)

y_test = torch.tensor(
    test[target].to_numpy().reshape(-1, 1),
    dtype=torch.float32
)

def build_model(input_size, dropout):
    return nn.Sequential(
        nn.Linear(input_size, 24),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(12, 1)
    )

def train_model(model, epochs=150):
    loss_function = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.001
    )

    best_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        loss = loss_function(model(X_train), y_train)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(X_validation),
                y_validation
            ).item()

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model

models = {
    "no_dropout": build_model(X_train.shape[1], 0.0),
    "dropout_025": build_model(X_train.shape[1], 0.25)
}

validation_rows = []

for name, model in models.items():
    trained = train_model(model)

    with torch.no_grad():
        probability = torch.sigmoid(
            trained(X_validation)
        ).numpy().ravel()

    prediction = (probability >= 0.50).astype(int)

    validation_rows.append({
        "model": name,
        "validation_f1": f1_score(
            y_validation.numpy().ravel(),
            prediction
        )
    })

results = pd.DataFrame(validation_rows)
print(results)

selected_name = results.sort_values(
    "validation_f1",
    ascending=False
).iloc[0]["model"]

selected_model = models[selected_name]

with torch.no_grad():
    validation_probability = torch.sigmoid(
        selected_model(X_validation)
    ).numpy().ravel()

best_threshold = 0.50
best_f1 = -1

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    prediction = (
        validation_probability >= threshold
    ).astype(int)

    score = f1_score(
        y_validation.numpy().ravel(),
        prediction
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

with torch.no_grad():
    test_probability = torch.sigmoid(
        selected_model(X_test)
    ).numpy().ravel()

test_prediction = (
    test_probability >= best_threshold
).astype(int)

print("Selected model:", selected_name)
print("Selected threshold:", best_threshold)
print(classification_report(
    y_test.numpy().ravel(),
    test_prediction,
    digits=3
))
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



### Challenge Solution — Regularisation

Tutors may reuse `build_model()` and `train_model()` from the completed tutorial. Before each run:

```python
torch.manual_seed(STUDENT_SEED)

assigned_dropout = {
    "A": 0.10,
    "B": 0.30,
    "C": 0.50
}[CHALLENGE_VARIANT]

challenge_models = {
    "dropout_000": build_model(
        X_train.shape[1], 0.0
    ),
    "assigned_dropout": build_model(
        X_train.shape[1], assigned_dropout
    )
}

challenge_rows = []

for name, challenge_model in challenge_models.items():
    trained_model = train_model(challenge_model)

    trained_model.eval()

    with torch.no_grad():
        train_probability = torch.sigmoid(
            trained_model(X_train)
        ).numpy().ravel()

        validation_probability = torch.sigmoid(
            trained_model(X_validation)
        ).numpy().ravel()

    train_prediction = (
        train_probability >= 0.50
    ).astype(int)

    validation_prediction = (
        validation_probability >= 0.50
    ).astype(int)

    train_f1 = f1_score(
        y_train.numpy().ravel(),
        train_prediction
    )

    validation_f1 = f1_score(
        y_validation.numpy().ravel(),
        validation_prediction
    )

    challenge_rows.append({
        "model": name,
        "training_f1": train_f1,
        "validation_f1": validation_f1,
        "absolute_gap": abs(
            train_f1 - validation_f1
        )
    })

display(pd.DataFrame(challenge_rows))
```

There is no requirement that dropout always improve F1. Students should report what their evidence shows.

## Practice Challenge

Explain:

- Did dropout improve validation performance?
- Why must the threshold be selected using validation data?
- Why should the test set not be repeatedly reused?

## Suggested Challenge Discussion

- Dropout is active in training mode and disabled in evaluation mode.
- The best model state is copied when validation loss improves.
- Threshold selection changes the balance between false positives and false negatives.
- The test set is evaluated only after model and threshold choices are fixed.

Tutors should require students to justify conclusions using their own outputs rather than assuming that a more complex model is automatically better.

## Submission

Submit the notebook and a one-page model-selection justification.
