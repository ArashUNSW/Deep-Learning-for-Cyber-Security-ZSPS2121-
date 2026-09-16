# Week 4 Tutorial — Evaluation, Regularisation, and Model Selection — Student Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will compare two neural networks, use dropout to reduce overfitting, select a threshold using validation data, and evaluate the test set once.



> **Version:** Student Version

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

## Algorithm to Implement

1. Build one MLP without dropout and one with dropout.
2. Train each model using AdamW and weight decay.
3. Keep the state with the lowest validation loss.
4. Compare validation F1-score.
5. Select a decision threshold using validation probabilities.
6. Freeze the procedure and evaluate the test set once.

The code contains guided `TODO` sections. Complete each section using the algorithm above before running the complete workflow.

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
    # TODO 1: Build input -> 24 -> 12 -> 1.
    # Add ReLU after hidden layers and Dropout after each ReLU.
    raise NotImplementedError("Build the regularised MLP.")

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

    # TODO 2: Restore the best validation-loss state
    # before returning the model.
    raise NotImplementedError("Restore the best model state.")

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

    # TODO 3: Calculate validation F1 and update
    # best_f1 and best_threshold when performance improves.
    raise NotImplementedError("Complete threshold selection.")

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



### Challenge — Regularisation Strength

Your assigned dropout probability is:

| Variant | Dropout |
|---|---:|
| A | 0.10 |
| B | 0.30 |
| C | 0.50 |

Compare your assigned model with `dropout=0.0`. Use `STUDENT_SEED`.

#### Algorithm

1. build both models from the same seed;
2. train both using the existing training function;
3. calculate validation F1;
4. calculate training F1 at the selected model state;
5. calculate the training–validation F1 gap;
6. select the better model using validation evidence.

Complete:

```python
assigned_dropout = {
    "A": 0.10,
    "B": 0.30,
    "C": 0.50
}[CHALLENGE_VARIANT]

# TODO: train no-dropout and assigned-dropout models.
# TODO: calculate training F1 and validation F1 for both.
# TODO: calculate absolute training-validation F1 gap.
raise NotImplementedError("Complete the regularisation challenge.")
```

#### Evidence Table

| Dropout | Training F1 | Validation F1 | Absolute Gap |
|---:|---:|---:|---:|
| 0.00 | | | |
| assigned | | | |

#### Code Explanation

Why must `model.eval()` be used when evaluating a model containing dropout?

#### Interpretation

Does your evidence show that the assigned dropout improved generalisation? Use the measured F1 values and gap.

## Practice Challenge

Explain:

- Did dropout improve validation performance?
- Why must the threshold be selected using validation data?
- Why should the test set not be repeatedly reused?

## Student Check Before Submission

- All `TODO` sections have been completed.
- The notebook runs from the first cell to the final cell.
- Generated outputs have been checked.
- Challenge questions have been answered using observed results.

## Submission

Submit the notebook and a one-page model-selection justification.
