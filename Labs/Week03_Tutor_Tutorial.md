# Week 3 Tutorial — Loss, Backpropagation, and Optimisation — Tutor Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will train the Week 2 neural network using binary cross-entropy loss, backpropagation, and the Adam optimiser.



> **Version:** Tutor Version — Complete Implementation

## Model Used

**Model:** Trained Authentication MLP  
**Model category:** Supervised neural classifier

The Week 2 MLP is trained by comparing its logits with known labels. Backpropagation calculates gradients, and Adam updates the weights.

## Purpose in the Deep-Learning Workflow

This tutorial demonstrates the complete learning cycle that turns a randomly initialised network into a model fitted to cybersecurity data.

## Model Input and Output

Input: transformed authentication tensors and binary labels. Output: a trained model plus training and validation loss curves.

## Important Limitation

A decreasing training loss does not guarantee generalisation. Model selection must be based on validation evidence, not repeated test-set tuning.

## Tutor Teaching Notes

- The required order is forward → loss → zero gradients → backward → optimiser step.
- BCEWithLogitsLoss is used directly on logits.
- Validation evaluation belongs inside torch.no_grad().
- Training and validation curves help identify poor learning or overfitting.

## Dataset

```text
C:\BlueWave\data\authentication_events.csv
```

## Steps

1. Create `Week03_Training.ipynb`.
2. Prepare the authentication features.
3. Train the neural network.
4. Plot training and validation loss.
5. Save the trained model.

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

DATA_DIR = Path(r"C:\BlueWave\data")
OUTPUT_DIR = Path(r"C:\BlueWave\outputs\week03")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

data = pd.read_csv(DATA_DIR / "authentication_events.csv")

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

y_train = torch.tensor(
    train[target].to_numpy().reshape(-1, 1),
    dtype=torch.float32
)
y_validation = torch.tensor(
    validation[target].to_numpy().reshape(-1, 1),
    dtype=torch.float32
)

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 1)
)

loss_function = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

history = []

for epoch in range(1, 101):
    model.train()

    train_logits = model(X_train)
    train_loss = loss_function(train_logits, y_train)

    optimiser.zero_grad()
    train_loss.backward()
    optimiser.step()

    model.eval()

    with torch.no_grad():
        validation_logits = model(X_validation)
        validation_loss = loss_function(
            validation_logits,
            y_validation
        )

    history.append({
        "epoch": epoch,
        "training_loss": train_loss.item(),
        "validation_loss": validation_loss.item()
    })

history = pd.DataFrame(history)
history.to_csv(
    OUTPUT_DIR / "week03_training_history.csv",
    index=False
)

plt.plot(history["epoch"], history["training_loss"], label="Training")
plt.plot(history["epoch"], history["validation_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "week03_loss_curve.png", dpi=150)
plt.show()

torch.save(
    model.state_dict(),
    OUTPUT_DIR / "week03_authentication_model.pt"
)
```

## Expected Output

```text
week03_training_history.csv
week03_loss_curve.png
week03_authentication_model.pt
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



### Challenge Solution — Learning Rate

```python
def run_learning_rate_experiment(learning_rate):
    torch.manual_seed(STUDENT_SEED)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 1)
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    for epoch in range(1, 101):
        model.train()
        train_logits = model(X_train)
        train_loss = loss_function(
            train_logits, y_train
        )

        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(X_validation),
                y_validation
            ).item()

        history.append({
            "epoch": epoch,
            "training_loss": train_loss.item(),
            "validation_loss": validation_loss
        })

    return pd.DataFrame(history)

alternative_map = {
    "A": 0.0005,
    "B": 0.002,
    "C": 0.005
}

baseline_history = run_learning_rate_experiment(0.001)
challenge_history = run_learning_rate_experiment(
    alternative_map[CHALLENGE_VARIANT]
)
```

Tutors should expect evidence from the curves rather than a predetermined “best” learning rate.

## Practice Challenge

Run the experiment with learning rates:

```text
0.0001
0.001
0.01
```

Explain which learning rate:

- learns too slowly;
- is stable;
- becomes unstable or produces worse validation loss.

## Suggested Challenge Discussion

- The required order is forward → loss → zero gradients → backward → optimiser step.
- BCEWithLogitsLoss is used directly on logits.
- Validation evaluation belongs inside torch.no_grad().
- Training and validation curves help identify poor learning or overfitting.

Tutors should require students to justify conclusions using their own outputs rather than assuming that a more complex model is automatically better.

## Submission

Submit the notebook, loss curve, training history, and a short optimisation comparison.
