# Week 8 Tutorial — Autoencoder for Network-Flow Anomaly Detection — Tutor Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will train an autoencoder using normal network flows and use reconstruction error as an anomaly score.



> **Version:** Tutor Version — Complete Implementation

## Model Used

**Model:** Autoencoder  
**Model category:** Unsupervised or semi-supervised representation model

The autoencoder compresses normal flow features into a small latent representation and reconstructs the original features.

## Purpose in the Deep-Learning Workflow

When reliable attack labels are limited, a model can learn normal behaviour. Large reconstruction error then becomes an anomaly score.

## Model Input and Output

Input: seven scaled network-flow features. Output: reconstructed features and a reconstruction-error anomaly score.

## Important Limitation

An anomaly is not automatically an attack. Attacks similar to normal data may have low error, while benign novelty may have high error.

## Tutor Teaching Notes

- Labels are used to select normal training rows and evaluate results, not as model inputs.
- The bottleneck encourages compact representations.
- The 95th percentile threshold marks approximately the highest-scoring normal validation flows.
- Threshold choice directly affects alert volume and missed anomalies.

## Dataset

```text
C:\BlueWave\data\network_flows.csv
```

## Steps

1. Create `Week08_Autoencoder.ipynb`.
2. Select numerical flow features.
3. Train only on normal training flows.
4. Calculate reconstruction error.
5. Select a threshold using validation data.
6. Evaluate the test split.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch import nn

DATA_DIR = Path(r"C:\BlueWave\data")
OUTPUT_DIR = Path(r"C:\BlueWave\outputs\week08")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

data = pd.read_csv(DATA_DIR / "network_flows.csv")

features = [
    "duration_seconds",
    "packet_count",
    "byte_count",
    "failed_connection_score",
    "unusual_port_score",
    "inbound_outbound_ratio",
    "destination_reputation_risk"
]

train = data[data["split"] == "train"]
validation = data[data["split"] == "validation"]
test = data[data["split"] == "test"]

normal_train = train[train["label_anomaly"] == 0]
normal_validation = validation[
    validation["label_anomaly"] == 0
]

scaler = StandardScaler()

X_train = torch.tensor(
    scaler.fit_transform(normal_train[features]),
    dtype=torch.float32
)

X_validation_normal = torch.tensor(
    scaler.transform(normal_validation[features]),
    dtype=torch.float32
)

X_validation = torch.tensor(
    scaler.transform(validation[features]),
    dtype=torch.float32
)

X_test = torch.tensor(
    scaler.transform(test[features]),
    dtype=torch.float32
)

model = nn.Sequential(
    nn.Linear(7, 12),
    nn.ReLU(),
    nn.Linear(12, 3),
    nn.ReLU(),
    nn.Linear(3, 12),
    nn.ReLU(),
    nn.Linear(12, 7)
)

loss_function = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    model.train()
    reconstruction = model(X_train)
    loss = loss_function(reconstruction, X_train)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

def error_scores(x):
    model.eval()
    with torch.no_grad():
        reconstruction = model(x)
        return torch.mean(
            (x - reconstruction) ** 2,
            dim=1
        ).numpy()

normal_validation_error = error_scores(
    X_validation_normal
)
validation_error = error_scores(X_validation)
test_error = error_scores(X_test)

threshold = np.quantile(
    normal_validation_error,
    0.95
)

test_prediction = (
    test_error > threshold
).astype(int)

print("Selected threshold:", threshold)
print(classification_report(
    test["label_anomaly"],
    test_prediction,
    digits=3
))

pd.DataFrame({
    "flow_id": test["flow_id"],
    "session_id": test["session_id"],
    "true_label": test["label_anomaly"],
    "reconstruction_error": test_error,
    "prediction": test_prediction
}).to_csv(
    OUTPUT_DIR / "week08_flow_anomaly_scores.csv",
    index=False
)
```

## Important Interpretation

A high reconstruction error means the flow is unusual for the learned normal pattern.

It does **not** automatically mean the flow is an attack.

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



### Challenge Solution — Anomaly Threshold

```python
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

assigned_percentile = {
    "A": 90,
    "B": 97,
    "C": 99
}[CHALLENGE_VARIANT]

challenge_rows = []

for percentile in [95, assigned_percentile]:
    threshold_value = np.percentile(
        normal_validation_error,
        percentile
    )

    prediction = (
        test_error > threshold_value
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        test["label_anomaly"],
        prediction,
        labels=[0, 1]
    ).ravel()

    challenge_rows.append({
        "percentile": percentile,
        "threshold": threshold_value,
        "precision": precision_score(
            test["label_anomaly"],
            prediction,
            zero_division=0
        ),
        "recall": recall_score(
            test["label_anomaly"],
            prediction,
            zero_division=0
        ),
        "f1": f1_score(
            test["label_anomaly"],
            prediction,
            zero_division=0
        ),
        "alerts": int(prediction.sum()),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    })

display(pd.DataFrame(challenge_rows))
```

Students should identify the threshold trade-off rather than equating a higher percentile with a better detector.

## Practice Challenge

Compare thresholds based on:

```text
90th percentile
95th percentile
99th percentile
```

Explain the effect on false positives and false negatives.

## Suggested Challenge Discussion

- Labels are used to select normal training rows and evaluate results, not as model inputs.
- The bottleneck encourages compact representations.
- The 95th percentile threshold marks approximately the highest-scoring normal validation flows.
- Threshold choice directly affects alert volume and missed anomalies.

Tutors should require students to justify conclusions using their own outputs rather than assuming that a more complex model is automatically better.

## Submission

Submit the notebook, anomaly-score CSV, and threshold comparison.
