# Week 8 Tutorial — Autoencoder for Network-Flow Anomaly Detection — Student Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will train an autoencoder using normal network flows and use reconstruction error as an anomaly score.



> **Version:** Student Version

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

## Algorithm to Implement

1. Select normal training flows only.
2. Fit the scaler on normal training data.
3. Encode seven features into a smaller latent vector.
4. Decode the latent vector back to seven reconstructed features.
5. Minimise mean squared reconstruction loss.
6. Calculate per-flow reconstruction error.
7. Use normal validation errors to select an anomaly threshold.

The code contains guided `TODO` sections. Complete each section using the algorithm above before running the complete workflow.

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

# TODO 1: Implement the autoencoder:
# 7 -> 12 -> 3 -> 12 -> 7
# Use ReLU after all hidden layers.
model = None

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
        # TODO 2: Reconstruct x and calculate
        # mean squared error for every row.
        raise NotImplementedError("Calculate reconstruction errors.")

normal_validation_error = error_scores(
    X_validation_normal
)
validation_error = error_scores(X_validation)
test_error = error_scores(X_test)

# TODO 3: Select the 95th percentile of
# normal validation reconstruction errors.
threshold = None

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



### Challenge — Anomaly Threshold Policy

Your assigned normal-validation percentile is:

| Variant | Percentile |
|---|---:|
| A | 90 |
| B | 97 |
| C | 99 |

Compare it with the tutorial percentile `95`.

#### Algorithm

For both percentiles:

1. calculate the threshold from `normal_validation_error`;
2. classify the test flows;
3. calculate precision, recall, F1;
4. calculate alert count;
5. calculate false positives and false negatives.

Complete:

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

# TODO: evaluate percentile 95 and assigned_percentile.
raise NotImplementedError("Complete the anomaly-threshold challenge.")
```

#### Evidence Table

| Percentile | Threshold | Precision | Recall | F1 | Alerts | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 95 | | | | | | | |
| assigned | | | | | | | |

#### Code Explanation

Why can a high reconstruction-error threshold reduce false positives while increasing false negatives?

#### Interpretation

Choose the more appropriate threshold **for the teaching scenario** and justify it from your measured error trade-off. Remember that anomaly does not mean attack.

## Practice Challenge

Compare thresholds based on:

```text
90th percentile
95th percentile
99th percentile
```

Explain the effect on false positives and false negatives.

## Student Check Before Submission

- All `TODO` sections have been completed.
- The notebook runs from the first cell to the final cell.
- Generated outputs have been checked.
- Challenge questions have been answered using observed results.

## Submission

Submit the notebook, anomaly-score CSV, and threshold comparison.
