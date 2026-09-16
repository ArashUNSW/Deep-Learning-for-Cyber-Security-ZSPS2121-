# Week 5 Tutorial — CNN for Synthetic File-Byte Sequences — Tutor Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will train a one-dimensional convolutional neural network to classify safe synthetic byte-like sequences.



> **Version:** Tutor Version — Complete Implementation

## Model Used

**Model:** One-Dimensional Convolutional Neural Network  
**Model category:** Local-pattern sequence model

A 1D CNN slides learned filters across byte positions. Weight sharing allows the same local pattern to be recognised at different locations.

## Purpose in the Deep-Learning Workflow

CNNs are useful when nearby values form meaningful local patterns. In cybersecurity, they can model byte sequences, packet fields, or short telemetry windows.

## Model Input and Output

Input: a tensor shaped batch × channel × 128 byte positions. Output: one suspicious-artefact logit and probability per synthetic artefact.

## Important Limitation

The synthetic byte patterns are not real malware semantics. A CNN may also learn dataset shortcuts and can miss long-range relationships.

## Tutor Teaching Notes

- Conv1d input shape is batch × channels × sequence length.
- Pooling reduces sequence length and summarises local activations.
- AdaptiveMaxPool1d(1) produces one value per output channel.
- Kernel size controls the width of local context examined by each filter.

## Dataset

```text
C:\BlueWave\data\file_byte_sequences.csv
```

The dataset contains 128 columns:

```text
byte_000 to byte_127
```

## Steps

1. Create `Week05_CNN_Byte_Sequences.ipynb`.
2. Load and normalise the byte values.
3. Convert the data to Conv1d tensor format.
4. Train the CNN.
5. Evaluate the test set.

```python
from pathlib import Path
import pandas as pd
import torch

from sklearn.metrics import classification_report
from torch import nn

DATA_DIR = Path(r"C:\BlueWave\data")
OUTPUT_DIR = Path(r"C:\BlueWave\outputs\week05")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

data = pd.read_csv(DATA_DIR / "file_byte_sequences.csv")
byte_columns = [
    column
    for column in data.columns
    if column.startswith("byte_")
]

train = data[data["split"] == "train"]
validation = data[data["split"] == "validation"]
test = data[data["split"] == "test"]

def make_tensor(frame):
    values = (
        frame[byte_columns]
        .to_numpy()
        .astype("float32")
        / 255.0
    )
    return torch.tensor(
        values[:, None, :],
        dtype=torch.float32
    )

X_train = make_tensor(train)
X_validation = make_tensor(validation)
X_test = make_tensor(test)

y_train = torch.tensor(
    train["label_suspicious_artifact"]
    .to_numpy()
    .reshape(-1, 1),
    dtype=torch.float32
)

y_validation = torch.tensor(
    validation["label_suspicious_artifact"]
    .to_numpy()
    .reshape(-1, 1),
    dtype=torch.float32
)

y_test = test[
    "label_suspicious_artifact"
].to_numpy()

model = nn.Sequential(
    nn.Conv1d(1, 16, kernel_size=7, padding=3),
    nn.ReLU(),
    nn.MaxPool1d(2),
    nn.Conv1d(16, 32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AdaptiveMaxPool1d(1),
    nn.Flatten(),
    nn.Linear(32, 1)
)

loss_function = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    loss = loss_function(model(X_train), y_train)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

model.eval()

with torch.no_grad():
    test_probability = torch.sigmoid(
        model(X_test)
    ).numpy().ravel()

test_prediction = (test_probability >= 0.50).astype(int)

print(classification_report(
    y_test,
    test_prediction,
    digits=3
))

pd.DataFrame({
    "artifact_id": test["artifact_id"],
    "session_id": test["session_id"],
    "true_label": y_test,
    "probability": test_probability,
    "prediction": test_prediction
}).to_csv(
    OUTPUT_DIR / "week05_artifact_predictions.csv",
    index=False
)

torch.save(
    model.state_dict(),
    OUTPUT_DIR / "week05_cnn_model.pt"
)
```

## Safety Note

The sequences are synthetic. They are not executable malware and must not be interpreted as real malicious files.

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



### Challenge Solution — CNN Kernel Size

```python
assigned_kernel = {
    "A": 3,
    "B": 9,
    "C": 15
}[CHALLENGE_VARIANT]

def build_challenge_cnn(kernel_size):
    padding = kernel_size // 2

    return nn.Sequential(
        nn.Conv1d(
            1, 16,
            kernel_size=kernel_size,
            padding=padding
        ),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(
            16, 32,
            kernel_size=5,
            padding=2
        ),
        nn.ReLU(),
        nn.AdaptiveMaxPool1d(1),
        nn.Flatten(),
        nn.Linear(32, 1)
    )
```

Tutors should require students to train both configurations under matching conditions and report measured parameter counts and F1. The result is dataset-specific.

## Practice Challenge

Change the first convolution kernel size to:

```text
3
7
15
```

Compare test F1-score and explain how kernel size changes the local pattern examined by the CNN.

## Suggested Challenge Discussion

- Conv1d input shape is batch × channels × sequence length.
- Pooling reduces sequence length and summarises local activations.
- AdaptiveMaxPool1d(1) produces one value per output channel.
- Kernel size controls the width of local context examined by each filter.

Tutors should require students to justify conclusions using their own outputs rather than assuming that a more complex model is automatically better.

## Submission

Submit the notebook, prediction CSV, model file, and kernel-size comparison.
