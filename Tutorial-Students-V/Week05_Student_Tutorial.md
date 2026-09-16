# Week 5 Tutorial — CNN for Synthetic File-Byte Sequences — Student Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will train a one-dimensional convolutional neural network to classify safe synthetic byte-like sequences.



> **Version:** Student Version

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

## Algorithm to Implement

1. Scale byte values from 0–255 to 0–1.
2. Add a channel dimension for Conv1d.
3. Apply convolution, ReLU, and pooling.
4. Apply a second convolution and global pooling.
5. Flatten the representation and produce one logit.
6. Train with BCEWithLogitsLoss and evaluate held-out artefacts.

The code contains guided `TODO` sections. Complete each section using the algorithm above before running the complete workflow.

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
    # TODO 1: Add the Conv1d channel dimension.
    # Required shape: rows x 1 x 128.
    raise NotImplementedError("Create the Conv1d tensor.")

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

# TODO 2: Implement the CNN algorithm:
# Conv1d(1,16,kernel=7) -> ReLU -> MaxPool
# -> Conv1d(16,32,kernel=5) -> ReLU
# -> AdaptiveMaxPool1d(1) -> Flatten -> Linear(32,1)
model = None

loss_function = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    # TODO 3: Complete forward pass, loss,
    # gradient clearing, backpropagation, and optimiser update.
    raise NotImplementedError("Complete CNN training.")

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



### Challenge — CNN Kernel Size

Your assigned first-layer kernel size is:

| Variant | Kernel size |
|---|---:|
| A | 3 |
| B | 9 |
| C | 15 |

Compare the assigned kernel with the tutorial kernel size `7`. Use `STUDENT_SEED`.

#### Algorithm

1. rebuild the CNN with the assigned first kernel;
2. choose padding so sequence length is preserved before pooling;
3. train the baseline and assigned model using the same seed;
4. calculate held-out F1;
5. calculate parameter count;
6. compare the local context represented by each kernel.

Complete:

```python
assigned_kernel = {
    "A": 3,
    "B": 9,
    "C": 15
}[CHALLENGE_VARIANT]

def build_challenge_cnn(kernel_size):
    padding = kernel_size // 2

    # TODO: implement the same CNN but replace
    # the first Conv1d kernel size and padding.
    raise NotImplementedError("Build the challenge CNN.")
```

#### Evidence Table

| Kernel | Trainable Parameters | Test F1 |
|---:|---:|---:|
| 7 | | |
| assigned | | |

#### Code Explanation

What does increasing the convolution kernel size change about the local byte context inspected by the first layer?

#### Interpretation

State whether the larger/smaller local context helped on your synthetic test data. Do not generalise the result to real malware.

## Practice Challenge

Change the first convolution kernel size to:

```text
3
7
15
```

Compare test F1-score and explain how kernel size changes the local pattern examined by the CNN.

## Student Check Before Submission

- All `TODO` sections have been completed.
- The notebook runs from the first cell to the final cell.
- Generated outputs have been checked.
- Challenge questions have been answered using observed results.

## Submission

Submit the notebook, prediction CSV, model file, and kernel-size comparison.
