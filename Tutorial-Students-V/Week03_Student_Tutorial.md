# Week 3 Tutorial — Loss, Backpropagation, and Optimisation — Student Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will train the Week 2 neural network using binary cross-entropy loss, backpropagation, and the Adam optimiser.



> **Version:** Student Version

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

## Algorithm to Implement

1. Perform a forward pass on the training tensor.
2. Calculate BCEWithLogitsLoss.
3. Clear old gradients.
4. Run backpropagation.
5. Update parameters with Adam.
6. Evaluate validation loss without calculating gradients.
7. Repeat for all epochs and save the learning history.

The code contains guided `TODO` sections. Complete each section using the algorithm above before running the complete workflow.

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

    # TODO 1: Complete one training update.
    # 1. Run the forward pass.
    # 2. Calculate BCEWithLogitsLoss.
    # 3. Clear previous gradients.
    # 4. Backpropagate.
    # 5. Update model parameters.
    raise NotImplementedError("Complete the training update.")

    model.eval()

    with torch.no_grad():
        # TODO 2: Run validation forward propagation
        # and calculate validation loss.
        raise NotImplementedError("Complete validation evaluation.")

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



### Challenge — Learning-Rate Experiment

Your assigned alternative learning rate is:

| Variant | Alternative learning rate |
|---|---:|
| A | 0.0005 |
| B | 0.002 |
| C | 0.005 |

Compare the assigned learning rate with the tutorial baseline `0.001`. Use `STUDENT_SEED` for both runs.

#### Algorithm

For each learning rate:

1. rebuild the same MLP from the same seed;
2. train for 100 epochs;
3. record the minimum validation loss;
4. record the epoch where minimum validation loss occurs;
5. record final validation loss;
6. save both validation-loss curves on one figure.

Complete a helper function:

```python
def run_learning_rate_experiment(learning_rate):
    torch.manual_seed(STUDENT_SEED)

    # TODO: rebuild the same MLP.
    model = None

    # TODO: create BCEWithLogitsLoss and Adam.
    loss_function = None
    optimiser = None

    history = []

    # TODO: train for 100 epochs and append
    # epoch, training_loss and validation_loss.
    raise NotImplementedError("Complete the learning-rate experiment.")
```

#### Evidence Table

| Learning Rate | Minimum Validation Loss | Epoch of Minimum | Final Validation Loss |
|---:|---:|---:|---:|
| 0.001 | | | |
| assigned | | | |

#### Code Explanation

Why must you rebuild the model using the same seed before comparing learning rates?

#### Interpretation

Use the observed curves and values to state which learning rate learned more effectively in your run.

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

## Student Check Before Submission

- All `TODO` sections have been completed.
- The notebook runs from the first cell to the final cell.
- Generated outputs have been checked.
- Challenge questions have been answered using observed results.

## Submission

Submit the notebook, loss curve, training history, and a short optimisation comparison.
