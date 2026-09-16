# Week 2 Tutorial — Neural Networks and Forward Propagation — Student Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will transform authentication data into numerical features, build a small neural network, and examine how data moves through the network before training.



> **Version:** Student Version

## Model Used

**Model:** Multilayer Perceptron (MLP)  
**Model category:** Feed-forward neural network

The MLP contains fully connected layers. Each layer applies learned weights, biases, and activation functions to create increasingly useful representations.

## Purpose in the Deep-Learning Workflow

The MLP introduces the core deep-learning computation: tensors move through multiple layers during forward propagation before the network produces a logit.

## Model Input and Output

Input: the transformed authentication feature vector. Output: one logit per session, converted to a probability with sigmoid.

## Important Limitation

An untrained MLP produces random outputs. Fully connected models also ignore special sequence or spatial structure unless that structure is engineered.

## Algorithm to Implement

1. Transform authentication fields into a numerical tensor.
2. Create a first linear layer from the input dimension to 24 hidden units.
3. Apply ReLU.
4. Create a second linear layer from 24 to 12 hidden units and apply ReLU.
5. Create a final linear layer from 12 units to one logit.
6. Run five sessions through the network and apply sigmoid.

The code contains guided `TODO` sections. Complete each section using the algorithm above before running the complete workflow.

## Dataset

```text
C:\BlueWave\data\authentication_events.csv
```

## Steps

1. Open Jupyter Notebook.
2. Create `Week02_Forward_Propagation.ipynb`.
3. Run the code.
4. Inspect the tensor shapes and output probabilities.
5. Remember that the network is not trained yet.

```python
from pathlib import Path
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

DATA_DIR = Path(r"C:\BlueWave\data")

data = pd.read_csv(DATA_DIR / "authentication_events.csv")
train = data[data["split"] == "train"]

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

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), categorical_features)
])

X_train = preprocessor.fit_transform(train[features])
X_tensor = torch.tensor(X_train, dtype=torch.float32)

class AuthenticationMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # TODO 1: Implement input -> 24 -> 12 -> 1.
        # Add ReLU after each hidden linear layer.
        self.network = None

    def forward(self, x):
        # TODO 2: Return the result of the network forward pass.
        raise NotImplementedError("Complete forward propagation.")

model = AuthenticationMLP(X_tensor.shape[1])

sample = X_tensor[:5]

with torch.no_grad():
    logits = model(sample)
    # TODO 3: Convert logits to probabilities.
    probabilities = None

print("Input shape:", sample.shape)
print("Logit shape:", logits.shape)
print("Probabilities:")
print(probabilities)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Trainable parameters:", parameter_count)
```

## Key Explanation

A neuron calculates:

```text
weighted sum + bias → activation
```

The final layer returns a **logit**. The sigmoid function converts the logit into a value between 0 and 1.

The probabilities are random because the model has not been trained.

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



### Challenge — Design a Different MLP Architecture

Your assigned architecture is:

| Variant | Hidden Layer 1 | Hidden Layer 2 |
|---|---:|---:|
| A | 16 | 8 |
| B | 32 | 8 |
| C | 32 | 16 |

Use `STUDENT_SEED` before building the model.

#### Algorithm

1. set the PyTorch random seed;
2. create `Linear(input_size, hidden_1)`;
3. apply ReLU;
4. create `Linear(hidden_1, hidden_2)`;
5. apply ReLU;
6. create `Linear(hidden_2, 1)`;
7. run the same five samples through the untrained network;
8. calculate the number of trainable parameters.

Complete:

```python
torch.manual_seed(STUDENT_SEED)

architecture_map = {
    "A": (16, 8),
    "B": (32, 8),
    "C": (32, 16)
}

hidden_1, hidden_2 = architecture_map[CHALLENGE_VARIANT]

# TODO: build the assigned network.
challenge_model = None

# TODO: calculate trainable parameter count.
challenge_parameter_count = None

# TODO: run the first five samples and convert logits to probabilities.
challenge_probabilities = None
```

#### Evidence Table

| Item | Result |
|---|---|
| Variant | |
| Hidden sizes | |
| Trainable parameters | |
| Probability for sample 1 | |
| Probability for sample 2 | |

#### Code Explanation

Explain why two students using different seeds can obtain different probabilities even when they use the same architecture.

#### Interpretation

Explain why these untrained probabilities must **not** be interpreted as cybersecurity predictions.

## Practice Challenge

Change the architecture to:

```text
input → 16 → 8 → 1
```

Record:

- the new parameter count;
- the output shape;
- whether the untrained probabilities become meaningful.

## Student Check Before Submission

- All `TODO` sections have been completed.
- The notebook runs from the first cell to the final cell.
- Generated outputs have been checked.
- Challenge questions have been answered using observed results.

## Submission

Submit the notebook and a short explanation of weights, biases, ReLU, logits, and probabilities.
