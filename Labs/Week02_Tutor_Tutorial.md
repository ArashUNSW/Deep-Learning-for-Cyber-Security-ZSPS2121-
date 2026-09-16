# Week 2 Tutorial — Neural Networks and Forward Propagation — Tutor Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will transform authentication data into numerical features, build a small neural network, and examine how data moves through the network before training.



> **Version:** Tutor Version — Complete Implementation

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

## Tutor Teaching Notes

- The architecture is input → 24 → 12 → 1.
- ReLU adds non-linearity; without it, stacked linear layers remain linear.
- The final layer returns a logit rather than a probability.
- Parameter count depends on layer dimensions, not batch size.

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
        self.network = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        )

    def forward(self, x):
        return self.network(x)

model = AuthenticationMLP(X_tensor.shape[1])

sample = X_tensor[:5]

with torch.no_grad():
    logits = model(sample)
    probabilities = torch.sigmoid(logits)

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



### Challenge Solution — MLP Architecture

```python
torch.manual_seed(STUDENT_SEED)

architecture_map = {
    "A": (16, 8),
    "B": (32, 8),
    "C": (32, 16)
}

hidden_1, hidden_2 = architecture_map[CHALLENGE_VARIANT]

challenge_model = nn.Sequential(
    nn.Linear(X_tensor.shape[1], hidden_1),
    nn.ReLU(),
    nn.Linear(hidden_1, hidden_2),
    nn.ReLU(),
    nn.Linear(hidden_2, 1)
)

challenge_parameter_count = sum(
    parameter.numel()
    for parameter in challenge_model.parameters()
    if parameter.requires_grad
)

with torch.no_grad():
    challenge_logits = challenge_model(X_tensor[:5])
    challenge_probabilities = torch.sigmoid(
        challenge_logits
    )

print("Parameters:", challenge_parameter_count)
print(challenge_probabilities)
```

**Expected reasoning:** the seed controls random initial weights. The network is untrained, so different initialisations produce different outputs; none establish predictive quality.

## Practice Challenge

Change the architecture to:

```text
input → 16 → 8 → 1
```

Record:

- the new parameter count;
- the output shape;
- whether the untrained probabilities become meaningful.

## Suggested Challenge Discussion

- The architecture is input → 24 → 12 → 1.
- ReLU adds non-linearity; without it, stacked linear layers remain linear.
- The final layer returns a logit rather than a probability.
- Parameter count depends on layer dimensions, not batch size.

Tutors should require students to justify conclusions using their own outputs rather than assuming that a more complex model is automatically better.

## Submission

Submit the notebook and a short explanation of weights, biases, ReLU, logits, and probabilities.
