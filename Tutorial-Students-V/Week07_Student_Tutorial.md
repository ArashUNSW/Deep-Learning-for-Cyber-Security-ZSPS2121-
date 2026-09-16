# Week 7 Tutorial — Transformer for Security Event Sequences — Student Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will reconstruct ordered event sequences and train a small transformer classifier.



> **Version:** Student Version

## Model Used

**Model:** Transformer Encoder Classifier  
**Model category:** Attention-based sequence model

The transformer converts event tokens into embeddings, adds positional information, and uses self-attention to compare events across the sequence.

## Purpose in the Deep-Learning Workflow

Transformers model ordered cybersecurity telemetry and can capture relationships between events that are separated by several positions.

## Model Input and Output

Input: an ordered sequence of event-token identifiers beginning with a CLS token. Output: one sequence logit from the encoded CLS representation.

## Important Limitation

Attention weights are not causal explanations. Small datasets may not justify transformer complexity, so simpler unigram or bigram baselines remain important.

## Algorithm to Implement

1. Sort event rows by sequence and event position.
2. Build a vocabulary using training sequences only.
3. Encode each sequence with CLS and token identifiers.
4. Create token embeddings and position embeddings.
5. Pass representations through transformer encoder layers.
6. Use the encoded CLS vector for binary classification.
7. Train with BCEWithLogitsLoss and evaluate the test sequences.

The code contains guided `TODO` sections. Complete each section using the algorithm above before running the complete workflow.

## Dataset

```text
C:\BlueWave\data\security_event_sequences.csv
```

Each sequence contains 18 ordered events.

## Steps

1. Create `Week07_Transformer.ipynb`.
2. Group events by `sequence_id`.
3. Build a vocabulary from training events.
4. Add a classification token.
5. Train a transformer encoder.
6. Evaluate the test split.

```python
from pathlib import Path
import pandas as pd
import torch

from sklearn.metrics import classification_report
from torch import nn

DATA_DIR = Path(r"C:\BlueWave\data")
OUTPUT_DIR = Path(r"C:\BlueWave\outputs\week07")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

events = pd.read_csv(
    DATA_DIR / "security_event_sequences.csv"
)

sequence_table = (
    events.sort_values(
        ["sequence_id", "event_position"]
    )
    .groupby("sequence_id", as_index=False)
    .agg(
        session_id=("session_id", "first"),
        split=("split", "first"),
        label=("label_suspicious_sequence", "first"),
        tokens=("event_type", list)
    )
)

train = sequence_table[sequence_table["split"] == "train"]
validation = sequence_table[
    sequence_table["split"] == "validation"
]
test = sequence_table[sequence_table["split"] == "test"]

special_tokens = ["<PAD>", "<UNK>", "<CLS>"]

event_types = sorted({
    token
    for sequence in train["tokens"]
    for token in sequence
})

event_to_id = {
    token: index
    for index, token in enumerate(
        special_tokens + event_types
    )
}

PAD_ID = event_to_id["<PAD>"]
UNK_ID = event_to_id["<UNK>"]
CLS_ID = event_to_id["<CLS>"]

def encode(tokens):
    # TODO 1: Start with CLS_ID and map every event token.
    # Unknown tokens must use UNK_ID.
    raise NotImplementedError("Encode the event sequence.")

def make_tensor(frame):
    return torch.tensor(
        [encode(tokens) for tokens in frame["tokens"]],
        dtype=torch.long
    )

X_train = make_tensor(train)
X_validation = make_tensor(validation)
X_test = make_tensor(test)

y_train = torch.tensor(
    train["label"].to_numpy().reshape(-1, 1),
    dtype=torch.float32
)

y_test = test["label"].to_numpy()

class EventTransformer(nn.Module):
    def __init__(self, vocabulary_size, sequence_length):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            32,
            padding_idx=PAD_ID
        )

        self.position_embedding = nn.Embedding(
            sequence_length,
            32
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Linear(32, 1)

    def forward(self, token_ids):
        positions = torch.arange(
            token_ids.shape[1]
        ).unsqueeze(0)

        # TODO 2: Add token and positional embeddings.
        representation = None

        encoded = self.encoder(representation)
        return self.classifier(encoded[:, 0, :])

model = EventTransformer(
    len(event_to_id),
    X_train.shape[1]
)

loss_function = nn.BCEWithLogitsLoss()
optimiser = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

for epoch in range(100):
    model.train()
    # TODO 3: Complete transformer training:
    # forward pass -> loss -> zero gradients
    # -> backward -> optimiser step.
    raise NotImplementedError("Complete transformer training.")

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
    "sequence_id": test["sequence_id"],
    "session_id": test["session_id"],
    "true_label": y_test,
    "probability": test_probability,
    "prediction": test_prediction
}).to_csv(
    OUTPUT_DIR / "week07_sequence_predictions.csv",
    index=False
)
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



### Challenge — Transformer Architecture Ablation

Your assigned change is:

| Variant | Change |
|---|---|
| A | Remove positional embeddings |
| B | Use `nhead=2` instead of 4 |
| C | Use one encoder layer instead of 2 |

Compare the assigned transformer with the tutorial transformer. Use `STUDENT_SEED`.

#### Algorithm

1. preserve the same training/test sequences;
2. change only the assigned architecture element;
3. train baseline and challenge models with the same seed;
4. calculate test F1;
5. record trainable parameter count;
6. explain what capability the changed component provides.

#### Student Implementation

Create a second class or modify the existing class through configuration.

```python
# TODO: create a challenge transformer that changes
# only the element assigned to your variant.
challenge_model = None

# TODO: train it using the same training algorithm.
# TODO: calculate test probability, prediction and F1.
raise NotImplementedError("Complete the transformer ablation.")
```

#### Evidence Table

| Model | Parameters | Test F1 |
|---|---:|---:|
| Tutorial transformer | | |
| Assigned ablation | | |

#### Code Explanation

Explain the role of the component you changed.

#### Interpretation

Use your observed values to state whether the extra transformer complexity was useful for this synthetic task.

## Practice Challenge

Train a second transformer without positional embeddings.

Explain:

- Does performance change?
- Why does a transformer need position information?
- Why is an attention score not proof that an event caused the decision?

## Student Check Before Submission

- All `TODO` sections have been completed.
- The notebook runs from the first cell to the final cell.
- Generated outputs have been checked.
- Challenge questions have been answered using observed results.

## Submission

Submit the notebook, sequence prediction CSV, and positional-embedding comparison.
