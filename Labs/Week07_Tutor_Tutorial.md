# Week 7 Tutorial — Transformer for Security Event Sequences — Tutor Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will reconstruct ordered event sequences and train a small transformer classifier.



> **Version:** Tutor Version — Complete Implementation

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

## Tutor Teaching Notes

- The vocabulary contains special PAD, UNK, and CLS tokens.
- Positional embeddings are necessary because self-attention alone is order-agnostic.
- The CLS representation summarises the sequence for classification.
- Attention visualisation must be combined with other explanation tests.

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
    return [
        CLS_ID
    ] + [
        event_to_id.get(token, UNK_ID)
        for token in tokens
    ]

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

        representation = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

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
    loss = loss_function(
        model(X_train),
        y_train
    )

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



### Challenge Solution — Transformer Ablation

Tutors should accept equivalent configurable implementations. The intended single-variable changes are:

```python
# Variant A:
# representation = self.token_embedding(token_ids)
# No position_embedding addition.

# Variant B:
encoder_layer = nn.TransformerEncoderLayer(
    d_model=32,
    nhead=2,
    dim_feedforward=64,
    batch_first=True
)

# Variant C:
self.encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=1
)
```

The baseline and ablation must otherwise use the same data, seed, optimiser, epoch count, and decision threshold. Students must not claim that attention values prove causal importance.

## Practice Challenge

Train a second transformer without positional embeddings.

Explain:

- Does performance change?
- Why does a transformer need position information?
- Why is an attention score not proof that an event caused the decision?

## Suggested Challenge Discussion

- The vocabulary contains special PAD, UNK, and CLS tokens.
- Positional embeddings are necessary because self-attention alone is order-agnostic.
- The CLS representation summarises the sequence for classification.
- Attention visualisation must be combined with other explanation tests.

Tutors should require students to justify conclusions using their own outputs rather than assuming that a more complex model is automatically better.

## Submission

Submit the notebook, sequence prediction CSV, and positional-embedding comparison.
