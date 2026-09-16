# Week 10 Part B Tutorial — Parallel and Distributed Model Benchmarking — Student Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will benchmark one fixed neural network on the Skillable Windows virtual machine and compare batch size, CPU/GPU availability, throughput, and inference latency.



> **Version:** Student Version

## Model Used

**Model:** Benchmark MLP  
**Model category:** Performance and deployment benchmark

A fixed MLP provides a controlled workload for measuring training throughput and inference latency across batch sizes and available devices.

## Purpose in the Deep-Learning Workflow

Deep-learning deployment requires evidence about efficiency, scalability, and reliability—not only predictive accuracy.

## Model Input and Output

Input: seven network-flow features. Output: two classification logits, benchmark throughput, and latency measurements.

## Important Limitation

Results apply only to the recorded VM, software, model, data, and batch settings. A CPU benchmark does not establish GPU or distributed scalability.

## Algorithm to Implement

1. Record the operating system, PyTorch version, and device availability.
2. Create one fixed MLP and dataset.
3. Start a timer before the measured training loop.
4. Train for the same number of epochs for each batch size.
5. Synchronise the GPU before stopping timers when CUDA is used.
6. Calculate samples processed per second.
7. Measure repeated inference latency and calculate median and p95.

The code contains guided `TODO` sections. Complete each section using the algorithm above before running the complete workflow.

## Dataset

```text
C:\BlueWave\data\network_flows.csv
```

## Steps

1. Create `Week10_Parallel_Benchmark.ipynb`.
2. Record the hardware and software environment.
3. Train the same model using different batch sizes.
4. Compare elapsed time and throughput.
5. Measure inference latency.
6. Save the benchmark table.

```python
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(r"C:\BlueWave\data")
OUTPUT_DIR = Path(r"C:\BlueWave\outputs\week10")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

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

scaler = StandardScaler()

X_train = torch.tensor(
    scaler.fit_transform(train[features]),
    dtype=torch.float32
)

y_train = torch.tensor(
    train["label_anomaly"].to_numpy(),
    dtype=torch.long
)

X_validation = torch.tensor(
    scaler.transform(validation[features]),
    dtype=torch.float32
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = TensorDataset(X_train, y_train)

def run_benchmark(batch_size):
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(7, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    loss_function = nn.CrossEntropyLoss()

    # TODO 1: Record the start time and initialise
    # the processed-sample counter.
    start = None
    processed = None

    for _ in range(10):
        model.train()

        for features_batch, labels_batch in loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            loss = loss_function(
                model(features_batch),
                labels_batch
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            processed += len(features_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # TODO 2: Calculate measured elapsed seconds.
    elapsed = None

    model.eval()
    sample = X_validation[:64].to(device)

    inference_times = []

    with torch.no_grad():
        for _ in range(50):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_inference = time.perf_counter()
            _ = model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_times.append(
                (time.perf_counter() - start_inference) * 1000
            )

    return {
        "device": str(device),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        # TODO 3: Calculate throughput, median latency,
        # and 95th-percentile latency.
        "samples_per_second": None,
        "median_inference_ms": None,
        "p95_inference_ms": None
    }

benchmark = pd.DataFrame([
    run_benchmark(64),
    run_benchmark(256),
    run_benchmark(1024)
])

print(benchmark)

benchmark.to_csv(
    OUTPUT_DIR / "week10_benchmark_results.csv",
    index=False
)
```

## Interpretation

A larger batch may improve throughput but increase memory use and latency.

A CPU result is valid. Do not claim GPU or multi-GPU performance when the VM does not provide those resources.

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



### Challenge — Repeated Performance Benchmark

Your assigned batch sizes are:

| Variant | Batch sizes |
|---|---|
| A | 32, 128, 512 |
| B | 64, 256, 1024 |
| C | 128, 256, 512 |

Run each configuration **three times**. Do not report only the fastest run.

#### Algorithm

For each batch size:

1. run `run_benchmark()` three times;
2. record throughput for all three runs;
3. calculate median throughput;
4. record p95 inference latency from each run;
5. calculate median p95 latency;
6. identify the batch size with best throughput and the batch size with best latency.

Complete:

```python
batch_map = {
    "A": [32, 128, 512],
    "B": [64, 256, 1024],
    "C": [128, 256, 512]
}

assigned_batches = batch_map[CHALLENGE_VARIANT]

challenge_rows = []

for batch_size in assigned_batches:
    # TODO: run three repeated benchmarks.
    # TODO: calculate median throughput.
    # TODO: calculate median p95 latency.
    raise NotImplementedError("Complete repeated benchmarking.")
```

#### Evidence Table

| Batch Size | Run 1 Throughput | Run 2 | Run 3 | Median Throughput | Median p95 Latency |
|---:|---:|---:|---:|---:|---:|
| assigned | | | | | |
| assigned | | | | | |
| assigned | | | | | |

#### Code Explanation

Why is reporting the fastest run only an unreliable benchmarking practice?

#### Interpretation

Recommend one batch size for **training throughput** and one for **interactive inference**, using your measured results. They may be different.

## Practice Challenge

Explain:

- Which batch size gives the highest throughput?
- Which batch size gives the lowest inference latency?
- Why can a faster training configuration still be unsuitable for deployment?
- What should be saved in a recovery checkpoint?

## Student Check Before Submission

- All `TODO` sections have been completed.
- The notebook runs from the first cell to the final cell.
- Generated outputs have been checked.
- Challenge questions have been answered using observed results.

## Submission

Submit the notebook, benchmark CSV, environment details, and a short scalability and reliability discussion.
