# Week 10 Part B Tutorial — Parallel and Distributed Model Benchmarking — Tutor Version

> **Skillable setup:** The Windows virtual machine already contains Python, Jupyter, pandas, scikit-learn, matplotlib, and PyTorch.  
> All datasets are stored in `C:\BlueWave\data`. Save your work in `C:\BlueWave\outputs`.


## Purpose

You will benchmark one fixed neural network on the Skillable Windows virtual machine and compare batch size, CPU/GPU availability, throughput, and inference latency.



> **Version:** Tutor Version — Complete Implementation

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

## Tutor Teaching Notes

- All benchmark comparisons must use the same workload and training procedure.
- CUDA work is asynchronous, so synchronisation is required for valid timing.
- Larger batches can improve throughput while worsening memory usage or request latency.
- Benchmark results should include environment details and model-quality checks.

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

    start = time.perf_counter()
    processed = 0

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

    elapsed = time.perf_counter() - start

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
        "samples_per_second": processed / elapsed,
        "median_inference_ms": float(
            np.median(inference_times)
        ),
        "p95_inference_ms": float(
            np.percentile(inference_times, 95)
        )
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



### Challenge Solution — Repeated Benchmarking

```python
batch_map = {
    "A": [32, 128, 512],
    "B": [64, 256, 1024],
    "C": [128, 256, 512]
}

assigned_batches = batch_map[CHALLENGE_VARIANT]
challenge_rows = []

for batch_size in assigned_batches:
    repeated_results = [
        run_benchmark(batch_size)
        for _ in range(3)
    ]

    throughputs = [
        item["samples_per_second"]
        for item in repeated_results
    ]

    p95_values = [
        item["p95_inference_ms"]
        for item in repeated_results
    ]

    challenge_rows.append({
        "batch_size": batch_size,
        "run_1_throughput": throughputs[0],
        "run_2_throughput": throughputs[1],
        "run_3_throughput": throughputs[2],
        "median_throughput": float(
            np.median(throughputs)
        ),
        "median_p95_latency_ms": float(
            np.median(p95_values)
        )
    })

challenge_results = pd.DataFrame(challenge_rows)
display(challenge_results)
```

Tutors should check that the student distinguishes throughput from latency and records the actual device/environment. A CPU-only result is valid.

## Practice Challenge

Explain:

- Which batch size gives the highest throughput?
- Which batch size gives the lowest inference latency?
- Why can a faster training configuration still be unsuitable for deployment?
- What should be saved in a recovery checkpoint?

## Suggested Challenge Discussion

- All benchmark comparisons must use the same workload and training procedure.
- CUDA work is asynchronous, so synchronisation is required for valid timing.
- Larger batches can improve throughput while worsening memory usage or request latency.
- Benchmark results should include environment details and model-quality checks.

Tutors should require students to justify conclusions using their own outputs rather than assuming that a more complex model is automatically better.

## Submission

Submit the notebook, benchmark CSV, environment details, and a short scalability and reliability discussion.
