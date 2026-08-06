# Deep-Learning-for-Cyber-Security-ZSPS2121

# BlueWave Deep-Learning Tutorials

This package contains two Markdown files for each tutorial:

- **Student Version:** model explanation, implementation algorithm, guided TODO code, challenge, and submission requirements.
- **Tutor Version:** complete executable code, teaching notes, and suggested challenge discussion.

## Skillable Paths

```text
Datasets: C:\BlueWave\data
Outputs:  C:\BlueWave\outputs
```

## Models Used

| Week | Model | Role in the course |
|---|---|---|
| Week 1 | Logistic Regression Baseline | A deep-learning project needs a credible baseline. The baseline shows whether later neural networks provide a real improvement over a simpler model. |
| Week 2 | Multilayer Perceptron (MLP) | The MLP introduces the core deep-learning computation: tensors move through multiple layers during forward propagation before the network produces a logit. |
| Week 3 | Trained Authentication MLP | This tutorial demonstrates the complete learning cycle that turns a randomly initialised network into a model fitted to cybersecurity data. |
| Week 4 | Regularised MLP with Early Model Selection | Deep models can fit training data without generalising. Regularisation, validation selection, and threshold selection create a more defensible final model. |
| Week 5 | One-Dimensional Convolutional Neural Network | CNNs are useful when nearby values form meaningful local patterns. In cybersecurity, they can model byte sequences, packet fields, or short telemetry windows. |
| Week 7 | Transformer Encoder Classifier | Transformers model ordered cybersecurity telemetry and can capture relationships between events that are separated by several positions. |
| Week 8 | Autoencoder | When reliable attack labels are limited, a model can learn normal behaviour. Large reconstruction error then becomes an anomaly score. |
| Week 9 | Logistic Stacking Fusion Model | Real deep-learning systems often contain several specialised models. A fusion layer creates one session-level decision while preserving missing-evidence information. |
| Week 10 Part B | Benchmark MLP | Deep-learning deployment requires evidence about efficiency, scalability, and reliability—not only predictive accuracy. |

There is intentionally no Week 6 tutorial.

## Tutor Use

Tutors may demonstrate the complete code from the Tutor Version, but students should submit work based on the Student Version and complete the guided TODO sections themselves.
