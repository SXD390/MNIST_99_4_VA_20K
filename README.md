
# 🧠 MNIST Digit Classifier — ERA Assignment

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![Device](https://img.shields.io/badge/Device-MPS%20|%20CUDA%20|%20CPU-blue)]()
[![Dataset](https://img.shields.io/badge/Dataset-MNIST-28x28-success)](http://yann.lecun.com/exdb/mnist/)

---

## 📌 Project Overview

This project explores **lightweight CNN architectures for MNIST** under the following constraints:

* ✅ **<20K parameters**
* ✅ **BatchNorm & Dropout used**
* ✅ **Trained <20 epochs**
* ✅ **Validation/Test Accuracy ≥ 99.4%**

Final result: **99.6% test accuracy** with **14.5K parameters** in just **15 epochs** 🎯

---

## 📊 Model at a Glance

* **Final Architecture:** Conv-BN-ReLU blocks + 1×1 transition layers + Dropout + GAP head
* **Parameters:** \~14.5K
* **Accuracy:** 99.6% (test)
* **Head:** Global Average Pooling + 1×1 Conv classifier
* **BatchNorm:** Yes (after each conv)
* **Dropout:** Yes (p=0.05 after each block)
* **Epochs:** 15
* **Scheduler:** OneCycleLR (per batch)

---

## 🏗️ Final Network Flow (Mermaid Diagram)

```mermaid
graph TD
  A[Input 1×28×28] --> B[Conv 3×3 1→16]
  B --> C[BN+ReLU]
  C --> D[Conv 3×3 16→16]
  D --> E[BN+ReLU]
  E --> F[MaxPool 2×2 28→14]
  F --> G[1×1 Conv 16→12]
  G --> H[Dropout 0.05]

  H --> I[Conv 3×3 12→28]
  I --> J[BN+ReLU]
  J --> K[Conv 3×3 28→28]
  K --> L[BN+ReLU]
  L --> M[MaxPool 2×2 14→7]
  M --> N[1×1 Conv 28→32]
  N --> O[Dropout 0.05]

  O --> P[GAP 7×7→1×1]
  P --> Q[1×1 Conv 32→10]
  Q --> R[Softmax (via CrossEntropy)]
```

---

## 📐 Layerwise Shape & Channels

| Layer  | In C×H×W | Kernel | Out C×H×W | Notes           |
| ------ | -------- | ------ | --------- | --------------- |
| Input  | 1×28×28  | –      | 1×28×28   | grayscale digit |
| Conv1  | 1×28×28  | 3×3    | 16×28×28  | edge detectors  |
| Conv2  | 16×28×28 | 3×3    | 16×28×28  | strokes/corners |
| Pool1  | 16×28×28 | 2×2    | 16×14×14  | downsample      |
| Trans1 | 16×14×14 | 1×1    | 12×14×14  | channel reduce  |
| Conv3  | 12×14×14 | 3×3    | 28×14×14  | digit parts     |
| Conv4  | 28×14×14 | 3×3    | 28×14×14  | refine features |
| Pool2  | 28×14×14 | 2×2    | 28×7×7    | downsample      |
| Trans2 | 28×7×7   | 1×1    | 32×7×7    | channel bump    |
| GAP    | 32×7×7   | –      | 32×1×1    | global summary  |
| Class  | 32×1×1   | 1×1    | 10×1×1    | logits          |

---

## 🔁 Change Log (Iterations)

| Iter | Params | Architecture Snapshot                          | Epochs | LR / Scheduler | Train Acc | Test Acc | What changed & why                                              |
| ---: | -----: | ---------------------------------------------- | -----: | -------------- | --------- | -------- | --------------------------------------------------------------- |
|   I0 | 18,060 | Tiny CNN, 2×Conv→Pool→2×Conv→Pool, FC(1176→10) |      1 | 0.1 / StepLR   | 93.0%     | 98.0%    | Baseline; simple & fast but underfits a bit                     |
|   I1 | 20,978 | +BN,+Dropout; same FC head                     |     10 | 0.2 / OneCycle | 99.5%     | 99.6%    | BN stabilizes; Dropout regularizes; OneCycle speeds convergence |
|   I2 | 19,494 | +1×1 transitions after pools                   |     12 | 0.2 / OneCycle | 99.6%     | 99.6%    | Channel compression improves parameter efficiency               |
|   I3 | 14,500 | GAP head (GAP→1×1 32→10), channels tuned       |     15 | 0.2 / OneCycle | 99.7%     | 99.6%    | Replace heavy FC with GAP; spend params on features             |

---

## 📈 Training Curves

*(Insert the loss/accuracy plots generated from your notebook here.)*

---

## ⚡ Why the Improvements Worked

* **BatchNorm**: stabilized training, allowed higher learning rates.
* **Dropout**: improved generalization by preventing overfitting.
* **1×1 Convs (Transition layers)**: reduced parameter count while keeping depth.
* **GAP head**: replaced heavy FC, saving \~9K params; freed capacity for better feature maps.
* **OneCycleLR**: aggressive learning rate scheduling → faster, smoother convergence.

---

## ▶️ How to Run

```bash
# clone repo
git clone https://github.com/yourusername/MNIST-ERA-Assignment.git
cd MNIST-ERA-Assignment

# create environment (example with uv)
uv venv venv
source venv/bin/activate

# install deps
pip install -r requirements.txt

# run notebook
jupyter notebook MNIST_U20K_99_4P.ipynb
```

---

## ✅ Compliance Checklist

* [x] <20K parameters (final \~14.5K)
* [x] BatchNorm used
* [x] Dropout used
* [x] GAP head (instead of FC)
* [x] <20 epochs (final run = 15)
* [x] Test accuracy ≥99.4% (achieved 99.6%)

---
