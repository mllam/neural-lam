# Beginner Guide – Neural-LAM

## 1. What is this project?

Neural-LAM is a machine learning project for **weather prediction** using **Graph Neural Networks (GNNs)**.

### Workflow

- **Input** → Weather data (temperature, wind, pressure, etc.)
- **Model** → Graph Neural Network (learns spatial & temporal patterns)
- **Output** → Predicted future weather conditions

---

## 2. Key Technologies

### 🔹 PyTorch + PyTorch Lightning

- PyTorch is used for building deep learning models
- PyTorch Lightning simplifies training by removing boilerplate code

Analogy:

- PyTorch = manual car 🚗
- Lightning = automatic car ⚡

---

### 🔹 Graph Neural Networks (GNNs)

- **Nodes** → Locations (grid points)
- **Edges** → Relationships (wind flow, pressure interaction)

Enables learning spatial dependencies in weather systems.

---

## 3. Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/<your-username>/neural-lam.git
cd neural-lam
```

---

### 2. Create a virtual environment

**Git Bash:**

```
python -m venv venv
source venv/Scripts/activate
```

**Command Prompt:**

```
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

This project uses `pyproject.toml`:

```
pip install .
```

**For development:**

```
pip install -e .
```

---

## 4. Project Structure

```
neural_lam/
│── models/        # GNN model implementations
│── datastore/     # Data loading and preprocessing
│── graph/         # Graph construction
│── utils/         # Helper functions

configs/           # YAML configuration files
```

---

## ▶ 5. Running the Project

```
python train.py --config configs/<config-file>.yaml
```

### Important

- Requires large weather datasets (not included)
- GPU recommended for training
- Config paths must be updated manually

---

## 6. Architecture Overview

```
Raw Weather Data
        ↓
Datastore
        ↓
Graph Construction
        ↓
GNN Model
        ↓
Trainer (Lightning)
        ↓
Predictions
```

---

## 7. Code Flow Walkthrough

1. Load config from `configs/*.yaml`
2. Initialize datastore
3. Build graph structure
4. Initialize model
5. Train using Lightning trainer
6. Log metrics

---

## 8. Common Issues

### Dataset not found

Update dataset path in config

### Installation issues

Use `pip install -e .`

### Training slow or fails

Use smaller configs or GPU

---

## 9. Contribution Tips

- Improve documentation
- Add visualization tools
- Improve error handling

---
