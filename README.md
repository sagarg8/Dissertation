# 🏎️ Enhancing Used Car Price Prediction with Multi-Modal Deep Learning
### The "Mechanic and the Bookkeeper" Approach

This repository contains the code for a machine learning dissertation focused on predicting used car prices using both tabular metadata (**The Bookkeeper**) and visual data (**The Mechanic**). 

By using a **Late Fusion** architecture, this model combines a Deep Learning vision branch (ResNet-50) with a dense neural network to achieve significantly higher accuracy than industry-standard tabular models

## 📊 The Results
The project compared a heavily optimised **XGBoost** baseline (hyperparameter-tuned) against the **PyTorch Multimodal Fusion** network on an identical test set.

| Model | MAE | RMSE | $R^2$ |
| :--- | :--- | :--- | :--- |
| **XGBoost (Tabular Only)** | £1,664.63 | £2,440.51 | 0.9004 |
| **PyTorch Fusion (Visual + Tabular)** | £1,279.47 | £1,890.44 | 0.9403 |

> **Conclusion:** The Multimodal approach reduced prediction error (RMSE) by **£550 per vehicle**, proving that visual cues are a statistically significant factor in vehicle valuation.

---

## 🔬 Explainable AI & Failure Analysis
To ensure the model was learning meaningful features rather than noise, two interpretability studies were conducted:

### 1. Saliency Mapping (Grad-CAM)
Using **Gradient-weighted Class Activation Mapping**, I visualised the ResNet-50's focus points.
* **Discovery:** The model successfully identified high-value "Approved Used" dealer plates and aerodynamic trim packages on luxury models (BMW M, Porsche).
* **Bias Note:** The model learned to use main-dealer branding as a visual proxy for premium pricing.

### 2. Ablation Study
I systematically "blindfolded" the model to measure the importance of each data stream.

| Condition | RMSE | Penalty |
| :--- | :--- | :--- |
| **Full Multimodal Fusion** | £1,890 | -- |
| **No Vision (Tabular Only)** | £2,145 | +£255 |
| **No Tabular (Vision Only)** | £7,546 | +£5,656 |

---

## 📂 Project Structure
```text
6003CMD_Dissertation/
├── data/
│   ├── processed/          # Cleaned CSVs, test indices, and baseline preds
├── project_documents/      # Contains draft documents, dev logs, and supervisor notes
├── models/                 # Saved model weights (.pth)
├── results/                
│   ├── case_studies/       # Top & Bottom 5 PyTorch victories (visual evidence)
│   └── heatmaps/           # Grad-CAM saliency visualisations
├── scripts/                # Execution scripts (The "Workflow")
├── src/                    # Core Architecture (Model & Dataset classes)
└── requirements.txt        # Python dependencies
```

---

## 🚀 Usage (Execution Order)
To replicate the results, execute the scripts in the following order:

1.  **Clean the data:**
2.  ```bash
    python scripts/data_cleaning.py
    ```
3.  **Train the Multimodal Network:** *(Defines the test split and saves weights to `/models`)*
    ```bash
    python scripts/train.py
    ```
4.  **Generate XGBoost Baseline:** *(Must be run after training to ensure identical test sets)*
    ```bash
    python scripts/xgboost_baseline.py
    ```
5.  **Final Evaluation & Visualisation:**
    ```bash
    python scripts/model_comparison.py        # XGBoost vs. Fusion Metrics (RMSE/MAE)
    python scripts/case_studies.py            # Case Studies & Failures
    python scripts/generate_heatmaps.py       # Grad-CAM
    python scripts/ablation_study.py          # Modality Testing
    ```

---

## 🧠 Architecture Overview
The system utilises a **Late Fusion** strategy:

* **Vision Branch:** Pre-trained ResNet-50 backbone (frozen) projecting to a 128-dimensional embedding.
* **Tabular Branch:** 4-layer MLP compressing 738 features into a 64-dimensional embedding.
* **Fusion Head:** Concatenation layer (192-dim) followed by regression layers utilising **Huber Loss** for outlier robustness.
