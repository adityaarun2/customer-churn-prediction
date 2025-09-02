# Customer Churn Prediction

**Live App:** [https://customer-churn-prediction-goih.onrender.com](https://customer-churn-prediction-goih.onrender.com)

A production-style churn prediction project built on the public IBM Telco Customer Churn dataset. It includes a reproducible training pipeline (data → features → model → metrics), an interactive Streamlit app with SHAP explanations and threshold/cost tuning, and an optional FastAPI service.

## Why churn prediction matters

Churn (also called customer attrition) refers to when a customer stops using a company’s product or service, such as canceling a subscription or switching to a competitor. In this context, churn prediction helps identify users who are likely to discontinue their telecom service so the business can take targeted actions to retain them.

Churn directly impacts revenue and growth. Predicting who is at risk lets teams:
- prioritize retention campaigns,
- allocate incentives strategically,
- forecast revenue more accurately.

This repo demonstrates how to ship that value end-to-end: from data and modeling to actionable UI/thresholds that reflect business trade-offs.

## Features
- Auto-downloads the IBM Telco Customer Churn dataset from Kaggle
- Cleans and preprocesses data
- Trains a baseline XGBoost model
- Saves pipeline and evaluation metrics
- Allows threshold and cost tuning via interactive Streamlit dashboard

## How to Run
You can run this project in two ways:

### Option 1: Local Setup (Recommended for Development)

#### 1. Clone the Repository
```bash
git clone https://github.com/adityaarun2/customer-churn-prediction.git
cd customer-churn-prediction
```

#### 2. Create and Activate the Conda Environment
```bash
conda env create -f environment.yml
conda activate churn
```

#### 3. Train the Model
```bash
make train
```
This runs `python -m src.train --config configs/config.yaml` to train both the models, saving outputs to the `models/` and `reports/` directories.

#### 4. Launch the Streamlit UI
```bash
make ui
```
This opens the customer churn prediction dashboard at [http://localhost:8501](http://localhost:8501).

---

### Option 2: Run with Docker

#### 1. Build the Docker Image
```bash
docker build -t churn .
```

#### 2. Run the Container
```bash
docker run -p 8501:8501 churn
```

Visit [http://localhost:8501](http://localhost:8501) to use the app.

> Make sure Docker is running.

---

### Optional: Clean Up Artifacts
To remove model checkpoints, caches, and reports:
```bash
make clean
```

## Results

After training on the Telco Customer Churn dataset using Logistic Regression, the following performance metrics were obtained:

| Model               | ROC‑AUC | PR‑AUC | Best Threshold |
|---------------------|:-------:|:------:|:--------------:|
| Logistic Regression | 0.832   | 0.648  | 0.47           |

- The model was trained on **5,174 non-churn** and **1,869 churn** customers.
- Train/Test split retained class distribution:
  - **Train:** 4,139 non-churn, 1,495 churn
  - **Test:** 1,035 non-churn, 374 churn

> ROC-AUC (Receiver Operating Characteristic - Area Under Curve) shows how well the model distinguishes between classes.
>
> PR-AUC (Precision-Recall AUC) is especially useful for imbalanced datasets like churn.
