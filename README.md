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
```bash
make setup
make train
make ui
```


