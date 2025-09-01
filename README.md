# Churn Radar

End-to-end churn prediction system built with scikit-learn and Streamlit.


## Features
- Auto-downloads the IBM Telco Customer Churn dataset from Kaggle
- Cleans and preprocesses data
- Trains a baseline model (Logistic Regression)
- Saves pipeline and evaluation metrics
- Interactive dashboard with Streamlit


## How to Run
```bash
make setup
make train
make ui