# Customer Churn Prediction

## Project Overview
This project focuses on building a machine learning model to predict customer churn for a telecom company. Customer churn occurs when customers stop using a company's services, resulting in potential revenue loss. By predicting churn, businesses can take proactive measures to retain customers.

---

## Dataset
The dataset contains customer information such as demographics, account details, and service usage. Key features include:

- Gender, Senior Citizen status, Partner, Dependents
- Tenure (months with the company)
- Types of services subscribed (Phone, Internet, Streaming, etc.)
- Billing information (Monthly Charges, Total Charges)
- Target variable: `Churn` (Yes/No)

---

## Key Objectives
- Preprocess and clean the data
- Perform exploratory data analysis (EDA)
- Build predictive models using machine learning algorithms
- Use cross-validation and hyperparameter tuning to optimize models
- Deploy the best model in a Streamlit web app for real-time churn prediction

---

## Technologies & Libraries
- Python 3.x
- pandas, numpy
- scikit-learn
- randomboostclassifier
- matplotlib, seaborn
- Streamlit
- joblib (for model persistence)

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction

