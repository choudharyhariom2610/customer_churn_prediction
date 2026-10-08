# 📡 Customer Churn Prediction

A complete end-to-end Machine Learning project that predicts telecom customer churn using **Python**, **Scikit-learn**, and **Streamlit**.

---

## Project Architecture

```
customer_churn_prediction/
│
├── data/
│   ├── generate_data.py       # Script to regenerate the synthetic dataset
│   └── customer_churn.csv     # 1,000-row synthetic telecom dataset
│
├── model.py                   # Model training, evaluation, and serialisation
├── model.pkl                  # Saved best-performing model (auto-generated)
├── app.py                     # Streamlit multi-tab dashboard
├── requirements.txt           # Python package dependencies
└── README.md                  # This file
```

---

## Dataset — `data/customer_churn.csv`

| Column | Type | Description |
|---|---|---|
| `CustomerID` | string | Unique customer identifier (CUST0001 … CUST1000) |
| `Gender` | string | Male / Female |
| `SeniorCitizen` | int | 1 = Senior, 0 = Non-Senior |
| `Tenure_Months` | int | Months the customer has been active (1–72) |
| `MonthlyCharges` | float | Monthly bill amount in USD |
| `TotalCharges` | float | Cumulative charges over lifetime |
| `ContractType` | string | Month-to-month / One year / Two year |
| `TechSupport` | string | Yes / No |
| `Churn` | string | **Target** — Yes / No |

Churn probabilities are generated using realistic business rules (contract type, tenure, monthly charges, senior status, tech support).

---

## Model Training — `model.py`

Two classifiers are trained and compared:

| Model | Key Settings |
|---|---|
| **Random Forest** | 200 trees, max_depth=8, class_weight=balanced |
| **Logistic Regression** | StandardScaler pipeline, max_iter=1000, class_weight=balanced |

Evaluation metrics reported: **Accuracy, Precision, Recall, F1 Score, Confusion Matrix**.  
The model with the higher F1 score is saved to `model.pkl`.

---

## Streamlit Dashboard — `app.py`

### Tab 1 — Dataset Overview & Analytics
- **KPI Cards**: Total Customers · Churn Rate % · Avg Monthly Charge · Avg Tenure
- **Churn Distribution** pie chart
- **Churn Rate by Contract Type** horizontal bar chart
- **Monthly Charges Distribution** by churn status (histogram)
- **Avg Tenure by Contract & Churn** grouped bar chart
- **Tech Support** and **Senior Citizen** impact bar charts
- Expandable **Raw Dataset** table

### Tab 2 — Interactive Churn Prediction
- Input form: Gender, Senior Citizen, Contract Type, Tenure, Monthly Charges, Total Charges, Tech Support
- **Churn Risk Category**: 🔴 High / 🟡 Medium / 🟢 Low
- **Probability Score** with donut gauge chart
- **Key Risk Factors** checklist for the predicted customer

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset *(only needed once — already included)*
```bash
python data/generate_data.py
```

### 3. Train the model *(already generated — run to retrain)*
```bash
python model.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Tech Stack

| Layer | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| ML models | `scikit-learn` (RandomForestClassifier, LogisticRegression) |
| Visualisation | `matplotlib`, `seaborn` |
| Dashboard UI | `streamlit` |

---

## Results (Sample Run)

```
Model         : Random Forest
  Accuracy    : 0.6600
  Precision   : 0.4177
  Recall      : 0.6000
  F1 Score    : 0.4925

Model         : Logistic Regression
  Accuracy    : 0.6300
  Precision   : 0.4095
  Recall      : 0.7818
  F1 Score    : 0.5375

Best model: Logistic Regression  (F1 = 0.5375)
```

> Metrics reflect a class-imbalanced synthetic dataset (~27 % churn rate). Class weights are balanced automatically.

---

## License

This project is for educational and demonstration purposes only.
