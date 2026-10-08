"""
Generates a synthetic telecom customer churn dataset with 1,000 realistic rows.
Run this script once to produce data/customer_churn.csv.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, N + 1)]
gender = np.random.choice(["Male", "Female"], size=N)
senior_citizen = np.random.choice([0, 1], size=N, p=[0.84, 0.16])
tenure_months = np.random.randint(1, 73, size=N)
monthly_charges = np.round(np.random.uniform(18.0, 120.0, size=N), 2)
total_charges = np.round(monthly_charges * tenure_months + np.random.uniform(-10, 10, size=N), 2)
total_charges = np.clip(total_charges, 0, None)
contract_type = np.random.choice(
    ["Month-to-month", "One year", "Two year"],
    size=N,
    p=[0.55, 0.25, 0.20],
)
tech_support = np.random.choice(["Yes", "No"], size=N, p=[0.41, 0.59])

# Churn probability driven by realistic business rules
churn_prob = (
    0.05
    + 0.30 * (contract_type == "Month-to-month").astype(float)
    + 0.08 * senior_citizen
    + 0.10 * (tech_support == "No").astype(float)
    - 0.004 * tenure_months
    + 0.002 * monthly_charges
)
churn_prob = np.clip(churn_prob, 0.02, 0.95)
churn_raw = np.random.binomial(1, churn_prob)
churn = np.where(churn_raw == 1, "Yes", "No")

df = pd.DataFrame(
    {
        "CustomerID": customer_ids,
        "Gender": gender,
        "SeniorCitizen": senior_citizen,
        "Tenure_Months": tenure_months,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "ContractType": contract_type,
        "TechSupport": tech_support,
        "Churn": churn,
    }
)

df.to_csv("data/customer_churn.csv", index=False)
print(f"Dataset saved -> data/customer_churn.csv  ({len(df)} rows)")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
