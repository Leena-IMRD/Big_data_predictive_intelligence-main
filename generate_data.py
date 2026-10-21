"""
Generates a large synthetic e-commerce customer dataset for predictive
analytics: customer demographics, behavior, and churn label.
"""
import numpy as np
import pandas as pd

np.random.seed(21)
n = 20000

age = np.clip(np.random.normal(38, 12, n), 18, 75).round(0).astype(int)
tenure_months = np.clip(np.random.exponential(24, n), 1, 96).round(0).astype(int)
monthly_spend = np.clip(np.random.normal(120, 60, n), 5, 800).round(2)
support_tickets = np.random.poisson(1.2, n)
num_orders = np.clip(np.random.poisson(8, n) + (monthly_spend / 20).astype(int), 0, None)
satisfaction_score = np.clip(np.random.normal(7, 1.8, n), 1, 10).round(1)
region = np.random.choice(["North", "South", "East", "West", "Central"], n)
category_pref = np.random.choice(
    ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports"], n
)
membership = np.random.choice(["Basic", "Silver", "Gold", "Platinum"], n, p=[0.4, 0.3, 0.2, 0.1])

# churn probability driven by low satisfaction, low tenure, high support tickets
churn_logit = (
    -0.35 * satisfaction_score
    - 0.02 * tenure_months
    + 0.55 * support_tickets
    - 0.01 * num_orders
    + np.random.normal(0, 1.0, n)
    + 2.0
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = (np.random.rand(n) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"C{100000+i}" for i in range(n)],
    "age": age,
    "region": region,
    "category_pref": category_pref,
    "membership_tier": membership,
    "tenure_months": tenure_months,
    "monthly_spend": monthly_spend,
    "num_orders_last_year": num_orders,
    "support_tickets_last_year": support_tickets,
    "satisfaction_score": satisfaction_score,
    "churned": churn,
})

df.to_csv("ecommerce_customers.csv", index=False)
print(df.head())
print(df.shape)
print("Churn rate:", df["churned"].mean().round(3))
