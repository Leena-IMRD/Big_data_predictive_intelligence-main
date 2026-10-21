"""
Big Data Analytics & Predictive Intelligence
Cognevance Technologies - Level 3 (Advanced) Project

End-to-end pipeline: load -> clean -> feature engineer -> train a customer
churn prediction model -> evaluate -> export predictions + charts + a data
summary for the interactive dashboard.

Run: python3 ml_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import json

DATA_PATH = "../data/ecommerce_customers.csv"
CHART_DIR = "../dashboard/assets"
import os
os.makedirs(CHART_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()
df = df.dropna()
print("Dataset shape after cleaning:", df.shape)
print("Churn rate:", df["churned"].mean().round(3))

# ---------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------
df["avg_order_value"] = (df["monthly_spend"] * 12 / df["num_orders_last_year"].replace(0, 1)).round(2)
df["tickets_per_order"] = (df["support_tickets_last_year"] / df["num_orders_last_year"].replace(0, 1)).round(3)

numeric_features = ["age", "tenure_months", "monthly_spend", "num_orders_last_year",
                     "support_tickets_last_year", "satisfaction_score",
                     "avg_order_value", "tickets_per_order"]
categorical_features = ["region", "category_pref", "membership_tier"]

X = df[numeric_features + categorical_features]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ---------------------------------------------------------------
# 3. Train models: Logistic Regression (baseline) + Random Forest
# ---------------------------------------------------------------
results = {}

for name, clf in [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
]:
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    if name == "Random Forest":
        best_pipe = pipe
        best_proba = proba
        best_pred = pred

print("\nModel comparison:")
print(pd.DataFrame(results).T)

# ---------------------------------------------------------------
# 4. Confusion matrix + ROC curve for the best model (Random Forest)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.colorbar()
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, best_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color="#C44E52", label=f"ROC AUC = {results['Random Forest']['roc_auc']}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------
ohe_cols = list(best_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features))
all_cols = numeric_features + ohe_cols
importances = best_pipe.named_steps["clf"].feature_importances_
fi = pd.Series(importances, index=all_cols).sort_values(ascending=False).head(10)

plt.figure(figsize=(7, 5))
fi.sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Top 10 Feature Importances - Churn Prediction")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Business KPI charts
# ---------------------------------------------------------------
churn_by_tier = df.groupby("membership_tier")["churned"].mean().sort_values(ascending=False)
plt.figure(figsize=(6, 4))
churn_by_tier.plot(kind="bar", color="#C44E52")
plt.title("Churn Rate by Membership Tier")
plt.ylabel("Churn Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/churn_by_tier.png", dpi=150)
plt.close()

spend_by_region = df.groupby("region")["monthly_spend"].mean().sort_values(ascending=False)
plt.figure(figsize=(6, 4))
spend_by_region.plot(kind="bar", color="#55A868")
plt.title("Average Monthly Spend by Region")
plt.ylabel("Avg Monthly Spend ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/spend_by_region.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Export summary JSON for the dashboard + report
# ---------------------------------------------------------------
summary = {
    "n_customers": int(len(df)),
    "overall_churn_rate": round(float(df["churned"].mean()), 4),
    "model_results": results,
    "top_features": fi.to_dict(),
    "churn_by_tier": churn_by_tier.round(4).to_dict(),
    "avg_spend_by_region": spend_by_region.round(2).to_dict(),
}
with open("../report/model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved charts to ../dashboard/assets/ and summary to ../report/model_summary.json")
print(json.dumps(summary, indent=2))
