# Project Documentation: Big Data Analytics & Predictive Intelligence

**Cognevance Technologies — Level 3 (Advanced) Project**

## 1. Objective
Develop a complete data analytics pipeline with predictive modeling and
business intelligence dashboards to predict e-commerce customer churn and
surface actionable business KPIs.

## 2. Business Domain
E-commerce customer analytics — predicting which customers are at risk of
churning based on demographics, spending, order history, and support
interactions.

## 3. Dataset
- **Source:** Synthetic large-scale e-commerce customer dataset (20,000 rows),
  built with realistic relationships between satisfaction, support tickets,
  tenure, and churn (as permitted by the brief: "Collect large-scale datasets
  from business, healthcare, finance, or e-commerce domains").
- **File:** `data/ecommerce_customers.csv`
- **Fields:** `customer_id, age, region, category_pref, membership_tier,
  tenure_months, monthly_spend, num_orders_last_year,
  support_tickets_last_year, satisfaction_score, churned`
- **Scale:** 20,000 customer records, 11 columns, ~1.6MB CSV.

## 4. Architecture / Workflow
```
data/ecommerce_customers.csv
        │
        ▼
src/ml_pipeline.py
  ├─ 1. Load & clean (dedupe, drop nulls)
  ├─ 2. Feature engineering (avg_order_value, tickets_per_order)
  ├─ 3. Train/test split (80/20, stratified)
  ├─ 4. Preprocessing pipeline (StandardScaler + OneHotEncoder)
  ├─ 5. Train Logistic Regression (baseline) + Random Forest
  ├─ 6. Evaluate (accuracy, precision, recall, F1, ROC-AUC)
  ├─ 7. Feature importance analysis
  └─ 8. Export charts (dashboard/assets/) + summary (report/model_summary.json)
        │
        ▼
dashboard/dashboard.html   ← interactive BI dashboard (Chart.js)
```

## 5. Tools & Technologies
- **Python 3** — pandas, numpy for data handling
- **scikit-learn** — preprocessing pipeline, Logistic Regression, Random
  Forest, evaluation metrics
- **matplotlib** — confusion matrix, ROC curve, feature importance, KPI charts
- **HTML/CSS/Chart.js** — interactive dashboard (stand-in for Power BI/Tableau,
  runs anywhere without a BI license — open `dashboard/dashboard.html` in any
  browser)

## 6. Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.677 | 0.660 | 0.509 | 0.575 | **0.732** |
| Random Forest | 0.670 | 0.676 | 0.445 | 0.536 | 0.726 |

Logistic Regression edges out Random Forest on ROC-AUC in this dataset,
suggesting the churn signal is largely linear/additive rather than requiring
complex non-linear splits — a useful finding for a real deployment where a
simpler, more interpretable model may be preferred.

## 7. Top Churn Drivers (Feature Importance)
1. **Satisfaction score** (28.1%) — by far the strongest predictor
2. **Support tickets last year** (15.3%)
3. **Tickets per order** (14.3%)
4. **Tenure (months)** (14.3%)
5. Average order value (6.0%)
6. Monthly spend (6.0%)

## 8. Business KPIs
- **Overall churn rate: 42.9%** across 20,000 customers
- **Churn by membership tier:** Platinum (44.0%) ≈ Silver (43.2%) ≈ Basic
  (43.0%) > Gold (41.7%) — churn is fairly flat across tiers in this dataset,
  suggesting membership tier alone is not a strong churn lever
- **Average monthly spend by region:** Central ($121.46) leads, South
  ($118.93) lowest — a narrow ~2% spread, indicating region is not a major
  differentiator either

## 9. Dashboards
- `dashboard/dashboard.html` — interactive dashboard with KPI cards, model
  comparison table, churn-by-tier and spend-by-region charts, feature
  importance chart, plus embedded confusion matrix and ROC curve images.
  Open directly in any browser — no server required.

## 10. Recommendations
1. **Prioritize satisfaction and support experience.** Satisfaction score and
   support-ticket volume dominate the churn signal — invest in proactive
   customer support and satisfaction tracking as the primary retention lever.
2. **Watch new customers closely.** Tenure is a top-4 driver — early-lifecycle
   customers are more churn-prone, so onboarding and early engagement programs
   are likely to have outsized impact.
3. **Don't over-index on membership tier or region for retention targeting** —
   churn is fairly evenly spread across both in this dataset; use the model's
   customer-level risk score instead of these broad segments.
4. **Deploy the Logistic Regression model** for production scoring given its
   higher ROC-AUC and easier interpretability/explainability for the business.

## 11. Limitations & Next Steps
- Dataset is synthetically generated; results and feature-importance rankings
  should be re-validated on real transactional data before use in production.
- Precision/recall trade-off (recall ~0.51 for the best model) means roughly
  half of actual churners are missed — a lower classification threshold or a
  cost-sensitive model could be explored to catch more at-risk customers.
- Model could be extended with time-series features (e.g. recent spend trend)
  and additional external data (marketing touch points, competitor activity).

## 12. Conclusion
This project delivers a full pipeline — data ingestion, cleaning, feature
engineering, model training/evaluation, and an interactive dashboard — for
predicting e-commerce customer churn. Satisfaction score and support
experience emerge as the dominant churn drivers, giving the business a clear,
actionable retention lever.
