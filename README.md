# Big Data Analytics & Predictive Intelligence

**Cognevance Technologies — Data Science & Data Analytics, Level 3 (Advanced)**

## 📌 Overview
An end-to-end predictive analytics pipeline that predicts e-commerce customer
churn from a 20,000-row customer dataset, with model evaluation, feature
importance analysis, and an interactive BI-style dashboard.

## 📂 Repository Structure
```
cognevance_big_data_predictive_intelligence/
├── data/
│   ├── ecommerce_customers.csv     # 20,000-row customer dataset
│   └── generate_data.py            # Script used to build the dataset
├── src/
│   └── ml_pipeline.py              # Full ML pipeline (clean → train → evaluate)
├── dashboard/
│   ├── dashboard.html              # Interactive dashboard (open in any browser)
│   └── assets/                     # Chart images embedded in the dashboard
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── feature_importance.png
│       ├── churn_by_tier.png
│       └── spend_by_region.png
├── report/
│   ├── Project_Documentation.md    # Full architecture + findings write-up
│   └── model_summary.json          # Machine-readable model results
└── README.md
```

## 🛠️ Tools Used
- Python 3, pandas, numpy
- scikit-learn (Logistic Regression, Random Forest, preprocessing pipeline)
- matplotlib (evaluation charts)
- HTML/CSS/Chart.js (interactive dashboard — no BI license required)

## ▶️ How to Run
```bash
cd src
pip install pandas numpy matplotlib scikit-learn
python3 ml_pipeline.py
```
Then open `dashboard/dashboard.html` directly in a browser to view the
interactive dashboard.

## 📊 Workflow
1. **Data Collection** — 20,000-row synthetic e-commerce customer dataset.
2. **Preprocessing & Feature Engineering** — cleaning, derived features
   (avg order value, tickets per order), scaling + one-hot encoding.
3. **Modeling** — Logistic Regression + Random Forest, trained via an
   sklearn `Pipeline`, evaluated on a held-out 20% test set.
4. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion
   matrix, ROC curve, feature importance.
5. **Business Intelligence** — churn-by-tier and spend-by-region KPI charts.
6. **Dashboard** — interactive HTML/Chart.js dashboard combining all results.
7. **Documentation** — full write-up in `report/Project_Documentation.md`.

## ✅ Key Results
- 20,000 customers analyzed, overall churn rate **42.9%**
- Best model: **Logistic Regression**, ROC-AUC **0.732**
- Top churn driver: **satisfaction score** (28% feature importance), followed
  by support ticket volume and account tenure
- Full interactive dashboard with KPIs, charts, and model diagnostics

## 📄 Deliverables
- [x] Complete analytics project source code
- [x] Machine learning model (Logistic Regression + Random Forest, compared)
- [x] Interactive dashboard (`dashboard/dashboard.html`)
- [x] Project documentation
- [x] Report
- [x] README

## ✍️ Author Notes
Dataset used here is a synthetically generated large-scale sample built with
realistic churn dynamics (as permitted by the brief: *"Collect large-scale
datasets from business, healthcare, finance, or e-commerce domains"*). Swap in
a real dataset with matching column names to reproduce the pipeline on real
business data. The HTML/Chart.js dashboard was used in place of Power BI/
Tableau so the deliverable can be viewed by anyone without a paid BI license —
swap in a `.pbix`/`.twbx` file instead if your submission requires it.
