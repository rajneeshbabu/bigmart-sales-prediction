# Big Mart Sales Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen)](https://lightgbm.readthedocs.io)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-blueviolet)](https://rajneeshbabu.github.io/bigmart-sales-prediction)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Predict item-level outlet sales across 10 Big Mart stores using machine learning. Built with comprehensive feature engineering, a 4-model benchmark, and a live web demo.

**Live Demo**: [rajneeshbabu.github.io/bigmart-sales-prediction](https://rajneeshbabu.github.io/bigmart-sales-prediction)

---

## How It Works

```
Raw Data (Train.csv / Test.csv)
        |
        v
Feature Engineering (14 features from 11 raw columns)
  - Fat content normalisation
  - Item category from identifier prefix
  - Outlet age, MRP band, log(MRP)
  - Visibility ratio, zero-visibility fix
  - Outlet size imputation by outlet type
        |
        v
4-Model Benchmark (5-fold CV)
  Ridge → Random Forest → XGBoost → LightGBM
        |
        v
Best Model: Tuned Random Forest
  RMSE: ₹1,094  |  R²: 0.589
        |
        v
Submission CSV + Streamlit App + GitHub Pages Demo
```

---

## Model Comparison (5-fold Cross-Validation)

| Model | CV RMSE | CV R² |
|---|---|---|
| Ridge (baseline) | ₹1,203 | 0.502 |
| XGBoost | ₹1,143 | 0.551 |
| LightGBM | ₹1,147 | 0.548 |
| Random Forest | ₹1,097 | 0.586 |
| **Blend (RF 80% + XGB 10% + LGB 10%)** ✅ | **₹1,098** | **0.586** |

**Best: Random Forest + Ensemble Blend** — RF dominates on this dataset due to its small size (~8,500 rows, 10 outlets). XGBoost and LightGBM add marginal benefit via blending.

---

## Feature Engineering

| Feature | Description |
|---|---|
| `Item_Fat_Content` | Normalised (LF → Low Fat, reg → Regular, Non-Consumable for household items) |
| `Item_Category` | Derived from Item_Identifier prefix: FD=Food, DR=Drinks, NC=Non-Consumable |
| `Outlet_Age` | Years since establishment (2013 − year) |
| `Item_MRP_Log` | log(1 + MRP) — reduces right skew |
| `MRP_Band` | Budget / Mid / Premium / Luxury |
| `Item_Visibility_Ratio` | Item visibility ÷ outlet mean visibility |
| `Outlet_Size` | Imputed by outlet type mode where missing |
| `Outlet_Type_Encoded` | Ordinal encoding: Grocery=0, S.Type1=1, S.Type2=2, S.Type3=3 |

---

## Project Structure

```
bigmart-sales-prediction/
├── index.html                      # GitHub Pages demo — no backend needed
├── app.py                          # Streamlit local app
├── requirements.txt
├── bigmart_sales_prediction.ipynb  # full EDA + training notebook
├── Train.csv                       # training data (8,523 rows)
├── Test.csv                        # test data (5,681 rows)
├── submission.csv                  # predicted sales for test set
├── README.md
│
├── models/
│   ├── rf_model.pkl                # trained Random Forest (≈ 50 MB)
│   ├── encoders.pkl                # label encoders for all categorical features
│   ├── metadata.json               # CV metrics, feature list, comparison results
│   └── comparison_results.json     # all 4-model benchmark scores
│
└── plots/
    ├── target_distribution.png
    ├── categorical_vs_target.png
    ├── numerical_vs_target.png
    ├── item_type_distribution.png
    ├── outlet_sales.png
    ├── model_comparison.png
    ├── feature_importance.png
    └── prediction_analysis.png
```

---

## Quick Start

### Option 1 — Open the live webpage (no setup needed)
Visit: **[rajneeshbabu.github.io/bigmart-sales-prediction](https://rajneeshbabu.github.io/bigmart-sales-prediction)**

### Option 2 — Run locally with Streamlit

```bash
# 1. Clone the repo
git clone https://github.com/rajneeshbabu/bigmart-sales-prediction.git
cd bigmart-sales-prediction

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
# Opens at http://localhost:8501
```

### Option 3 — Re-run the notebook

```bash
jupyter notebook bigmart_sales_prediction.ipynb
# Run all cells — regenerates plots/, models/, and submission.csv
```

---

## Dataset

- **Source**: [Big Mart Sales — Analytics Vidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/)
- **Train**: 8,523 rows × 12 columns
- **Test**: 5,681 rows × 11 columns
- **Target**: `Item_Outlet_Sales` — annual sales of item at a given outlet
- **Missing**: `Item_Weight` (1,463 rows), `Outlet_Size` (2,410 rows) — both imputed

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | Random Forest (scikit-learn) |
| Benchmark | XGBoost, LightGBM, Ridge |
| Data | Big Mart Sales dataset |
| Visualisation | matplotlib, seaborn |
| Web page | Pure HTML / CSS / JavaScript |
| Local app | Streamlit |
| Hosting | GitHub Pages (static) |

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with scikit-learn · XGBoost · LightGBM · Streamlit · GitHub Pages*
