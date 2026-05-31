<!--
  House Price Prediction Pipeline
  Bilingual README – Paste into Choquri2000/House_Price_Prediction_Pipeline
-->

<p align="center">
  <a href="#english">🇺🇸 English</a> &nbsp;•&nbsp;
  <a href="#georgian">🇬🇪 ქართული</a>
</p>

<hr>

<!-- ############################## ENGLISH ############################## -->
<a id="english"></a>

<h1 align="center">House Price Prediction Pipeline</h1>
<p align="center"><em>Statistical rigor · Optuna optimization · Blended ensemble</em></p>

---

### 📋 Overview

End-to-end regression pipeline predicting sale prices on the **Ames Housing** dataset. Built for statistical reproducibility — modular `src/` structure, YAML-driven configuration, full logging, and a 5-model **Pent-Ensemble** that generalizes beyond any single learner.

**Metric:** Log-RMSE = **0.11337** (Optuna-tuned XGBoost, confirmed via 5-fold CV).

---

### 🧪 Statistical Rigor

| Component | Approach |
|-----------|----------|
| **Split Strategy** | 80/10/10 train/val/test — stratified by `OverallQual` to preserve distribution |
| **Cross-Validation** | 5-fold shuffled with fixed seed (42) — RMSE tracked per fold |
| **Feature Selection** | Correlation threshold (>0.3 with target) + mutual information ranking + domain veto (e.g., forced-in `KitchenAbvGr`) |
| **Encoding** | Target encoding for high-cardinality categoricals; one-hot for low-cardinality |
| **Scaling** | RobustScaler on skewed features (IQR-based, outlier-resistant) |

---

### 🔬 Optuna Hyperparameter Search

Bayesian optimization over **100+ trials** using TPESampler:

```
search_space = {
    'n_estimators':      (100, 1000),
    'max_depth':         (3, 12),
    'learning_rate':     (0.005, 0.3, log=True),
    'subsample':         (0.6, 1.0),
    'colsample_bytree':  (0.4, 1.0),
    'reg_alpha':         (0, 10),
    'reg_lambda':        (0, 10),
}
objective = lambda trial: rmse(cv_predict(X_train, y_train, trial))
```

Best trial achieved **Log-RMSE 0.11337** — 7.2% improvement over default XGBoost baseline.

---

### 🏗️ Pent-Ensemble Architecture

```
                    ┌──── XGBoost (Optuna) ── 0.30 ─┐
                    │    CatBoost (default)  ── 0.20 │
  Raw Features ──── ┤    ElasticNet (α=0.5)  ── 0.15 │───▶ Weighted Average ───▶ Prediction
                    │    Ridge (α=1.0)       ── 0.15 │
                    └──── LinearRegression    ── 0.20 ┘
```

Weights learned via **stacking meta-regressor** on validation fold residuals. Blending reduces holdout RMSE by ~4% versus best single model.

---

### 🔄 Pipeline

```
raw/ ─▶ ingestion ─▶ validation ─▶ preprocessing ─▶ feature_eng ─▶ tuning ─▶ ensemble ─▶ submission.csv
```

Each stage is an isolated module in `src/` with its own logger and schema contract.

---

### 🚀 Run

```bash
git clone https://github.com/Choquri2000/House_Price_Prediction_Pipeline.git
cd House_Price_Prediction_Pipeline
pip install -r requirements.txt
python main.py
```

Output: `data/processed/submission.csv`

---

<hr>

<!-- ############################## GEORGIAN ############################## -->
<a id="georgian"></a>

<h1 align="center">სახლის ფასების პროგნოზირების პაიპლაინი</h1>
<p align="center"><em>სტატისტიკური სიმკაცრე · Optuna ოპტიმიზაცია · ანსამბლური ბლენდინგი</em></p>

---

### 📋 მიმოხილვა

რეგრესიული პაიპლაინი Ames Housing-ის მონაცემებზე სახლების გასაყიდი ფასების პროგნოზირებისთვის. მოდულარული `src/` სტრუქტურა, YAML კონფიგურაცია, სრული ლოგირება და 5-მოდელიანი **Pent-Ensemble**.

**მეტრიკა:** Log-RMSE = **0.11337** (Optuna-ოპტიმიზებული XGBoost, 5-fold CV).

---

### 🧪 სტატისტიკური მიდგომა

| კომპონენტი | მეთოდი |
|------------|--------|
| **გაყოფა** | 80/10/10 train/val/test, სტრატიფიცირებული `OverallQual`-ით |
| **CV** | 5-fold shuffled, ფიქსირებული seed (42) |
| **ფიჩერების შერჩევა** | კორელაციური ზღვარი >0.3 + mutual information + დომენური ვეტო |
| **ენკოდინგი** | Target encoding მაღალი კარდინალობის კატეგორიებისთვის |
| **სკალირება** | RobustScaler (IQR-ზე დაფუძნებული, outlier-რეზისტენტული) |

---

### 🔬 Optuna ჰიპერპარამეტრების ძიება

100+ ცდა TPESampler ალგორითმით:

```
n_estimators:      (100, 1000)
max_depth:         (3, 12)
learning_rate:     (0.005, 0.3, log)
subsample:         (0.6, 1.0)
colsample_bytree:  (0.4, 1.0)
reg_alpha:         (0, 10)
reg_lambda:        (0, 10)
```

საუკეთესო შედეგმა Log-RMSE = **0.11337** მისცა — 7.2%-ით უკეთესი ვიდრე default XGBoost.

---

### 🏗️ Pent-Ensemble არქიტექტურა

```
                    ┌──── XGBoost (Optuna) ── 0.30
                    │    CatBoost (default)  ── 0.20
  მახასიათებლები ───┤    ElasticNet (α=0.5) ── 0.15 ───▶ შეწონილი საშუალო ───▶ პროგნოზი
                    │    Ridge (α=1.0)      ── 0.15
                    └──── LinearRegression   ── 0.20
```

წონები განსაზღვრულია stacking meta-regressor-ით ვალიდაციის fold-ებზე. ბლენდინგი ამცირებს holdout RMSE-ს ~4%-ით.

---

### 🔄 პაიპლაინი

```
raw/ ─▶ ინგესცია ─▶ ვალიდაცია ─▶ პრეპროცესინგი ─▶ ფიჩერები ─▶ ტიუნინგი ─▶ ანსამბლი ─▶ submission.csv
```

---

### 🚀 გაშვება

```bash
git clone https://github.com/Choquri2000/House_Price_Prediction_Pipeline.git
cd House_Price_Prediction_Pipeline
pip install -r requirements.txt
python main.py
```

შედეგი: `data/processed/submission.csv`
