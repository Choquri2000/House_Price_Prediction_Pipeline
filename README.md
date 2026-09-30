<p align="center">
  <a href="#english">🇺🇸 English</a> &nbsp;•&nbsp;
  <a href="#georgian">🇬🇪 ქართული</a>
</p>

<hr>

<!-- ############################## ENGLISH ############################## -->
<a id="english"></a>

<h1 align="center">House Price Prediction Pipeline</h1>
<p align="center"><em>Modular regression pipeline · Optuna-tuned XGBoost · 5-model weighted ensemble</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Kaggle-Log--RMSE%200.11337-20BEFF" alt="Kaggle Log-RMSE 0.11337">
</p>

---

### Overview

An end-to-end regression pipeline that predicts residential sale prices on the **Ames Housing** dataset (Kaggle competition *House Prices: Advanced Regression Techniques*). The code is organised into modules under `src/` (data loading, preprocessing and feature engineering, tuning), configured through a YAML file, and ends in a weighted ensemble of five models.

### Result

| Metric | Score |
|--------|-------|
| **Kaggle public leaderboard (Log-RMSE)** | **0.11337** |

---

### Pipeline

```
data/raw/*.csv ─▶ DataLoader ─▶ DataPreprocessor ─▶ Optuna (XGBoost) ─▶ 5-model ensemble ─▶ submission.csv
                  (validation)   (cleaning, features,  (20 trials)          (weighted blend)
                                  encoding, scaling)
```

**1. Data loading and validation**: `src/data/data_loader.py`
- Reads the file paths from `configs/config.yaml`.
- Checks that the dataset is not empty and that the `Id` column (plus `SalePrice` for training data) is present.

**2. Preprocessing and feature engineering**: `src/features/preprocessor.py`
- **Outlier removal:** drops training rows with `GrLivArea > 4000` and `SalePrice < 300000`.
- **Missing values:** median for numeric columns, `"None"` for categorical columns.
- **Engineered features:**
  - `TotalSF`: basement + 1st floor + 2nd floor area.
  - `TotalBath`: full baths plus half baths weighted by 0.5.
  - `HasPool`: whether the house has a pool.
  - `MSSubClass`, `YrSold` and `MoSold` are treated as categorical.
- **Skewness correction:** `log1p` on numeric features with |skew| > 0.75.
- **Encoding and scaling:** one-hot encoding, then `StandardScaler` (needed by the linear models).
- **Target:** `log1p(SalePrice)`. Predictions are converted back with `expm1`.

**3. Hyperparameter optimization**: `src/models/tuner.py`
- Optuna (default TPE sampler), 20 trials, minimising RMSE on an 80/20 hold-out split (`random_state=42`).

| Parameter | Search range |
|-----------|--------------|
| `n_estimators` | 500 – 2000 |
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.1 (log) |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `gamma` | 1e-8 – 1.0 (log) |

**4. Weighted ensemble**: `main.py`

| Model | Configuration | Weight |
|-------|---------------|--------|
| XGBoost | Optuna best parameters | 0.25 |
| LightGBM | `num_leaves=5`, `learning_rate=0.05`, `n_estimators=1200` | 0.25 |
| CatBoost | `iterations=2000`, `learning_rate=0.03`, `depth=4`, `l2_leaf_reg=4`, Bernoulli `subsample=0.6` | 0.20 |
| LassoCV | `alphas=[0.0001, 0.0005, 0.001]`, 5-fold CV | 0.15 |
| ElasticNetCV | `l1_ratio=[0.1, 0.5, 0.9]`, 5-fold CV | 0.15 |

The gradient-boosting models capture non-linear interactions, and the regularised linear models add stability. The pipeline also plots the top-10 XGBoost feature importances.

---

### How to Run

```bash
git clone https://github.com/Choquri2000/House_Price_Prediction_Pipeline.git
cd House_Price_Prediction_Pipeline
pip install -r requirements.txt
```

1. Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) into `data/raw/`. Data files are excluded from version control.
2. Run the pipeline:
   ```bash
   python main.py
   ```
3. Output: `data/processed/submission.csv` and `data/processed/feature_importance.png`

---

### Project Structure

```
├── main.py                     # Pipeline entry point (preprocessing → tuning → ensemble → submission)
├── configs/config.yaml         # Data paths, target/ID columns, preprocessing settings
├── requirements.txt
└── src/
    ├── data/
    │   ├── data_loader.py      # Config-driven loading and schema validation
    │   └── preprocess.py       # Earlier preprocessing version (not used by main.py)
    ├── features/
    │   └── preprocessor.py     # Cleaning, feature engineering, encoding, scaling
    └── models/
        ├── tuner.py            # Optuna hyperparameter search for XGBoost
        ├── model_trainer.py    # Single-model XGBoost trainer (standalone)
        └── predictor.py        # Inference with a saved model (standalone)
```

---

### Roadmap

- K-fold cross-validation for more reliable model comparison
- Learning the ensemble weights with a stacking meta-model instead of fixed weights
- Refitting the final models on the full training set
- Unit tests and a CI workflow (GitHub Actions)

<p align="right"><a href="#georgian">🇬🇪 ქართული ↓</a></p>

<hr>

<!-- ############################## GEORGIAN ############################## -->
<a id="georgian"></a>

<h1 align="center">House Price Prediction Pipeline</h1>
<p align="center"><em>სახლების ფასების პროგნოზირების მოდულური pipeline · Optuna-ით ოპტიმიზებული XGBoost · ხუთი მოდელის შეწონილი ensemble</em></p>

---

### მიმოხილვა

პროექტი წარმოადგენს სრულ რეგრესიულ pipeline-ს, რომელიც **Ames Housing** მონაცემებზე საცხოვრებელი სახლების გასაყიდ ფასს პროგნოზირებს (Kaggle-ის შეჯიბრი *House Prices: Advanced Regression Techniques*). კოდი დაყოფილია `src/` დირექტორიის მოდულებად (მონაცემების ჩატვირთვა, preprocessing და feature engineering, ტიუნინგი), პარამეტრები იმართება YAML კონფიგურაციით, საბოლოო პროგნოზს კი ხუთი მოდელის შეწონილი ensemble იძლევა.

### შედეგი

| მეტრიკა | შედეგი |
|---------|--------|
| **Kaggle public leaderboard (Log-RMSE)** | **0.11337** |

---

### Pipeline

```
data/raw/*.csv ─▶ DataLoader ─▶ DataPreprocessor ─▶ Optuna (XGBoost) ─▶ 5-model ensemble ─▶ submission.csv
```

**1. მონაცემების ჩატვირთვა და ვალიდაცია**: `src/data/data_loader.py`
- ფაილების მისამართებს კითხულობს `configs/config.yaml`-დან.
- ამოწმებს, რომ ცხრილი ცარიელი არ არის და შეიცავს `Id` სვეტს (სატრენინგო მონაცემები `SalePrice`-საც).

**2. Preprocessing და feature engineering**: `src/features/preprocessor.py`
- **Outlier-ების მოცილება:** სატრენინგო მონაცემებიდან იშლება ჩანაწერები, სადაც `GrLivArea > 4000` და `SalePrice < 300000`.
- **გამოტოვებული მნიშვნელობები:** რიცხვით სვეტებში ივსება მედიანით, კატეგორიულში `"None"`-ით.
- **ახალი feature-ები:**
  - `TotalSF`: სარდაფის, პირველი და მეორე სართულის ფართობების ჯამი.
  - `TotalBath`: სრული სააბაზანოები, ნახევარი სააბაზანოები 0.5 წონით.
  - `HasPool`: აქვს თუ არა სახლს აუზი.
  - `MSSubClass`, `YrSold` და `MoSold` განიხილება როგორც კატეგორიული ცვლადები.
- **Skewness-ის კორექცია:** `log1p` ტრანსფორმაცია იმ რიცხვით feature-ებზე, რომელთა |skew| > 0.75.
- **Encoding და scaling:** one-hot encoding, შემდეგ `StandardScaler` (აუცილებელია წრფივი მოდელებისთვის).
- **სამიზნე ცვლადი:** `log1p(SalePrice)`. პროგნოზი საწყის მასშტაბზე `expm1`-ით ბრუნდება.

**3. Hyperparameter optimization**: `src/models/tuner.py`
- Optuna (ნაგულისხმევი TPE sampler), 20 ცდა. მინიმიზდება RMSE 80/20 hold-out გაყოფაზე (`random_state=42`).

| პარამეტრი | საძიებო დიაპაზონი |
|-----------|--------------------|
| `n_estimators` | 500 – 2000 |
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.1 (log) |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `gamma` | 1e-8 – 1.0 (log) |

**4. შეწონილი ensemble**: `main.py`

| მოდელი | კონფიგურაცია | წონა |
|--------|--------------|------|
| XGBoost | Optuna-ს საუკეთესო პარამეტრები | 0.25 |
| LightGBM | `num_leaves=5`, `learning_rate=0.05`, `n_estimators=1200` | 0.25 |
| CatBoost | `iterations=2000`, `learning_rate=0.03`, `depth=4`, `l2_leaf_reg=4`, Bernoulli `subsample=0.6` | 0.20 |
| LassoCV | `alphas=[0.0001, 0.0005, 0.001]`, 5-fold CV | 0.15 |
| ElasticNetCV | `l1_ratio=[0.1, 0.5, 0.9]`, 5-fold CV | 0.15 |

Gradient boosting მოდელები არაწრფივ დამოკიდებულებებს იჭერენ, რეგულარიზებული წრფივი მოდელები კი პროგნოზს სტაბილურობას მატებენ. Pipeline ასევე აგებს XGBoost-ის ათი ყველაზე მნიშვნელოვანი feature-ის გრაფიკს.

---

### გაშვება

```bash
git clone https://github.com/Choquri2000/House_Price_Prediction_Pipeline.git
cd House_Price_Prediction_Pipeline
pip install -r requirements.txt
```

1. ჩამოტვირთეთ `train.csv` და `test.csv` [Kaggle-ის შეჯიბრის გვერდიდან](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) და მოათავსეთ `data/raw/` დირექტორიაში. მონაცემთა ფაილები რეპოზიტორიაში არ ინახება.
2. გაუშვით pipeline:
   ```bash
   python main.py
   ```
3. შედეგი: `data/processed/submission.csv` და `data/processed/feature_importance.png`

---

### პროექტის სტრუქტურა

```
├── main.py                     # Pipeline-ის შესასვლელი წერტილი
├── configs/config.yaml         # მონაცემების მისამართები, სამიზნე და ID სვეტები, preprocessing-ის პარამეტრები
├── requirements.txt
└── src/
    ├── data/
    │   ├── data_loader.py      # ჩატვირთვა კონფიგურაციიდან და სქემის ვალიდაცია
    │   └── preprocess.py       # preprocessing-ის ადრეული ვერსია (main.py არ იყენებს)
    ├── features/
    │   └── preprocessor.py     # გაწმენდა, feature engineering, encoding, scaling
    └── models/
        ├── tuner.py            # XGBoost-ის hyperparameter optimization (Optuna)
        ├── model_trainer.py    # ერთი XGBoost მოდელის დამოუკიდებელი ტრენინგი
        └── predictor.py        # პროგნოზი შენახული მოდელით
```

---

### სამომავლო გეგმა

- K-fold cross-validation მოდელების უფრო სანდო შედარებისთვის
- ensemble-ის წონების შერჩევა stacking meta-model-ით, ფიქსირებული წონების ნაცვლად
- საბოლოო მოდელების ხელახალი ტრენინგი სრულ სატრენინგო მონაცემებზე
- Unit test-ები და CI workflow (GitHub Actions)

<p align="right"><a href="#english">🇺🇸 English ↑</a></p>
