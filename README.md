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
  <img src="https://github.com/Choquri2000/House_Price_Prediction_Pipeline/actions/workflows/tests.yml/badge.svg" alt="tests">
</p>

---

### Overview

An end-to-end regression pipeline that predicts residential sale prices on the **Ames Housing** dataset (Kaggle competition *House Prices: Advanced Regression Techniques*). The code is organised into modules under `src/` (data loading, preprocessing and feature engineering, tuning), configured through a YAML file, and ends in a weighted ensemble of five models.

### Result

| Metric | Score |
|--------|-------|
| **Kaggle public leaderboard (Log-RMSE)** | **0.11337** |

**5-fold cross-validation** (`python evaluate.py`, RMSE on log price, mean ± std; XGBoost uses fixed default parameters here, not the Optuna result):

| Model | CV RMSE |
|-------|---------|
| XGBoost | 0.1157 ± 0.0062 |
| LightGBM | 0.1219 ± 0.0088 |
| CatBoost | 0.1116 ± 0.0076 |
| LassoCV | 0.1182 ± 0.0104 |
| ElasticNetCV | 0.1121 ± 0.0096 |
| **Weighted blend** | **0.1103 ± 0.0079** |

The blend scores better than every single model.

---

### Pipeline

```
data/raw/*.csv ─▶ DataLoader ─▶ DataPreprocessor ─▶ Optuna (XGBoost) ─▶ 5-model ensemble ─▶ submission.csv
                  (validation)   (cleaning, features,  (n_trials, default 20) (weighted blend)
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
- Optuna (default TPE sampler), minimising RMSE on an 80/20 hold-out split (`random_state=42`).
- Number of trials is set by `tuning.n_trials` in `configs/config.yaml` (default 20; raise it for a longer search).

| Parameter | Search range |
|-----------|--------------|
| `n_estimators` | 500 – 2000 |
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.1 (log) |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `gamma` | 1e-8 – 1.0 (log) |

**4. Weighted ensemble**: `src/models/ensemble.py` (used by `main.py`)

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
4. Optional: `python evaluate.py` runs 5-fold cross-validation (`--folds N` to change); `pytest` runs the tests.

---

### Project Structure

```
├── main.py                     # Pipeline entry point (preprocessing → tuning → ensemble → submission)
├── evaluate.py                 # K-fold CV of each model and of the blend
├── configs/config.yaml         # Data paths, target/ID columns, tuning trials
├── requirements.txt
├── tests/                      # pytest suite (runs on synthetic data, no Kaggle files needed)
├── .github/workflows/tests.yml # CI: pytest on every push
└── src/
    ├── data/
    │   └── data_loader.py      # Config-driven loading and schema validation
    ├── features/
    │   └── preprocessor.py     # Cleaning, feature engineering, encoding, scaling
    └── models/
        ├── tuner.py            # Optuna hyperparameter search for XGBoost
        ├── ensemble.py         # The five models and the blend weights
        ├── model_trainer.py    # Single-model XGBoost trainer (standalone)
        └── predictor.py        # Inference with a saved model (standalone)
```

---

### Roadmap

- Learning the ensemble weights with a stacking meta-model instead of fixed weights
- Refitting the final models on the full training set
- Tuning inside the CV loop (currently `evaluate.py` uses fixed default XGBoost parameters)

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

**5-fold cross-validation** (`python evaluate.py`, RMSE ლოგარითმულ ფასზე, საშუალო ± სტანდარტული გადახრა; აქ XGBoost ფიქსირებულ ნაგულისხმევ პარამეტრებს იყენებს, Optuna-ს შედეგს არა):

| მოდელი | CV RMSE |
|--------|---------|
| XGBoost | 0.1157 ± 0.0062 |
| LightGBM | 0.1219 ± 0.0088 |
| CatBoost | 0.1116 ± 0.0076 |
| LassoCV | 0.1182 ± 0.0104 |
| ElasticNetCV | 0.1121 ± 0.0096 |
| **შეწონილი ensemble** | **0.1103 ± 0.0079** |

ensemble ყველა ცალკეულ მოდელზე უკეთეს შედეგს აჩვენებს.

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
- Optuna (ნაგულისხმევი TPE sampler). მინიმიზდება RMSE 80/20 hold-out გაყოფაზე (`random_state=42`).
- ცდების რაოდენობა განისაზღვრება `configs/config.yaml`-ის `tuning.n_trials` პარამეტრით (ნაგულისხმევი 20; გრძელი ძიებისთვის გაზარდეთ).

| პარამეტრი | საძიებო დიაპაზონი |
|-----------|--------------------|
| `n_estimators` | 500 – 2000 |
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.1 (log) |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `gamma` | 1e-8 – 1.0 (log) |

**4. შეწონილი ensemble**: `src/models/ensemble.py` (გამოიყენება `main.py`-ში)

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
4. არასავალდებულო: `python evaluate.py` უშვებს 5-fold cross-validation-ს (`--folds N`); `pytest` უშვებს ტესტებს.

---

### პროექტის სტრუქტურა

```
├── main.py                     # Pipeline-ის შესასვლელი წერტილი
├── evaluate.py                 # თითოეული მოდელის და ensemble-ის K-fold CV
├── configs/config.yaml         # მონაცემების მისამართები, სამიზნე და ID სვეტები, ცდების რაოდენობა
├── requirements.txt
├── tests/                      # pytest ტესტები (სინთეზურ მონაცემებზე, Kaggle-ის ფაილები არ სჭირდება)
├── .github/workflows/tests.yml # CI: pytest ყოველ push-ზე
└── src/
    ├── data/
    │   └── data_loader.py      # ჩატვირთვა კონფიგურაციიდან და სქემის ვალიდაცია
    ├── features/
    │   └── preprocessor.py     # გაწმენდა, feature engineering, encoding, scaling
    └── models/
        ├── tuner.py            # XGBoost-ის hyperparameter optimization (Optuna)
        ├── ensemble.py         # ხუთი მოდელი და ensemble-ის წონები
        ├── model_trainer.py    # ერთი XGBoost მოდელის დამოუკიდებელი ტრენინგი
        └── predictor.py        # პროგნოზი შენახული მოდელით
```

---

### სამომავლო გეგმა

- ensemble-ის წონების შერჩევა stacking meta-model-ით, ფიქსირებული წონების ნაცვლად
- საბოლოო მოდელების ხელახალი ტრენინგი სრულ სატრენინგო მონაცემებზე
- Tuning-ის CV ციკლში გადატანა (ამჟამად `evaluate.py` XGBoost-ის ფიქსირებულ პარამეტრებს იყენებს)

<p align="right"><a href="#english">🇺🇸 English ↑</a></p>
