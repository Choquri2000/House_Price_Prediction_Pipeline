# 🏠 House Price Prediction Pipeline

🌐 **აირჩიეთ ენა / Choose Language:**  
[🇬🇪 ქართული](#georgian) | [🇺🇸 English](#english)

---

<a name="georgian"></a>
## 🇬🇪 ქართული ვერსია

### 📝 პროექტის მიმოხილვა
ეს არის **Data Science** პაიპლაინი სახლის ფასების პროგნოზირებისთვის, რომელიც იყენებს ანსამბლურ მეთოდებსა და ჰიპერპარამეტრების ავტომატიზებულ ოპტიმიზაციას. პროექტი აგებულია Ames Housing-ის მონაცემებზე და ფოკუსირებულია კოდის მოდულურობასა და MLOps-ის პრინციპებზე.

### 🛠️ ტექნოლოგიური სტეკი
- **Core:** Python, Pandas, NumPy
- **ML Models:** XGBoost, CatBoost, Scikit-Learn (Linear/ElasticNet)
- **Tuning:** **Optuna** ჰიპერპარამეტრების ოპტიმიზაციისთვის
- **Architecture:** მოდულური `src/` სტრუქტურა, YAML კონფიგურაციები

### 🔄 სამუშაო პროცესი (Workflow)
1. **Data Ingestion:** ნედლი მონაცემების ჩატვირთვა `data/raw/`-დან.
2. **Validation:** მონაცემთა მთლიანობისა და სტრუქტურის ვალიდაცია.
3. **Preprocessing:** გამოტოვებული მნიშვნელობების შევსება, კატეგორიული ენკოდინგი და სკალირება.
4. **Feature Engineering:** დომენზე მორგებული ახალი მახასიათებლების შექმნა.
5. **Hyperparameter Tuning:** XGBoost-ის ოპტიმიზაცია Optuna-ს მეშვეობით.
6. **Ensemble Training:** **Pent-Ensemble** (5 მოდელის ბლენდინგი) მაქსიმალური სიზუსტისთვის.
7. **Prediction:** საბოლოო `submission.csv` ფაილის გენერაცია.

### 📊 მოდელის შედეგები
- **Best XGBoost (Optuna):** Log-RMSE = 0.11337.
- **Ensemble-ის უპირატესობა:** მოდელების გაერთიანება ამცირებს Overfitting-ს და აუმჯობესებს განზოგადებას.

### 🚀 გაშვების ინსტრუქცია
```bash
git clone https://github.com/Choquri2000/House_Price_Prediction_Pipeline.git
pip install -r requirements.txt
python main.py
```

---

<a name="english"></a>
## 🇺🇸 English Version

### 📋 Project Overview
End-to-end **Data Science** pipeline for predicting house prices using advanced ensemble methods and automated hyperparameter tuning. This project demonstrates production-ready MLOps practices with modular code and reproducible workflows.

### 🛠️ Tech Stack
- **Core**: Python, Pandas, NumPy
- **ML Models**: XGBoost, CatBoost, Scikit-Learn
- **Tuning**: **Optuna** for hyperparameter optimization
- **Pipeline**: Modular `src/` structure, YAML configs, logging

### 🔄 Pipeline Workflow
1. **Data Ingestion**: Loading raw train/test data from `data/raw/`.
2. **Validation**: Checking data integrity and shape consistency.
3. **Preprocessing**: Handling missing values, categorical encoding, and scaling.
4. **Feature Engineering**: Creating domain-specific housing features.
5. **Hyperparameter Tuning**: Optuna optimizes XGBoost parameters for peak performance.
6. **Ensemble Training**: 5-model **Pent-Ensemble** blending (XGBoost + CatBoost + Linear models).
7. **Prediction**: Generating Kaggle-ready `submission.csv`.

### 📊 Performance Highlights
- **Best XGBoost (Optuna)**: Log-RMSE = 0.11337.
- **Ensemble Benefit**: Blending 5 models significantly reduces overfitting and improves generalization.
- **Final Output**: Competition-ready `submission.csv` located in `data/processed/`.

### 🚀 Setup & Run
```bash
git clone https://github.com/Choquri2000/House_Price_Prediction_Pipeline.git
pip install -r requirements.txt
python main.py
```

---
*📩 Contact [Choquri2000](https://github.com/Choquri2000) for Data Science & ML collaborations.*
