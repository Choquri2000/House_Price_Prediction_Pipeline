# 🏠 House Price Prediction Pipeline

## 📋 Project Overview
End-to-end machine learning pipeline for predicting house sale prices using advanced ensemble methods and automated hyperparameter tuning. Built for the Ames Housing Dataset (Kaggle competition), this project demonstrates production-ready MLOps practices with modular code, configuration management, and reproducible training workflows.

## 🛠️ Tech Stack
- **Core**: Python 3.9+, Pandas, NumPy
- **ML Models**: XGBoost, CatBoost, Scikit-Learn (Linear/ElasticNet)
- **Tuning**: Optuna for hyperparameter optimization
- **Pipeline**: Modular `src/` structure, YAML configs, logging
- **Output**: Kaggle-ready `submission.csv`

## 📂 Project Structure
```
House_Price_Prediction/
├── House_Price_Prediction_Pipeline/  # Core pipeline package
│   ├── configs/              # YAML configuration files
│   ├── src/                  # Modular source code
│   │   ├── data/             # Data loading & preprocessing
│   │   ├── features/         # Feature engineering
│   │   └── models/           # Training, tuning, prediction
│   ├── main.py               # Pipeline entry point
│   ├── config.py             # Config loader
│   └── requirements.txt      # Dependencies
├── catboost_info/            # CatBoost training logs
├── data/                     # Raw & processed datasets
└── README.md                 # This file
```

## 🔄 Pipeline Workflow
1. **Data Ingestion**: Loads raw train/test data from `data/raw/`
2. **Validation**: Checks data integrity and shape consistency
3. **Preprocessing**: Handles missing values, encodes categorical features, scales numerics
4. **Feature Engineering**: Creates domain-specific housing features
5. **Hyperparameter Tuning**: Optuna optimizes XGBoost (n_estimators, max_depth, learning_rate, etc.)
6. **Ensemble Training**: 5-model Pent-Ensemble (XGBoost + CatBoost + Linear models) with blending
7. **Prediction**: Generates final predictions to `data/processed/submission.csv`

## 📊 Model Performance
- **Best XGBoost (Optuna)**: Log-RMSE = 0.11337 (Kaggle evaluation metric)
- **Ensemble Benefit**: Blending 5 models reduces overfitting and improves generalization
- **Final Output**: Competition-ready `submission.csv`

## 🚀 How to Run
1. Clone the repo:
```bash
git clone https://github.com/Choquri2000/House_Price_Prediction_Pipeline.git
cd House_Price_Prediction_Pipeline/House_Price_Prediction_Pipeline
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the pipeline:
```bash
python main.py
```
4. Results in `data/processed/submission.csv`

## 🔧 Key Improvements Made
- Implemented 5-model ensemble to boost accuracy vs single models
- Added Optuna hyperparameter tuning for XGBoost (20 trials)
- Modularized code into reusable `src/` components
- Added logging for pipeline transparency
- Automated end-to-end workflow from raw data to submission

## 🎯 Conclusion
This pipeline delivers a complete, reproducible ML solution for housing price prediction, combining state-of-the-art ensemble methods with automated tuning. The modular design makes it easily extensible for other regression tasks.

---
*For questions or collaboration, contact [Choquri2000](https://github.com/Choquri2000)*
