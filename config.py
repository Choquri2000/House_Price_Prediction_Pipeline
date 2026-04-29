"""
Configuration file for House Price Prediction Project
Centralizes all paths, parameters, and settings
"""

import os

# Base directory (where this config file is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# File names
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Feature columns that were added (macroeconomic features)
MACRO_FEATURES = [
    'GDP-Yr', 'HPI-Qrtr', 'LaborForce', 'Emplmnt', 'Unemplmnt',
    'UnemplmntRate', 'PrCapPersInc', 'IntRate', 'NmnlMedHshldInc',
    'RlMedHshldInc', 'NmnlAdjMedFamInc', 'InflAdjMedFamInc',
    'NmnlrCapInc', 'RlPrCapInc'
]

# Outlier IDs to remove (from your analysis)
OUTLIER_IDS = [1299, 524]

# Missing data threshold (drop columns with > X missing values)
MISSING_THRESHOLD = 7

# Lasso parameters
LASSO_ALPHAS = [1, 0.1, 0.001, 0.0005]

# XGBoost parameters (from your Optuna tuning)
XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.855770219034712,
    'subsample': 0.7242078792488182,
    'colsample_bytree': 0.9060873499476084,
    'reg_lambda': 0.00018229924748670902,
    'reg_alpha': 7.984351980892972e-05,
    'n_estimators': 360
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'lasso': 0.7,
    'xgb': 0.3
}
