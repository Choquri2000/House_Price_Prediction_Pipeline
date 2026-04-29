"""
Model Training Module
Handles Lasso, XGBoost, and ensemble training based on EDA insights
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna
import pickle
import os
from config import MODELS_DIR, LASSO_ALPHAS, XGB_PARAMS, ENSEMBLE_WEIGHTS


def rmse_cv(model, X, y, cv=5):
    """
    Calculate RMSE using cross-validation
    """
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv))
    return rmse


def train_lasso(X_train, y, alphas=None):
    """
    Train Lasso model with CV for alpha selection
    """
    if alphas is None:
        alphas = LASSO_ALPHAS
    
    print("\n" + "="*50)
    print("Training Lasso Model")
    print("="*50)
    
    model = LassoCV(alphas=alphas).fit(X_train, y)
    
    # Calculate CV RMSE
    rmse = rmse_cv(model, X_train, y).mean()
    print(f"Lasso RMSE (CV): {rmse:.6f}")
    print(f"Best alpha: {model.alpha_}")
    
    # Get feature importance
    coef = pd.Series(model.coef_, index=X_train.columns)
    n_features = sum(coef != 0)
    print(f"Lasso selected {n_features} features (eliminated {sum(coef == 0)} features)")
    
    # Show top features
    print("\nTop 10 features (positive):")
    print(coef.sort_values(ascending=False).head(10))
    print("\nTop 10 features (negative):")
    print(coef.sort_values(ascending=True).head(10))
    
    return model, rmse


def objective_xgb(trial, X_train, y_train):
    """
    Optuna objective function for XGBoost hyperparameter tuning
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 100),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 100)
    }
    
    model = xgb.XGBRegressor(n_estimators=360, **params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    
    mse = mean_squared_error(y_train, y_pred)
    return mse


def train_xgboost(X_train, y, n_trials=100):
    """
    Train XGBoost with Optuna hyperparameter tuning
    """
    print("\n" + "="*50)
    print("Training XGBoost Model with Optuna")
    print("="*50)
    
    # Create study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_xgb(trial, X_train, y), n_trials=n_trials)
    
    print(f"\nBest trial: RMSE = {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    
    # Train final model with best params
    best_params = study.best_params
    best_params['n_estimators'] = 360
    
    model = xgb.XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 important features:")
    print(importance.head(10))
    
    return model, study.best_value


def create_ensemble_predictions(lasso_model, xgb_model, X_test):
    """
    Create ensemble predictions (weighted average)
    """
    weights = ENSEMBLE_WEIGHTS
    
    lasso_pred = np.expm1(lasso_model.predict(X_test))  # Reverse log transform
    xgb_pred = np.expm1(xgb_model.predict(X_test))    # Reverse log transform
    
    ensemble_pred = weights['lasso'] * lasso_pred + weights['xgb'] * xgb_pred
    
    print(f"\nEnsemble predictions created (Lasso: {weights['lasso']}, XGB: {weights['xgb']})")
    
    return lasso_pred, xgb_pred, ensemble_pred


def save_model(model, name):
    """
    Save trained model to disk
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")
    return model_path


def load_model(name):
    """
    Load trained model from disk
    """
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from: {model_path}")
        return model
    else:
        print(f"Model not found: {model_path}")
        return None


def train_pipeline(X_train, y, X_test=None, use_optuna=True):
    """
    Full training pipeline: Lasso + XGBoost + Ensemble
    """
    print("\n" + "="*50)
    print("FULL TRAINING PIPELINE")
    print("="*50)
    
    # 1. Train Lasso
    lasso_model, lasso_rmse = train_lasso(X_train, y)
    
    # 2. Train XGBoost (with or without Optuna)
    if use_optuna:
        xgb_model, xgb_rmse = train_xgboost(X_train, y, n_trials=100)
    else:
        print("\nTraining XGBoost with config params (no Optuna)...")
        from config import XGB_PARAMS
        xgb_model = xgb.XGBRegressor(**XGB_PARAMS, random_state=42)
        xgb_model.fit(X_train, y)
        xgb_rmse = rmse_cv(xgb_model, X_train, y).mean()
        print(f"XGBoost RMSE (CV): {xgb_rmse:.6f}")
    
    # 3. Save models
    save_model(lasso_model, "lasso_model")
    save_model(xgb_model, "xgb_model")
    
    # 4. Create predictions if test data provided
    if X_test is not None:
        print("\nCreating predictions...")
        lasso_pred, xgb_pred, ensemble_pred = create_ensemble_predictions(
            lasso_model, xgb_model, X_test
        )
        return lasso_model, xgb_model, lasso_pred, xgb_pred, ensemble_pred
    
    return lasso_model, xgb_model
