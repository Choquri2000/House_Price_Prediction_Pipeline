"""
Main Pipeline Orchestrator
Run this file to execute the full ML pipeline:
1. Load data
2. Preprocess
3. Feature engineering
4. Train models
5. Generate predictions
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import load_raw_data
from src.data.preprocessor import preprocess_train
from src.features.engineer import engineer_features
from src.models.trainer import train_pipeline
import pickle
import pandas as pd
import numpy as np


def main():
    print("="*60)
    print("HOUSE PRICE PREDICTION PIPELINE")
    print("="*60)
    
    # Step 1: Load raw data
    print("\n[Step 1/5] Loading raw data...")
    train_df, test_df = load_raw_data()
    
    # Step 2: Preprocess data
    print("\n[Step 2/5] Preprocessing data...")
    train_processed, test_processed, all_data = preprocess_train(train_df, test_df)
    
    # Prepare features and target
    X_train = all_data[:train_processed.shape[0]]
    X_test = all_data[train_processed.shape[0]:]
    y = train_processed['SalePrice']
    
    # Step 3: Feature engineering
    print("\n[Step 3/5] Engineering features...")
    X_train = engineer_features(X_train, is_train=True)
    X_test = engineer_features(X_test, is_train=False)
    
    # Align columns between train and test
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    print(f"\nFinal shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Step 4: Train models
    print("\n[Step 4/5] Training models...")
    lasso_model, xgb_model, lasso_pred, xgb_pred, ensemble_pred = train_pipeline(
        X_train, y, X_test, use_optuna=True
    )
    
    # Step 5: Save predictions
    print("\n[Step 5/5] Saving predictions...")
    test_ids = test_df['Id']
    
    # Lasso predictions
    lasso_solution = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": lasso_pred
    })
    lasso_solution.to_csv("lasso_predictions.csv", index=False)
    print("Saved: lasso_predictions.csv")
    
    # XGBoost predictions
    xgb_solution = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": xgb_pred
    })
    xgb_solution.to_csv("xgb_predictions.csv", index=False)
    print("Saved: xgb_predictions.csv")
    
    # Ensemble predictions
    ensemble_solution = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": ensemble_pred
    })
    ensemble_solution.to_csv("ensemble_predictions.csv", index=False)
    print("Saved: ensemble_predictions.csv")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
