"""K-fold cross-validation of every ensemble member and of the blend.

Usage:
    python evaluate.py              # 5 folds, default XGBoost params
    python evaluate.py --folds 10

Scores are RMSE on log1p(SalePrice), i.e. the Kaggle metric. The blend is built
the same way as in main.py: predictions are converted back to prices, weighted,
then compared with the true price on the log scale.
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.data.data_loader import DataLoader
from src.features.preprocessor import DataPreprocessor
from src.models.ensemble import build_models, blend


def rmse_log(y_log_true, price_pred):
    return float(np.sqrt(np.mean((y_log_true - np.log1p(price_pred)) ** 2)))


def cross_validate(X, y_log, n_splits=5, seed=42):
    """Return a DataFrame with per-fold RMSE for each model and the blend."""
    rows = []
    for fold, (tr, va) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(X), 1):
        X_tr, X_va, y_tr, y_va = X.iloc[tr], X.iloc[va], y_log.iloc[tr], y_log.iloc[va]
        models = build_models(random_state=seed)
        preds = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds[name] = np.expm1(model.predict(X_va))
        scores = {name: rmse_log(y_va, p) for name, p in preds.items()}
        scores["blend"] = rmse_log(y_va, blend(preds))
        rows.append({"fold": fold, **scores})
        logging.info("fold %d: %s", fold, {k: round(v, 4) for k, v in scores.items()})
    return pd.DataFrame(rows).set_index("fold")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    loader = DataLoader(config_path=args.config)
    train_df, test_df = loader.load_raw_data()
    train_cleaned, _ = (DataPreprocessor(train_df, test_df)
                        .handle_missing_values()
                        .add_custom_features()
                        .fix_skewness_and_scale()
                        .get_processed_data())
    target_col = loader.config["model"]["target_col"]
    id_col = loader.config["model"]["id_col"]
    X = train_cleaned.drop(columns=[target_col, id_col]).reset_index(drop=True)
    y = np.log1p(train_cleaned[target_col]).reset_index(drop=True)

    scores = cross_validate(X, y, n_splits=args.folds)
    summary = pd.DataFrame({"mean": scores.mean(), "std": scores.std()}).round(4)
    print(f"\n{args.folds}-fold CV RMSE (log scale)\n{summary.to_string()}")
    os.makedirs("data/processed", exist_ok=True)
    summary.to_csv("data/processed/cv_scores.csv")


if __name__ == "__main__":
    main()
