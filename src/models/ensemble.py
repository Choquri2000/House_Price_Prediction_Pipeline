"""Model definitions and blend weights for the 5-model ensemble."""
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV

# Fixed blend weights (sum to 1). Gradient boosting captures interactions,
# regularised linear models add stability.
WEIGHTS = {"xgb": 0.25, "lgb": 0.25, "cat": 0.20, "lasso": 0.15, "enet": 0.15}

# Used when no Optuna result is available (e.g. cross-validation without tuning).
DEFAULT_XGB_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "gamma": 0.001,
    "random_state": 42,
}


def build_models(xgb_params=None, random_state=42):
    """Return an unfitted dict of the five ensemble members."""
    xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    return {
        "xgb": xgb.XGBRegressor(**xgb_params),
        "lgb": lgb.LGBMRegressor(objective="regression", num_leaves=5, learning_rate=0.05,
                                 n_estimators=1200, verbosity=-1, random_state=random_state),
        "cat": CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=4, l2_leaf_reg=4,
                                 bootstrap_type="Bernoulli", subsample=0.6, silent=True,
                                 random_seed=random_state),
        "lasso": LassoCV(alphas=[0.0001, 0.0005, 0.001], cv=5),
        "enet": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5),
    }


def blend(predictions):
    """Weighted blend of per-model predictions (dict name -> array)."""
    if set(predictions) != set(WEIGHTS):
        raise ValueError(f"Expected predictions for {sorted(WEIGHTS)}, got {sorted(predictions)}")
    return sum(WEIGHTS[name] * np.asarray(pred) for name, pred in predictions.items())
