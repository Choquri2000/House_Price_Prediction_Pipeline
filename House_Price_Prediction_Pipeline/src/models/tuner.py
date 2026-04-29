import optuna
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

class HyperparameterTuner:
    # Ensure it takes exactly 4 data arguments + self
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def objective(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**param)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_val)
        return np.sqrt(mean_squared_error(self.y_val, preds))

    def tune(self, n_trials=50):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params