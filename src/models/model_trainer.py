import xgboost as xgb
import joblib
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('TkAgg') # 🚀 This forces a physical window
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def prepare_data(self, df):
        target = self.config['model']['target_col']
        # We also drop SalePrice and Id
        X = df.drop(columns=[target, self.config['model']['id_col']])
        y = np.log1p(df[target])
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        logging.info("Training XGBoost model...")
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_val, y_val):
        preds = self.model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        logging.info(f"Validation RMSE (Log Scale): {rmse:.4f}")
        return rmse

    def plot_importance(self, X):
        sns.set(style="darkgrid")
        plt.style.use("dark_background")

        importances = self.model.feature_importances_
        data = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        data = data.sort_values(by='Importance', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        # Added hue and legend=False to fix the warning
        sns.barplot(x='Importance', y='Feature', data=data, palette='Blues_r', hue='Feature', legend=False)
        plt.title('Top 10 Critical Market Drivers')
        plt.tight_layout()
        plt.show()

    def save_model(self, path="models/xgb_model.pkl"):
        import os
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, path)
        logging.info(f"Model saved to {path}")