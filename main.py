# %% 
import os
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader
from src.features.preprocessor import DataPreprocessor
from src.models.tuner import HyperparameterTuner

# %%
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("🚀 Deploying Pent-Ensemble Final Strike...")
    loader = DataLoader(config_path='configs/config.yaml')

    try:
        # 1. Full Preprocessing Pipeline
        train_df, test_df = loader.load_raw_data()
        preprocessor = DataPreprocessor(train_df, test_df)
        train_cleaned, test_cleaned = (preprocessor
                                       .handle_missing_values()
                                       .add_custom_features()
                                       .fix_skewness_and_scale() # 🚀 ACTIVATED
                                       .get_processed_data())

        target_col = loader.config['model']['target_col']
        id_col = loader.config['model']['id_col']
        
        X = train_cleaned.drop(columns=[target_col, id_col])
        y = np.log1p(train_cleaned[target_col])

        # Validation Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. XGBoost Tuning
        logging.info("🧠 Tuning XGBoost...")
        tuner = HyperparameterTuner(X_train, y_train, X_val, y_val)
        best_params_xgb = tuner.tune(n_trials=20)

        # 3. Training the Pent-Squad
        logging.info("⚔️ Training the 5-Model Ensemble...")

        m_xgb = xgb.XGBRegressor(**best_params_xgb).fit(X_train, y_train)
        m_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=1200, verbosity=-1).fit(X_train, y_train)
        m_cat = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=4, l2_leaf_reg=4, bootstrap_type='Bernoulli', subsample=0.6, silent=True).fit(X_train, y_train)
        m_lasso = LassoCV(alphas=[0.0001, 0.0005, 0.001], cv=5).fit(X_train, y_train)
        m_enet = ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5).fit(X_train, y_train)

        # 📊 Visualization: Top Market Drivers
        sns.set(style="darkgrid")
        plt.style.use("dark_background")
        importances = pd.DataFrame({'Feature': X.columns, 'Importance': m_xgb.feature_importances_})
        importances = importances.sort_values(by='Importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances, palette='Blues_r', hue='Feature', legend=False)
        plt.title('Top 10 Critical Market Drivers')
        plt.show()

        # 4. Blended Prediction
        logging.info("🤝 Blending predictions...")
        test_feats = test_cleaned.drop(columns=[target_col, id_col])
        
        p_xgb = np.expm1(m_xgb.predict(test_feats))
        p_lgb = np.expm1(m_lgb.predict(test_feats))
        p_cat = np.expm1(m_cat.predict(test_feats))
        p_lasso = np.expm1(m_lasso.predict(test_feats))
        p_enet = np.expm1(m_enet.predict(test_feats))

        # ⚖️ Weighted Master Ratio
        final_preds = (0.25 * p_xgb) + (0.25 * p_lgb) + (0.20 * p_cat) + (0.15 * p_lasso) + (0.15 * p_enet)

        # 5. Export Results
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "submission.csv")
        
        pd.DataFrame({id_col: test_cleaned[id_col], target_col: final_preds}).to_csv(output_path, index=False)
        logging.info(f"🏆 Pent-Ensemble Mission Complete: {output_path}")

    except Exception as e:
        logging.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()