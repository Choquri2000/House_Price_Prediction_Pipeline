# %% 
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader
from src.features.preprocessor import DataPreprocessor
from src.models.tuner import HyperparameterTuner
from src.models.ensemble import build_models, blend

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
        n_trials = loader.config.get('tuning', {}).get('n_trials', 20)
        best_params_xgb = tuner.tune(n_trials=n_trials)

        # 3. Training the Pent-Squad
        logging.info("⚔️ Training the 5-Model Ensemble...")

        models = build_models(best_params_xgb)
        for model in models.values():
            model.fit(X_train, y_train)
        m_xgb = models['xgb']

        # 📊 Visualization: Top Market Drivers
        sns.set(style="darkgrid")
        plt.style.use("dark_background")
        importances = pd.DataFrame({'Feature': X.columns, 'Importance': m_xgb.feature_importances_})
        importances = importances.sort_values(by='Importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances, palette='Blues_r', hue='Feature', legend=False)
        plt.title('Top 10 Critical Market Drivers')
        plt.tight_layout()
        os.makedirs("data/processed", exist_ok=True)
        plt.savefig("data/processed/feature_importance.png", dpi=150)
        plt.show()

        # 4. Blended Prediction
        logging.info("🤝 Blending predictions...")
        test_feats = test_cleaned.drop(columns=[target_col, id_col])
        
        # Blend in price space: expm1 each prediction, then apply the fixed weights
        preds = {name: np.expm1(m.predict(test_feats)) for name, m in models.items()}
        final_preds = blend(preds)

        # 5. Export Results
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "submission.csv")
        
        pd.DataFrame({id_col: test_cleaned[id_col], target_col: final_preds}).to_csv(output_path, index=False)
        logging.info(f"🏆 Pent-Ensemble Mission Complete: {output_path}")

    except Exception:
        logging.exception("❌ Pipeline failed")
        raise

if __name__ == "__main__":
    main()