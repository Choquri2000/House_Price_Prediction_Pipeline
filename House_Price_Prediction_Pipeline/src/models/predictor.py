import pandas as pd
import numpy as np
import joblib


class Predictor:
    def __init__(self, model_path, config):
        self.model = joblib.load(model_path)
        self.config = config

    def predict(self, test_df):
        target_col = self.config['model']['target_col']
        id_col = self.config['model']['id_col']

        # Drop ID and Target if they exist in the test set
        cols_to_drop = [c for c in [target_col, id_col] if c in test_df.columns]
        features = test_df.drop(columns=cols_to_drop)

        preds_log = self.model.predict(features)
        preds_final = np.expm1(preds_log)

        submission = pd.DataFrame({
            id_col: test_df[id_col],
            target_col: preds_final
        })
        return submission