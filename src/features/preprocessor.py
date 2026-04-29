import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, train_df, test_df):
        self.train = train_df.copy()
        self.test = test_df.copy()
        
        # 🎯 Outlier removal: 98% of top solutions remove these 2 records
        self.train = self.train.drop(
            self.train[(self.train['GrLivArea'] > 4000) & (self.train['SalePrice'] < 300000)].index)
        
        self.combined = pd.concat([self.train, self.test], axis=0).reset_index(drop=True)

    def handle_missing_values(self):
        numeric_cols = self.combined.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.combined[col] = self.combined[col].fillna(self.combined[col].median())

        categorical_cols = self.combined.select_dtypes(include=['object']).columns
        self.combined[categorical_cols] = self.combined[categorical_cols].fillna("None")
        return self

    def add_custom_features(self):
        # Feature Synergies
        self.combined['TotalSF'] = self.combined['TotalBsmtSF'] + self.combined['1stFlrSF'] + self.combined['2ndFlrSF']
        self.combined['HasPool'] = self.combined['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        self.combined['TotalBath'] = (self.combined['FullBath'] + (0.5 * self.combined['HalfBath']) + 
                                     self.combined['BsmtFullBath'] + (0.5 * self.combined['BsmtHalfBath']))
        
        # 🎯 Converting categorical numbers to strings for proper One-Hot Encoding
        self.combined['MSSubClass'] = self.combined['MSSubClass'].astype(str)
        self.combined['YrSold'] = self.combined['YrSold'].astype(str)
        self.combined['MoSold'] = self.combined['MoSold'].astype(str)
        return self

    def fix_skewness_and_scale(self):
        # 1. Log Transform Skewed Numeric Features
        numeric_feats = self.combined.select_dtypes(include=[np.number]).columns.difference(['Id', 'SalePrice'])
        skewed_feats = self.combined[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        high_skew = skewed_feats[abs(skewed_feats) > 0.75].index
        self.combined[high_skew] = np.log1p(self.combined[high_skew])

        # 2. One-Hot Encoding (OHE)
        self.combined = pd.get_dummies(self.combined)

        # 3. ⚔️ The XGBoost Shield: Force 100% Numeric (No strings, no bools)
        for col in self.combined.columns:
            if not np.issubdtype(self.combined[col].dtype, np.number):
                self.combined[col] = self.combined[col].astype(float)
            if self.combined[col].dtype == 'bool':
                self.combined[col] = self.combined[col].astype(int)

        # 4. Standard Scaling: Mandatory for Lasso and ElasticNet
        scaler = StandardScaler()
        cols_to_scale = self.combined.columns.difference(['Id', 'SalePrice'])
        self.combined[cols_to_scale] = scaler.fit_transform(self.combined[cols_to_scale])
        
        return self

    def get_processed_data(self):
        train_len = len(self.train)
        train_proc = self.combined[:train_len].copy()
        test_proc = self.combined[train_len:].copy()
        return train_proc, test_proc