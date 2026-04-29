"""
Feature Engineering Module
Creates new features based on EDA insights
"""
import pandas as pd
import numpy as np


def create_domain_features(df):
    """
    Create domain-specific features based on EDA findings
    """
    df_new = df.copy()
    
    # House age features
    if 'YearBuilt' in df.columns:
        # Assuming data is from 2010 (Kaggle dataset)
        df_new['HouseAge'] = 2010 - df['YearBuilt']
        print("Created feature: HouseAge")
    
    if 'YearRemodAdd' in df.columns:
        df_new['RemodAge'] = 2010 - df['YearRemodAdd']
        df_new['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
        print("Created features: RemodAge, IsRemodeled")
    
    # Total square footage
    if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df_new['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        print("Created feature: TotalSF")
    
    # Total bathrooms
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    available_bath = [col for col in bath_cols if col in df.columns]
    if available_bath:
        df_new['TotalBathrooms'] = sum(df[col] for col in available_bath)
        print(f"Created feature: TotalBathrooms (from {available_bath})")
    
    # Total porch area
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    available_porch = [col for col in porch_cols if col in df.columns]
    if available_porch:
        df_new['TotalPorchSF'] = sum(df[col] for col in available_porch)
        print(f"Created feature: TotalPorchSF")
    
    # Quality × Area interactions (from EDA: strong correlation)
    if all(col in df.columns for col in ['OverallQual', 'GrLivArea']):
        df_new['Qual_x_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
        print("Created feature: Qual_x_GrLivArea")
    
    if all(col in df.columns for col in ['OverallQual', 'TotalBsmtSF']):
        df_new['Qual_x_TotalBsmtSF'] = df['OverallQual'] * df['TotalBsmtSF']
        print("Created feature: Qual_x_TotalBsmtSF")
    
    print(f"Feature engineering complete. New shape: {df_new.shape}")
    return df_new


def handle_multicollinearity(df):
    """
    Handle multicollinearity detected in EDA:
    - TotalBsmtSF vs 1stFlrSF (keep TotalBsmtSF)
    - GarageCars vs GarageArea (keep GarageCars)
    """
    df_new = df.copy()
    
    # Drop 1stFlrSF if TotalBsmtSF exists (high correlation detected)
    if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns:
        df_new = df_new.drop('1stFlrSF', axis=1)
        print("Dropped '1stFlrSF' (multicollinear with TotalBsmtSF)")
    
    # Drop GarageArea if GarageCars exists (redundant info)
    if 'GarageCars' in df.columns and 'GarageArea' in df.columns:
        df_new = df_new.drop('GarageArea', axis=1)
        print("Dropped 'GarageArea' (multicollinear with GarageCars)")
    
    return df_new


def create_price_bins(df, target='SalePrice', n_bins=10):
    """
    Create price bins for stratified analysis (from EDA)
    """
    if target in df.columns:
        df_new = df.copy()
        df_new[f'{target}_bin'] = pd.qcut(df[target], n_bins, labels=False, duplicates='drop')
        print(f"Created {target}_bin with {df_new[f'{target}_bin'].nunique()} bins")
        return df_new
    return df


def engineer_features(df, is_train=True):
    """
    Full feature engineering pipeline
    """
    print("\n" + "="*50)
    print("Feature Engineering Pipeline")
    print("="*50)
    
    # 1. Create domain features
    df = create_domain_features(df)
    
    # 2. Handle multicollinearity
    df = handle_multicollinearity(df)
    
    # 3. Create dummy variables for categorical features
    print("\nCreating dummy variables...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Found {len(categorical_cols)} categorical columns")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"Shape after dummies: {df.shape}")
    
    return df
