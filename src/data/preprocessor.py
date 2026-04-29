"""
Data Preprocessing Module
Handles cleaning, missing data, outliers, and transformations
"""
import pandas as pd
import numpy as np
from scipy.stats import skew
from config import MISSING_THRESHOLD, OUTLIER_IDS, MACRO_FEATURES


def handle_missing_data(df, threshold=None):
    """
    Drop columns with more than threshold missing values
    """
    if threshold is None:
        from config import MISSING_THRESHOLD as threshold
    
    missing = df.isnull().sum()
    to_drop = missing[missing > threshold].index
    print(f"Dropping {len(to_drop)} columns with > {threshold} missing values: {list(to_drop)}")
    return df.drop(columns=to_drop)


def remove_outliers(df):
    """
    Remove outlier rows by ID (from config)
    """
    from config import OUTLIER_IDS
    initial_len = len(df)
    df = df[~df['Id'].isin(OUTLIER_IDS)]
    removed = initial_len - len(df)
    print(f"Removed {removed} outlier rows (IDs: {OUTLIER_IDS})")
    return df


def remove_row_missing(df, column='Electrical'):
    """
    Remove rows where a specific column is missing
    """
    if column in df.columns:
        missing_rows = df[df[column].isnull()]
        if len(missing_rows) > 0:
            print(f"Removing {len(missing_rows)} rows with missing {column}")
            df = df.dropna(subset=[column])
    return df


def log_transform_target(df, target='SalePrice'):
    """
    Apply log1p transformation to target variable
    """
    if target in df.columns:
        print(f"Log transforming target: {target}")
        df[target] = np.log1p(df[target])
    return df


def log_transform_skewed_features(df, skew_threshold=0.75):
    """
    Identify numeric features with high skewness and apply log1p transform
    """
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > skew_threshold]
    
    print(f"Log transforming {len(skewed_feats)} skewed features (skew > {skew_threshold})")
    df[skewed_feats.index] = np.log1p(df[skewed_feats.index])
    return df


def create_dummy_variables(df):
    """
    Convert categorical variables to dummy/indicator variables
    """
    print(f"Creating dummy variables. Shape before: {df.shape}")
    df = pd.get_dummies(df)
    print(f"Shape after: {df.shape}")
    return df


def fill_missing_with_mean(df):
    """
    Fill remaining missing values with column means
    """
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"Filling missing values in {len(missing_cols)} columns with mean")
        df = df.fillna(df.mean())
    return df


def preprocess_train(train_df, test_df=None):
    """
    Full preprocessing pipeline for train (and optionally test) data
    Returns processed train, test (if provided), and all_data concatenated
    """
    # Work on copies
    train = train_df.copy()
    if test_df is not None:
        test = test_df.copy()
    else:
        test = None
    
    # 1. Handle missing columns (drop high-missing columns)
    train = handle_missing_data(train)
    if test is not None:
        # Use same columns as train
        test = test[train.columns.intersection(test.columns)]
        # Drop same columns from test
        cols_to_drop = set(train.columns) - set(test.columns)
        if cols_to_drop:
            print(f"Warning: Columns in train but not in test: {cols_to_drop}")
    
    # 2. Remove rows with missing critical columns
    train = remove_row_missing(train, 'Electrical')
    
    # 3. Remove outliers
    train = remove_outliers(train)
    
    # 4. Log transform target (train only)
    train = log_transform_target(train, 'SalePrice')
    
    # 5. Combine for feature transformations
    if test is not None:
        all_data = pd.concat([train.select_dtypes(include=[np.number]).drop('SalePrice', axis=1) if 'SalePrice' in train.columns else train.select_dtypes(include=[np.number]),
                          test.select_dtypes(include=[np.number])])
    else:
        all_data = train.select_dtypes(include=[np.number])
        if 'SalePrice' in train.columns:
            all_data = pd.concat([all_data.drop('SalePrice', axis=1), train[['SalePrice']]], axis=1)
    
    # 6. Log transform skewed numeric features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    if 'SalePrice' in numeric_feats:
        numeric_feats = numeric_feats.drop('SalePrice')
    
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    
    if len(skewed_feats) > 0:
        print(f"Log transforming skewed features in all_data")
        all_data[skewed_feats.index] = np.log1p(all_data[skewed_feats.index])
        # Apply same to train and test
        train[skewed_feats.index] = np.log1p(train[skewed_feats.index])
        if test is not None:
            test[skewed_feats.index] = np.log1p(test[skewed_feats.index])
    
    # 7. Create dummy variables (categorical)
    # For simplicity, we'll handle this separately in feature engineering
    
    # 8. Fill missing with mean
    train = fill_missing_with_mean(train)
    if test is not None:
        test = fill_missing_with_mean(test)
        all_data = fill_missing_with_mean(all_data)
    
    print(f"Preprocessing complete. Train shape: {train.shape}, Test shape: {test.shape if test is not None else 'N/A'}")
    
    if test is not None:
        return train, test, all_data
    else:
        return train, None, all_data
