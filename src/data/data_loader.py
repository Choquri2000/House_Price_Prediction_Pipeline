"""
Data Loading Module
Handles loading train and test data using config paths
"""
import pandas as pd
import os
from config import RAW_DATA_DIR, TRAIN_FILE, TEST_FILE


def load_raw_data():
    """
    Load raw train and test data from configured paths
    
    Returns:
        train_df (DataFrame): Raw training data
        test_df (DataFrame): Raw test data
    """
    train_path = os.path.join(RAW_DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(RAW_DATA_DIR, TEST_FILE)
    
    print(f"Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df


def get_macro_features():
    """
    Returns list of macroeconomic features added to the dataset
    
    Returns:
        list: Macro feature column names
    """
    from config import MACRO_FEATURES
    return MACRO_FEATURES
