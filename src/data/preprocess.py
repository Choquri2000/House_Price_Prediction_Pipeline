import pandas as pd
import yaml


class DataPreprocessor:
    def __init__(self, config_path: str = 'configs/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes specific outlier rows based on their IDs from the config."""
        outlier_ids = self.config['preprocessing']['outlier_ids']
        id_col = self.config['model']['id_col']

        # Filter out the rows where the ID is in our outlier list
        df_cleaned = df[~df[id_col].isin(outlier_ids)].copy()
        print(f"Removed {len(df) - len(df_cleaned)} outliers.")
        return df_cleaned

    def handle_missing_values(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Drops columns with too many missing values and fills the rest with the mean."""
        threshold = self.config['preprocessing']['missing_threshold']

        # 1. Find columns to drop in Train (missing > threshold)
        missing_train = train_df.isnull().sum()
        cols_to_drop_train = missing_train[missing_train > threshold].index.tolist()

        # 2. Find columns to drop in Test (missing > threshold)
        missing_test = test_df.isnull().sum()
        cols_to_drop_test = missing_test[missing_test > threshold].index.tolist()

        # Combine the lists and remove duplicates
        cols_to_drop = list(set(cols_to_drop_train + cols_to_drop_test))
        print(f"Dropping {len(cols_to_drop)} columns due to missing values > {threshold}.")

        train_df = train_df.drop(columns=cols_to_drop)
        test_df = test_df.drop(columns=cols_to_drop)

        # 3. Drop the row where 'Electrical' is missing (only in train)
        if 'Electrical' in train_df.columns:
            train_df = train_df.dropna(subset=['Electrical'])

        # 4. Fill remaining missing values with the MEAN of each column
        # Note: In ML, we must fill numerical missing values
        numeric_cols_train = train_df.select_dtypes(include=['number']).columns
        numeric_cols_test = test_df.select_dtypes(include=['number']).columns

        train_df[numeric_cols_train] = train_df[numeric_cols_train].fillna(train_df[numeric_cols_train].mean())
        test_df[numeric_cols_test] = test_df[numeric_cols_test].fillna(test_df[numeric_cols_test].mean())

        print("Filled remaining numerical missing values with column means.")
        return train_df, test_df