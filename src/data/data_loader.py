import pandas as pd
import yaml
import logging
from pathlib import Path

# Set up basic logging (better than print statements!)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataLoader:
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """Initializes the DataLoader by reading the config file."""
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Reads the yaml configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logging.error(f"Config file not found at {self.config_path}")
            raise

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train and test datasets and performs basic validation."""
        train_path = Path(self.config['data']['raw_train_path'])
        test_path = Path(self.config['data']['raw_test_path'])

        logging.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)

        logging.info(f"Loading testing data from {test_path}")
        test_df = pd.read_csv(test_path)

        self._validate_data(train_df, is_train=True)
        self._validate_data(test_df, is_train=False)

        return train_df, test_df

    def _validate_data(self, df: pd.DataFrame, is_train: bool):
        """Validates that expected columns exist and data is not empty."""
        if df.empty:
            raise ValueError("The loaded dataframe is empty!")

        id_col = self.config['model']['id_col']
        if id_col not in df.columns:
            raise ValueError(f"Missing mandatory ID column: {id_col}")

        if is_train:
            target_col = self.config['model']['target_col']
            if target_col not in df.columns:
                raise ValueError(f"Missing target column: {target_col} in training data!")

        logging.info(f"Validation passed. Data shape: {df.shape}")