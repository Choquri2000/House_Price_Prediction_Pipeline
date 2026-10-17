import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.data_loader import DataLoader
from src.features.preprocessor import DataPreprocessor
from src.models.ensemble import WEIGHTS, blend


def make_frames(n_train=12, n_test=4, seed=0):
    """Tiny frames with the Ames columns the preprocessor touches."""
    rng = np.random.default_rng(seed)

    def frame(n, start_id, with_target):
        df = pd.DataFrame({
            "Id": range(start_id, start_id + n),
            "MSSubClass": rng.choice([20, 60, 120], n),
            "GrLivArea": rng.integers(800, 2500, n),
            "TotalBsmtSF": rng.integers(0, 1500, n).astype(float),
            "1stFlrSF": rng.integers(500, 1500, n),
            "2ndFlrSF": rng.integers(0, 900, n),
            "PoolArea": rng.choice([0, 0, 0, 300], n),
            "FullBath": rng.integers(1, 3, n),
            "HalfBath": rng.integers(0, 2, n),
            "BsmtFullBath": rng.integers(0, 2, n),
            "BsmtHalfBath": rng.integers(0, 2, n),
            "YrSold": rng.choice([2008, 2009], n),
            "MoSold": rng.integers(1, 13, n),
            "Neighborhood": rng.choice(["NAmes", "CollgCr", None], n),
        })
        df.loc[0, "TotalBsmtSF"] = np.nan
        if with_target:
            df["SalePrice"] = rng.integers(100_000, 400_000, n)
        return df

    return make_frames_train(frame, n_train), frame(n_test, 1000, False)


def make_frames_train(frame, n):
    return frame(n, 1, True)


def run_preprocessor(train, test):
    return (DataPreprocessor(train, test)
            .handle_missing_values()
            .add_custom_features()
            .fix_skewness_and_scale()
            .get_processed_data())


def test_known_outliers_are_dropped():
    train, test = make_frames()
    train.loc[0, ["GrLivArea", "SalePrice"]] = [5000, 150_000]
    out_train, _ = run_preprocessor(train, test)
    assert len(out_train) == len(train) - 1


def test_features_added_and_no_missing_values():
    train, test = make_frames()
    out_train, out_test = run_preprocessor(train, test)
    for col in ("TotalSF", "TotalBath", "HasPool"):
        assert col in out_train.columns
    assert not out_train.drop(columns=["SalePrice"]).isna().any().any()
    assert not out_test.drop(columns=["SalePrice"]).isna().any().any()


def test_output_is_fully_numeric_and_aligned():
    train, test = make_frames()
    out_train, out_test = run_preprocessor(train, test)
    assert list(out_train.columns) == list(out_test.columns)
    assert all(np.issubdtype(t, np.number) for t in out_train.dtypes)
    assert len(out_test) == len(test)


def test_blend_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_blend_is_weighted_average():
    preds = {name: np.full(3, 100.0) for name in WEIGHTS}
    assert blend(preds) == pytest.approx(np.full(3, 100.0))
    with pytest.raises(ValueError):
        blend({"xgb": np.ones(3)})


def write_config(tmp_path, train, test):
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    cfg = {
        "data": {"raw_train_path": str(tmp_path / "train.csv"), "raw_test_path": str(tmp_path / "test.csv")},
        "model": {"target_col": "SalePrice", "id_col": "Id"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_loader_reads_valid_data(tmp_path):
    train, test = make_frames()
    train_df, test_df = DataLoader(write_config(tmp_path, train, test)).load_raw_data()
    assert len(train_df) == len(train) and len(test_df) == len(test)


def test_loader_rejects_missing_target(tmp_path):
    train, test = make_frames()
    cfg = write_config(tmp_path, train.drop(columns=["SalePrice"]), test)
    with pytest.raises(ValueError, match="target"):
        DataLoader(cfg).load_raw_data()
