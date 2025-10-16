"""Aggregate raw CSV files into a processed dataset."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from bisindo.utils import project_root

SEED = 42
np.random.seed(SEED)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "extract_features.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)


def collect_csv_files(raw_dir: Path) -> List[Path]:
    return sorted(raw_dir.glob("*.csv"))


def main() -> None:
    project_dir = project_root()
    raw_dir = project_dir / "data" / "raw"
    processed_dir = project_dir / "data" / "processed"
    models_dir = project_dir / "models"
    runs_dir = project_dir / "runs"

    setup_logging(runs_dir)

    csv_files = collect_csv_files(raw_dir)
    if not csv_files:
        logging.error("No CSV files found in %s", raw_dir)
        sys.exit(1)

    logging.info("Found %d raw CSV files", len(csv_files))

    dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(dataframes, ignore_index=True)
    logging.info("Initial combined rows: %d", len(df))

    before_dedup = len(df)
    df = df.drop_duplicates()
    logging.info("Removed %d duplicate rows", before_dedup - len(df))

    df = df.dropna()
    logging.info("Rows after dropping NaN: %d", len(df))

    if "label" not in df.columns:
        logging.error("Dataset missing 'label' column")
        sys.exit(1)

    feature_cols = sorted(
        (col for col in df.columns if col.startswith("f")),
        key=lambda x: int(x[1:]) if x[1:].isdigit() else x,
    )
    if not feature_cols:
        logging.error("No feature columns starting with 'f' found")
        sys.exit(1)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()

    assert X.ndim == 2, "Features must be a 2D array"
    assert X.shape[0] == y.shape[0], "Feature and label length mismatch"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_dir.mkdir(parents=True, exist_ok=True)
    dump(scaler, models_dir / "scaler.joblib")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    dump(label_encoder, models_dir / "label_encoder.joblib")

    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = processed_dir / "dataset.csv"
    df.to_csv(dataset_csv, index=False)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.3,
        random_state=SEED,
        stratify=y_encoded,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp,
    )

    np.save(processed_dir / "train_X.npy", X_train)
    np.save(processed_dir / "train_y.npy", y_train)
    np.save(processed_dir / "val_X.npy", X_val)
    np.save(processed_dir / "val_y.npy", y_val)
    np.save(processed_dir / "test_X.npy", X_test)
    np.save(processed_dir / "test_y.npy", y_test)

    logging.info("Classes: %d", len(label_encoder.classes_))
    logging.info("Feature dimension: %d", X.shape[1])
    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        logging.info("Label %s: %d samples", label, count)

    logging.info(
        "Train/Val/Test shapes: %s, %s, %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )


if __name__ == "__main__":
    main()
