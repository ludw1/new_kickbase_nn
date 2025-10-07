"""
Data loading and preprocessing module.

Contains functions for loading, transforming, and preprocessing time series data
with static covariates.
"""

import numpy as np
import pandas as pd
import logging
from darts import TimeSeries
from sklearn.preprocessing import RobustScaler
from transform_data import transform_data
from config import Config
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)
def julian_to_date(julian_date: int) -> str:
    """Convert a Julian date to a standard date format (YYYY-MM-DD)."""
    reference_date = datetime(1970, 1, 1)
    converted_date = reference_date + timedelta(days=julian_date)
    return converted_date.strftime("%d.%m.%Y")

def load_and_preprocess_data(
    data_file: str = "all_player_data.json",
) -> tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Load and preprocess data using actual values with series-based split.

    Instead of splitting each time series temporally, we split the players themselves
    into train/val/test sets. This allows the model to learn from complete seasonal
    patterns and test generalization to entirely new players.
    """
    raw_data, static_cov_data, first_date = transform_data(data_file)
    logger.info(f"Processing {len(raw_data)} player time series...")
    # Set random seed for reproducible splits

    np.random.seed(Config.SEED)

    # Shuffle the player indices
    num_players = len(raw_data)
    indices = np.random.permutation(num_players)

    # Calculate split points
    train_end = int(num_players * Config.TRAIN_SPLIT)
    val_end = train_end + int(num_players * Config.VAL_SPLIT)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    logger.info(
        f"Split: {len(train_indices)} train players, {len(val_indices)} val players, {len(test_indices)} test players"
    )

    train_series_unscaled = []
    val_series_unscaled = []
    test_series_unscaled = []
    train_static_cov = []
    val_static_cov = []
    test_static_cov = []
    filtered = 0

    def process_series(player_values):
        """Use actual values directly and create TimeSeries."""
        # Use actual values directly (no differencing)
        values = np.array(player_values)

        # Create a datetime index starting from a reference date
        # Assuming daily data
        time_index = pd.date_range(
            start=julian_to_date(first_date), 
            periods=len(values), 
            freq='D'
        )
        
        # Create TimeSeries with datetime index
        series = TimeSeries.from_times_and_values(
            times=time_index,
            values=values
        )

        return series

    # Process training players (convert to differences)
    for idx in train_indices:
        try:
            series = process_series(raw_data[idx])
            static_cov = pd.DataFrame([static_cov_data[idx]])

            # Check if series is long enough for training
            if len(series) > Config.INPUT_SIZE + Config.OUTPUT_SIZE:
                train_series_unscaled.append(series)
                train_static_cov.append(static_cov)
            else:
                filtered += 1
        except Exception as e:
            logger.warning(f"Failed to process training player {idx}: {e}")
            filtered += 1

    # Process validation players (convert to differences)
    for idx in val_indices:
        try:
            series = process_series(raw_data[idx])
            static_cov = pd.DataFrame([static_cov_data[idx]])

            if len(series) > Config.INPUT_SIZE + Config.OUTPUT_SIZE:
                val_series_unscaled.append(series)
                val_static_cov.append(static_cov)
            else:
                filtered += 1
        except Exception as e:
            logger.warning(f"Failed to process validation player {idx}: {e}")
            filtered += 1

    # Process test players (convert to differences)
    for idx in test_indices:
        try:
            series = process_series(raw_data[idx])
            static_cov = pd.DataFrame([static_cov_data[idx]])

            if len(series) > Config.INPUT_SIZE + Config.OUTPUT_SIZE:
                test_series_unscaled.append(series)
                test_static_cov.append(static_cov)
            else:
                filtered += 1
        except Exception as e:
            logger.warning(f"Failed to process test player {idx}: {e}")
            filtered += 1

    logger.info(f"Filtered {filtered} series (too short)")

    logger.info(
        f"Processed counts: {len(train_series_unscaled)} train, {len(val_series_unscaled)} val, {len(test_series_unscaled)} test series"
    )
    logger.info(
        f"Static covariates: {len(train_static_cov)} train, {len(val_static_cov)} val, {len(test_static_cov)} test"
    )

    # Analyze and handle outliers
    all_train_values = np.concatenate(
        [s.values() for s in train_series_unscaled]
    )

    logger.info("\nTraining values statistics:")
    logger.info(f"  Mean: {np.mean(all_train_values):.4f}")
    logger.info(f"  Std: {np.std(all_train_values):.4f}")
    logger.info(f"  Median: {np.median(all_train_values):.4f}")
    logger.info(f"  Min: {np.min(all_train_values):.4f}")
    logger.info(f"  Max: {np.max(all_train_values):.4f}")
    logger.info(f"  25th percentile: {np.percentile(all_train_values, 25):.4f}")
    logger.info(f"  75th percentile: {np.percentile(all_train_values, 75):.4f}")
    logger.info(f"  99th percentile: {np.percentile(all_train_values, 99):.4f}")
    logger.info(f"  1st percentile: {np.percentile(all_train_values, 1):.4f}")

    # Clip extreme outliers at 1st and 99th percentile
    lower_bound = np.percentile(all_train_values, 1)
    upper_bound = np.percentile(all_train_values, 99)
    logger.info(f"\nClipping outliers to range [{lower_bound:.0f}, {upper_bound:.0f}]")

    def clip_series(series_list):
        clipped_list = []
        for s in series_list:
            clipped_vals = np.clip(s.values(), lower_bound, upper_bound)
            clipped_list.append(
                TimeSeries.from_times_and_values(
                    s.time_index, clipped_vals, columns=s.columns
                )
            )
        return clipped_list

    train_series_clipped = clip_series(train_series_unscaled)
    val_series_clipped = clip_series(val_series_unscaled)
    test_series_clipped = clip_series(test_series_unscaled)

    # Use RobustScaler which is resistant to outliers (uses median and IQR)
    scaler = RobustScaler()
    all_train_clipped = np.concatenate([s.values() for s in train_series_clipped])
    scaler.fit(all_train_clipped)

    logger.info(f"After clipping - Mean: {np.mean(all_train_clipped):.4f}, Std: {np.std(all_train_clipped):.4f}")

    def scale_series(series_list, fitted_scaler):
        scaled_list = []
        for s in series_list:
            scaled_vals = fitted_scaler.transform(s.values())
            scaled_list.append(
                TimeSeries.from_times_and_values(
                    s.time_index, scaled_vals, columns=s.columns
                )
            )
        return scaled_list

    logger.info("\nTransforming all datasets with RobustScaler...")
    train_series = scale_series(train_series_clipped, scaler)
    val_series = scale_series(val_series_clipped, scaler)
    test_series = scale_series(test_series_clipped, scaler)

    return train_series, val_series, test_series, train_static_cov, val_static_cov, test_static_cov