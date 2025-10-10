"""
Model loading functionality.

Contains functions for loading trained models from checkpoints
and retrieving their configurations.
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import LinearRegressionModel
from train_model.models import Models
from typing import Tuple, Any, Optional

logger = logging.getLogger(__name__)


def _create_dummy_dataset(
    input_size: int, output_size: int, with_static_covariates: bool = False
) -> TimeSeries:
    """Create a minimal dummy dataset to initialize the model.

    Args:
        input_size: Input chunk length for the model
        output_size: Output chunk length for the model
        with_static_covariates: Whether to add static covariates to the series

    Returns:
        A minimal TimeSeries for model initialization
    """
    # Create a dummy time series with enough length for the model to initialize
    # We need at least input_size + output_size points
    n_points = input_size + output_size + 10  # Add some buffer
    dummy_values = np.random.randn(n_points, 1)  # Single component
    time_index = pd.date_range(start="2020-01-01", periods=n_points, freq="D")
    series = TimeSeries.from_times_and_values(times=time_index, values=dummy_values)

    # Add static covariates if needed (matching the structure used during training)
    if with_static_covariates:
        # Create dummy static covariates
        static_cov = pd.DataFrame({"feature_1": [0.5], "feature_2": [0.5]})
        series = series.with_static_covariates(static_cov)

    return series


def load_model_from_checkpoint(
    model_name: str, model_path: str, encoders_path: Optional[str]
) -> Tuple[Any, int, int]:
    """Load a trained model from checkpoint and get its input/output sizes.

    Args:
        model_name: Name of the model type (nhits, nlinear, tide, linear_regression)
        model_path: Path to the model checkpoint (state dict for nhits/nlinear/tide, model file for linear_regression)

    Returns:
        Tuple of (loaded model, input_size, output_size)
    """
    logger.info(f"Loading {model_name} model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load model based on type
    if model_name == "nhits":
        # Create a new model instance with the original configuration
        model_config = Models.NHiTSModelConfig()
        model, _, _, input_size, output_size = model_config.setup_model()

        # Fit on dummy data to initialize the PLForecastingModule
        # We need to pass a validation series to satisfy the early stopping callback
        logger.info("Initializing model with dummy data...")
        dummy_train = _create_dummy_dataset(input_size, output_size)
        dummy_val = _create_dummy_dataset(input_size, output_size)
        model.fit(dummy_train, val_series=dummy_val, epochs=1, verbose=False)

        # Load state dict into the initialized model
        state_dict = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )
        model.model.load_state_dict(state_dict)
        # Set to eval mode
        model.model.eval()
        logger.info("Loaded NHiTS model from state dict and set to eval mode")

    elif model_name == "nlinear":
        model_config = Models.NLinearConfig()
        model, _, _, input_size, output_size = model_config.setup_model()

        # NLinear was trained with static covariates, so include them in dummy data
        logger.info("Initializing model with dummy data...")
        dummy_train = _create_dummy_dataset(
            input_size, output_size, with_static_covariates=True
        )
        dummy_val = _create_dummy_dataset(
            input_size, output_size, with_static_covariates=True
        )
        model.fit(dummy_train, val_series=dummy_val, epochs=1, verbose=False)

        state_dict = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )
        model.model.load_state_dict(state_dict)
        model.model.eval()
        logger.info("Loaded NLinear model from state dict and set to eval mode")

    elif model_name == "tide":

        model_config = Models.TiDEConfig()
        model, _, _, input_size, output_size = model_config.setup_model()

        logger.info("Initializing model with dummy data...")
        dummy_train = _create_dummy_dataset(
            input_size, output_size, with_static_covariates=True
        )
        dummy_val = _create_dummy_dataset(
            input_size, output_size, with_static_covariates=True
        )
        model.fit(dummy_train, val_series=dummy_val, epochs=1, verbose=False)


        state_dict = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )
        model.model.load_state_dict(state_dict)

        # Load the extracted encoders/scalers from the checkpoint directory
        if encoders_path and os.path.exists(encoders_path):
            logger.info(f"Loading fitted encoders from: {encoders_path}")
            try:
                # Try loading with torch.load first
                loaded_encoders = torch.load(
                    encoders_path, map_location=torch.device("cpu"), weights_only=False
                )

                # Check if it's a dict (wrong format) or an encoder object
                if isinstance(loaded_encoders, dict):
                    logger.warning("Loaded encoders as dict - this may cause issues")
                    logger.warning("The encoders file may have been saved incorrectly")
                    logger.warning(
                        "TiDE will use encoders fitted on dummy data - performance may be degraded!"
                    )
                else:
                    model.encoders = loaded_encoders
                    logger.info(
                        f"Restored fitted encoders/scalers for TiDE model (type: {type(loaded_encoders).__name__})"
                    )
            except Exception as e:
                logger.error(f"Failed to load encoders: {e}")
                logger.warning(
                    "TiDE will use encoders fitted on dummy data - performance may be degraded!"
                )
        else:
            logger.warning(f"Encoders file not found: {encoders_path}")
            logger.warning(
                "TiDE will use encoders fitted on dummy data - performance may be degraded!"
            )


        model.model.eval()
        logger.info("Loaded TiDE model from state dict and set to eval mode")

    elif model_name == "linear_regression":
        # Linear regression still uses the old method
        model = LinearRegressionModel.load(model_path)
        _, _, _, input_size, output_size = Models.LinearRegressionConfig().setup_model()
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    logger.info(
        f"{model_name} model loaded successfully (input_size={input_size}, output_size={output_size})"
    )
    return model, input_size, output_size
