import json
from get_data.get_player_data import get_api_data
import logging
from darts import TimeSeries
from evaluate_model.model_loading import load_model_from_checkpoint
from inference_pipeline.transform_data import transform_data
from train_model.utils import setup_directories
from config import EvaluationConfig, PipelineConfig, TrainingConfig
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Any

logger = logging.getLogger(__name__)


def julian_to_date(julian_date: int) -> str:
    """Convert a Julian date to a standard date format (YYYY-MM-DD)."""
    reference_date = datetime(1970, 1, 1)
    converted_date = reference_date + timedelta(days=julian_date)
    return converted_date.strftime("%d.%m.%Y")


def process_series(player_values, first_date):
    values = np.array(player_values)

    # Create a datetime index starting from a reference date
    # Assuming daily data
    time_index = pd.date_range(
        start=julian_to_date(first_date), periods=len(values), freq="D"
    )

    series = TimeSeries.from_times_and_values(times=time_index, values=values)

    return series


def load_models_for_inference() -> dict[str, dict]:
    """Main inference function - loads and prepares all models."""

    logger.info("Starting comprehensive model inference...")
    logger.info(
        f"Prediction length multiplier: {EvaluationConfig.PREDICTION_LENGTH_MULTIPLIER}x"
    )
    tide_encoder_path = None
    # Check which models are available
    available_models = {}
    for model_name, model_path in EvaluationConfig.MODEL_PATHS.items():
        if os.path.exists(model_path):
            if "encoder" in model_path:
                tide_encoder_path = model_path
            else:
                available_models[model_name] = model_path
            logger.info(f"Found {model_name} model")
        else:
            logger.warning(f"Model not found: {model_name}")

    if not available_models:
        logger.error("No model checkpoints found! Please train models first.")
        return {}

    # Load all models and their configurations
    models_dict = {}
    for model_name, model_path in available_models.items():
        try:
            model, input_size, output_size = load_model_from_checkpoint(
                model_name, model_path, tide_encoder_path
            )
            models_dict[model_name] = {
                "model": model,
                "input_size": input_size,
                "output_size": output_size,
            }
        except Exception as e:
            logger.exception(f"Failed to load {model_name} model: {e}")
            continue

    if not models_dict:
        logger.error("Failed to load any models!")
        return {}

    return models_dict


def load_data_for_inference() -> Tuple[dict[str, tuple[TimeSeries, dict]], Any]:
    """Load and preprocess data for inference."""
    # Load and preprocess data
    raw_data = transform_data(PipelineConfig.DATA_FILE)
    transformed_data = {}
    for player, (player_values, static_cov, first_date) in raw_data.items():
        time_series = process_series(player_values, first_date)
        transformed_data[player] = (time_series, static_cov)

    # Load scaler
    scaler = joblib.load(TrainingConfig.CHECKPOINT_DIR + "/scaler.pkl")
    def scale_series(timeseries: TimeSeries, fitted_scaler) -> TimeSeries:
        return TimeSeries.from_times_and_values(
            timeseries.time_index,
            fitted_scaler.transform(timeseries.values()),
            columns=timeseries.columns,
        )

    transformed_data = {
        player: (scale_series(ts, scaler), cov)
        for player, (ts, cov) in transformed_data.items()
    }
    return transformed_data, scaler


def inference_models(scaled_data: dict[str, tuple[TimeSeries, dict]], models_dict: dict[str, dict]) -> dict[str, dict[str, TimeSeries]]:
    """Run inference using the loaded models and prepared data.
    Args:
        scaled_data (dict[str, tuple[TimeSeries, dict]]): Scaled time series data and static covariates keyed by player.
        models_dict (dict[str, dict]): Loaded models with their configurations.
    Returns:
        dict[str, dict[str, TimeSeries]]: Predictions from each model keyed by model name and player.
    """
    predictions = {}
    for model_name, model_info in models_dict.items():
        model = model_info["model"]
        input_size = model_info["input_size"]
        output_size = model_info["output_size"]
        extended_output_size = int(
            output_size * EvaluationConfig.PREDICTION_LENGTH_MULTIPLIER
        )
        logger.info(
            f"Running inference with {model_name} (input size: {input_size}, output size: {output_size}, extended output size: {extended_output_size})..."
        )

        model_predictions = {}
        for player, (series, static_covariates) in scaled_data.items():
            if len(series) < input_size:
                logger.warning(
                    f"Series for {player} is shorter than the required input size ({len(series)} < {input_size}). Skipping."
                )
                continue
            input_series = series[-input_size:]
            try:
                if model_name == "nhits": # Does not support static covariates
                    pred = model.predict(n=extended_output_size, series=input_series)
                else:
                    input_series = input_series.with_static_covariates(
                        pd.DataFrame(static_covariates, index=[0])
                    )
                    pred = model.predict(
                        n=extended_output_size,
                        series=input_series,
                    )
                model_predictions[player] = pred
            except Exception as e:
                logger.exception(
                    f"Failed to predict for series {player} with {model_name}: {e}"
                )
                continue
        predictions[model_name] = model_predictions
        logger.info(f"Completed inference with {model_name}.")
    return predictions

def save_data_to_json(predictions: dict[str, dict[str, TimeSeries]], raw_data: dict[str, tuple[TimeSeries, dict]], scaler, output_file: str):
    """Save predictions to a JSON file. Form of the JSON should be {PlayerName: {model_name: [all_values], ...}, ...}"""
    player_dict = {}
    for player, (series, static_covariates) in raw_data.items():
        # Inverse transform the original series
        inv_scaled_values = scaler.inverse_transform(series.values())
        first_date = series.start_time()
        if not isinstance(first_date, int):
            first_date = first_date.strftime("%d.%m.%Y")
        player_dict[player] = {"original": inv_scaled_values.flatten().tolist(), "first_date": first_date}
    for model_name, model_preds in predictions.items():
        for player, series in model_preds.items():
            if player not in player_dict:
                player_dict[player] = {}
            player_dict[player][model_name] = scaler.inverse_transform(series.values()).flatten().tolist()

    with open(output_file, "w") as f:
        json.dump(player_dict, f, indent=4)
    logger.info(f"Predictions saved to {output_file}")


async def run_inference_pipeline():
    """Sets up directories and runs the inference pipeline."""
    setup_directories()
    # await get_api_data()
    model_dict = load_models_for_inference()
    scaled_data, scaler = load_data_for_inference()
    if not model_dict or not scaled_data:
        logger.error("Inference pipeline aborted due to previous errors.")
        return
    predictions = inference_models(scaled_data, model_dict)
    output_file = os.path.join(PipelineConfig.DATA_DIR, "inference_results.json")
    save_data_to_json(predictions, scaled_data, scaler, output_file)
    logger.info("Inference pipeline completed.")

