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
import gzip
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Any, Dict, List

logger = logging.getLogger(__name__)


def julian_to_date(julian_date: int) -> str:
    """Convert a Julian date to a standard date format (DD.MM.YYYY)."""
    reference_date = datetime(1970, 1, 1)
    converted_date = reference_date + timedelta(days=julian_date)
    return converted_date.strftime("%d.%m.%Y")

def create_date_range(start_date: str, length: int) -> list[str]:
    """Create a list of date strings starting from start_date for a given length."""
    start_dt = datetime.strptime(start_date, "%d.%m.%Y")
    return [
        (start_dt + timedelta(days=i)).strftime("%m.%d.%Y") for i in range(length)
    ]

def process_series(player_values, first_date):
    values = np.array(player_values)

    # Create a datetime index starting from a reference date
    # Assuming daily data
    temp = create_date_range(julian_to_date(first_date), len(values))
    time_index = pd.DatetimeIndex(data=temp, freq="D")
    series = TimeSeries.from_times_and_values(times=time_index, values=values)

    return series


def extract_performance_data(performance_data: dict) -> List[Dict]:
    """Extract and optimize performance data from API response.

    Args:
        performance_data: Performance data from API

    Returns:
        List of performance entries with date, points, and minutes played
    """
    if not performance_data or not performance_data.get('it'):
        return []

    performances = []
    for season in performance_data['it']:
        for match in season.get('ph', []):
            # Extract only essential fields
            match_date = match.get('md', '')
            points = match.get('p', 0)
            minutes_played = match.get('mp', '0\'')
            status = match.get('st', 0)

            # Convert date format from ISO to YYYY-MM-DD
            if match_date:
                try:
                    # Parse ISO date and format
                    date_obj = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%Y-%m-%d')
                except Exception as e:
                    # Fallback to first 10 characters if parsing fails
                    logger.warning(f"Date parsing failed for {match_date}: {e}")
                    formatted_date = match_date[:10]
            else:
                formatted_date = ''

            # Clean minutes played (remove ' character)
            if isinstance(minutes_played, str):
                minutes_clean = minutes_played.replace("'", "")
                try:
                    minutes_int = int(minutes_clean)
                except Exception as e:
                    logger.warning(f"Minutes parsing failed for {minutes_clean}: {e}")
                    minutes_int = 0
            else:
                minutes_int = int(minutes_played) if minutes_played else 0

            performances.append({
                'date': formatted_date,
                'points': points if points is not None else 0,
                'minutes': minutes_int,
                'status': status if status is not None else 0
            })

    # Sort by date and keep only last 100 performances to optimize file size
    performances.sort(key=lambda x: x['date'], reverse=True)
    return performances[:100]


def extract_player_metadata(player_info: dict, api_info: dict) -> Dict:
    """Extract essential player metadata from API responses.

    Args:
        player_info: Basic player info from teams data
        api_info: Detailed player info from API

    Returns:
        Dictionary with essential player metadata
    """
    if not api_info:
        # Fallback to basic info if API data not available
        return {
            'full_name': player_info.get('name', ''),
            'team': player_info.get('team_name', ''),
            'position': player_info.get('pos', 0),
            'status': 0,
            'status_text': 'Unknown',
            'total_points': 0,
            'avg_points': 0,
            'goals': 0,
            'assists': 0
        }

    # Construct full name
    first_name = api_info.get('fn', '') or ''
    last_name = api_info.get('ln', '') or ''
    full_name = f"{first_name} {last_name}".strip()
    if not full_name:
        full_name = player_info.get('name', '')

    return {
        'full_name': full_name,
        'team': api_info.get('tn', player_info.get('team_name', '')),
        'position': api_info.get('pos', player_info.get('pos', 0)),
        'status': api_info.get('st', 0),
        'status_text': api_info.get('stxt', 'Fit' if api_info.get('st', 0) == 0 else 'Injured'),
        'total_points': api_info.get('tp', 0) or 0,
        'avg_points': api_info.get('ap', 0) or 0,
        'goals': api_info.get('g', 0) or 0,
        'assists': api_info.get('a', 0) or 0
    }


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


def load_data_for_inference() -> Tuple[dict[str, tuple[TimeSeries, dict]], Any, dict]:
    """Load and preprocess data for inference."""
    # Load raw player data for enhanced info extraction
    with open(PipelineConfig.DATA_FILE, "r") as f:
        raw_player_data = json.load(f)

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
    return transformed_data, scaler, raw_player_data


def inference_models(
    scaled_data: dict[str, tuple[TimeSeries, dict]], models_dict: dict[str, dict]
) -> dict[str, dict[str, TimeSeries]]:
    """Run inference using the loaded models and prepared data.
    Args:
        scaled_data (dict[str, tuple[TimeSeries, dict]]): Scaled time series data and static covariates keyed by player.
        models_dict (dict[str, dict]): Loaded models with their configurations.
    Returns:
        dict[str, dict[str, TimeSeries]]: Predictions from each model keyed by model name and player.
    """
    # Add AutoARIMA model to the models dict (fitted on-the-fly, no pre-training needed)
    from darts.models import AutoARIMA

    models_dict_with_arima = models_dict.copy()
    models_dict_with_arima["autoarima"] = {
        "model": AutoARIMA(),
        "input_size": 30,  # Minimum recommended historical points
        "output_size": 3,  # Default forecast horizon
    }
    logger.info("Added AutoARIMA model (will be fitted on-the-fly for each series)")

    predictions = {}
    for model_name, model_info in models_dict_with_arima.items():
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

            try:
                if model_name == "autoarima":
                    # AutoARIMA needs to be fitted on the entire historical series
                    # It doesn't use static covariates and works on the full series
                    model.fit(series)
                    pred = model.predict(n=extended_output_size)
                elif model_name == "nhits":
                    # NHiTS does not support static covariates
                    input_series = series[-input_size:]
                    pred = model.predict(n=extended_output_size, series=input_series)
                else:
                    # Other models support static covariates
                    input_series = series[-input_size:]
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


def load_and_update_inference_results(
    output_file: str,
    raw_data: dict[str, tuple[TimeSeries, dict]],
    scaler,
) -> dict:
    """Load existing inference results and update with new raw data using date-aware merging.

    This function:
    1. Loads the existing inference_results.json if it exists
    2. Removes old predictions (keeps only 'original' and 'first_date')
    3. Uses date-based logic to append only new data points
    4. Can recover from missed days by comparing dates

    Args:
        output_file: Path to the inference results JSON file
        raw_data: New raw time series data from API
        scaler: Fitted scaler for inverse transformation

    Returns:
        Dictionary with updated original data (predictions will be added later)
    """
    player_dict = {}

    # Load existing data if file exists
    if os.path.exists(output_file):
        logger.info(f"Loading existing inference results from {output_file}")
        with open(output_file, "r") as f:
            existing_data = json.load(f)

        # Keep core data and enhanced info, remove only predictions
        for player, data in existing_data.items():
            if "original" in data and "first_date" in data:
                player_dict[player] = {
                    "original": data["original"],
                    "first_date": data["first_date"],
                }

                # Preserve enhanced player information if it exists
                if "player_meta" in data:
                    player_dict[player]["player_meta"] = data["player_meta"]
                if "performance" in data:
                    player_dict[player]["performance"] = data["performance"]

                logger.debug(
                    f"Loaded existing data for {player}: {len(data['original'])} points"
                )
    else:
        logger.info(
            f"No existing inference results found at {output_file}, creating new file"
        )

    # Update with new raw data using date-aware logic
    for player, (series, static_covariates) in raw_data.items():
        inv_scaled_values = scaler.inverse_transform(series.values()).flatten().tolist()
        new_first_date = series.start_time()
        if not isinstance(new_first_date, int):
            new_first_date_str = new_first_date.strftime(
                "%d.%m.%Y"
            )  # For some reason the date format changes
        else:
            new_first_date_str = julian_to_date(new_first_date)
        logger.info(
            f"Processing new data for {player} starting from {new_first_date_str} with {len(inv_scaled_values)} points"
        )
        if player in player_dict:
            # Date-aware appending
            existing_values = player_dict[player]["original"]
            existing_first_date_str = player_dict[player]["first_date"]

            # Parse existing first date
            try:
                existing_first_date = datetime.strptime(
                    existing_first_date_str, "%d.%m.%Y"
                )
            except Exception as e:
                logger.warning(
                    f"Could not parse existing first_date for {player}: {existing_first_date_str} - {e}"
                )
                # Fall back to simple append
                player_dict[player]["original"].extend(inv_scaled_values)
                logger.info(
                    f"Appended {len(inv_scaled_values)} values for {player} (fallback mode)"
                )
                continue

            # Calculate the last date in existing data
            # last_date = first_date + len(existing_values) days
            existing_last_date = existing_first_date + timedelta(
                days=len(existing_values) - 1
            )

            # Parse new first date
            try:
                new_first_date_dt = datetime.strptime(new_first_date_str, "%d.%m.%Y")
            except Exception as e:
                logger.warning(
                    f"Could not parse new first_date for {player}: {new_first_date_str} - {e}"
                )
                continue

            # Calculate the last date in new data
            new_last_date = new_first_date_dt + timedelta(
                days=len(inv_scaled_values) - 1
            )

            logger.debug(
                f"{player}: Existing data ends on {existing_last_date.strftime('%d.%m.%Y')}, "
                f"new data spans {new_first_date_str} to {new_last_date.strftime('%d.%m.%Y')}"
            )

            # Check if new data extends beyond existing data
            if new_last_date > existing_last_date:
                # Determine the starting index in new data to append
                # If new_first_date <= existing_last_date, there's overlap
                if new_first_date_dt <= existing_last_date:
                    # Calculate overlap
                    days_from_new_start_to_existing_end = (
                        existing_last_date - new_first_date_dt
                    ).days
                    start_index = (
                        days_from_new_start_to_existing_end + 1
                    )  # Start from day after existing_last_date

                    if start_index < len(inv_scaled_values):
                        values_to_append = inv_scaled_values[start_index:]
                        player_dict[player]["original"].extend(values_to_append)
                        logger.info(
                            f"Appended {len(values_to_append)} new values for {player} "
                            f"(total: {len(player_dict[player]['original'])} points)"
                        )
                    else:
                        logger.info(
                            f"No new values to append for {player} - data already up to date"
                        )
                else:
                    # Gap detected - new data starts after existing data ends
                    gap_days = (new_first_date_dt - existing_last_date).days - 1
                    if gap_days > 0:
                        logger.warning(
                            f"Gap of {gap_days} days detected for {player} between "
                            f"{existing_last_date.strftime('%d.%m.%Y')} and {new_first_date_str}"
                        )

                    # Append all new values
                    player_dict[player]["original"].extend(inv_scaled_values)
                    logger.info(
                        f"Appended {len(inv_scaled_values)} new values for {player} "
                        f"(total: {len(player_dict[player]['original'])} points)"
                    )
            else:
                logger.info(
                    f"No new data for {player} - existing data is already up to date or newer"
                )
        else:
            # New player, add all data
            player_dict[player] = {
                "original": inv_scaled_values,
                "first_date": new_first_date_str,
            }
            logger.info(
                f"Added new player {player} with {len(inv_scaled_values)} values"
            )

    return player_dict


def save_data_to_json(
    predictions: dict[str, dict[str, TimeSeries]],
    raw_data: dict[str, tuple[TimeSeries, dict]],
    scaler,
    output_file: str,
):
    """Save predictions to a JSON file. Form of the JSON should be {PlayerName: {model_name: [all_values], ...}, ...}"""
    player_dict = {}
    for player, (series, static_covariates) in raw_data.items():
        # Inverse transform the original series
        inv_scaled_values = scaler.inverse_transform(series.values())
        first_date = series.start_time()
        if not isinstance(first_date, int):
            first_date = first_date.strftime("%d.%m.%Y")
        player_dict[player] = {
            "original": inv_scaled_values.flatten().tolist(),
            "first_date": first_date,
        }
    for model_name, model_preds in predictions.items():
        for player, series in model_preds.items():
            if player not in player_dict:
                player_dict[player] = {}
            player_dict[player][model_name] = (
                scaler.inverse_transform(series.values()).flatten().tolist()
            )

    with open(output_file, "w") as f:
        json.dump(player_dict, f, indent=4)
    logger.info(f"Predictions saved to {output_file}")


def save_updated_data_to_json(
    player_dict: dict,
    predictions: dict[str, dict[str, TimeSeries]],
    scaler,
    output_file: str,
    raw_player_data: dict = {},
):
    """Save updated player data with new predictions to JSON file.

    Args:
        player_dict: Dictionary with updated original data
        predictions: New predictions from models
        scaler: Fitted scaler for inverse transformation
        output_file: Path to save the JSON file
        raw_player_data: Raw player data with enhanced info from API
    """
    # Add predictions to the player dictionary
    for model_name, model_preds in predictions.items():
        for player, series in model_preds.items():
            if player not in player_dict:
                player_dict[player] = {}
            player_dict[player][model_name] = (
                scaler.inverse_transform(series.values()).flatten().tolist()
            )

    # Add enhanced player information if available
    if raw_player_data:
        for player_key, player_data in player_dict.items():
            # Extract player ID from the key (format: "PlayerName_ID")
            player_id = player_key.split("_")[-1] if "_" in player_key else player_key

            # Get corresponding data from raw player data
            raw_data_entry = None
            for raw_key, raw_value in raw_player_data.items():
                # Match by ID - the raw data key might also have "_ID" format
                raw_key_id = raw_key.split("_")[-1] if "_" in raw_key else raw_key
                if raw_key_id == player_id:
                    raw_data_entry = raw_value
                    break

            if raw_data_entry:
                # Extract and add player metadata
                player_info = raw_data_entry.get('player_info', {})
                api_info = raw_data_entry.get('info', {})
                performance_data = raw_data_entry.get('performance', {})

                metadata = extract_player_metadata(player_info, api_info)
                performance = extract_performance_data(performance_data)

                # Only add if we have meaningful data
                if metadata.get('full_name') or performance:
                    player_data['player_meta'] = metadata
                    if performance:
                        player_data['performance'] = performance

    compressed_path = output_file + ".gz"
    with gzip.open(compressed_path, "wt", encoding="utf-8") as f:
        json.dump(player_dict, f, indent=2, default=str)
    logger.info(f"Saved compressed data to {compressed_path}")
    # Fallback to uncompressed
    with open(output_file, "w") as f:
        json.dump(player_dict, f, indent=2, default=str)
    logger.info(f"Fallback: Saved uncompressed data to {output_file}")


async def run_inference_pipeline(append_mode: bool = True):
    """Sets up directories and runs the inference pipeline.

    Args:
        append_mode: If True, loads existing inference results and appends new data.
                    If False, overwrites existing results (legacy behavior).
    """
    setup_directories()
    await get_api_data()
    model_dict = load_models_for_inference()
    scaled_data, scaler, raw_player_data = load_data_for_inference()
    if not model_dict or not scaled_data:
        logger.error("Inference pipeline aborted due to previous errors.")
        return

    output_file = os.path.join(PipelineConfig.DATA_DIR, "inference_results.json")

    if append_mode:
        # Load existing data and append new values
        logger.info("Running in append mode - loading existing results and updating")
        player_dict = load_and_update_inference_results(
            output_file, scaled_data, scaler
        )

        # Run predictions
        predictions = inference_models(scaled_data, model_dict)

        # Save with updated data and enhanced player info
        save_updated_data_to_json(player_dict, predictions, scaler, output_file, raw_player_data)

    else:
        # Legacy mode - overwrite everything
        logger.info("Running in overwrite mode - creating fresh predictions")
        predictions = inference_models(scaled_data, model_dict)
        save_data_to_json(predictions, scaled_data, scaler, output_file)

    logger.info("Inference pipeline completed.")
