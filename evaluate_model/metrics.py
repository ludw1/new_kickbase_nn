"""
Metrics calculation and model evaluation functionality.

Contains functions for calculating various metrics and evaluating
model performance on test datasets.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from darts.metrics import mae, rmse, mape, smape
from typing import List, Dict, Any, Tuple
from config import TrainingConfig as Config
from .backtesting import run_backtests
logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    test_series: List,
    test_static_cov: List,
    input_size: int,
    output_size: int
) -> Tuple[float, float]:
    """Evaluate the trained model on the test set.
    
    Args:
        model: Trained model
        test_series: Test time series
        test_static_cov: Test static covariates
        input_size: Input size used by the model
        output_size: Output size used by the model
        
    Returns:
        tuple: (test_mae, test_rmse) - average test metrics
    """
    logger.info("="*50)
    logger.info("EVALUATING MODEL ON TEST SET")
    logger.info("="*50)
    
    # For proper evaluation, we need to split test series into input and target
    # The model uses the last input_size points to predict the next output_size points
    test_inputs = []
    test_targets = []

    for i, series in enumerate(test_series):
        # Use everything except the last output_size points as input
        # Use the last output_size points as target
        if len(series) >= input_size + output_size:
            input_series = series[: -output_size]
            target_series = series[-output_size :]

            # Add static covariates for models that support them
            if Config.MODEL_TYPE in ["tide", "nlinear", "linear_regression"]:
                input_series = input_series.with_static_covariates(test_static_cov[i])

            test_inputs.append(input_series)
            test_targets.append(target_series)

    logger.info(f"Evaluating on {len(test_inputs)} test series...")

    # Predict the next output_size points for each test input
    test_predictions = model.predict(n=output_size, series=test_inputs)
    
    # Plot example prediction
    plt.figure(figsize=(12, 6))
    test_inputs[0].plot(label="Test Input")
    test_targets[0].plot(label="Test Target")
    test_predictions[0].plot(label="Test Prediction")
    plt.legend()
    plt.title("Test Input, Target, and Prediction Example")
    plt.savefig(os.path.join(Config.LOG_DIR, "training_example.png"))
    plt.close()

    # Calculate test metrics by comparing predictions with actual targets
    test_mae_val = mae(test_targets, test_predictions)
    test_rmse_val = rmse(test_targets, test_predictions)

    logger.info("Test Set Metrics:")
    logger.info(f"  MAE: {np.mean(test_mae_val):.4f}")
    logger.info(f"  RMSE: {np.mean(test_rmse_val):.4f}")

    return np.mean(test_mae_val), np.mean(test_rmse_val)


def evaluate_single_model(
    model_name: str,
    model: Any,
    train_series: List,
    val_series: List,
    test_series: List,
    train_static_cov: List,
    val_static_cov: List,
    test_static_cov: List,
    input_size: int,
    output_size: int,
    extended_output_size: int
) -> Dict:
    """Evaluate a single model on the test set with both standard and extended predictions.
    
    Args:
        model_name: Name of the model
        model: Trained model instance
        train_series: Training time series
        val_series: Validation time series
        test_series: Test time series
        train_static_cov: Training static covariates
        val_static_cov: Validation static covariates
        test_static_cov: Test static covariates
        input_size: Input size used by the model
        output_size: Original trained output size
        extended_output_size: Extended output size for out-of-range testing
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name.upper()}...")
    
    # Prepare test inputs and targets
    test_inputs_standard = []
    test_targets_standard = []
    test_inputs_extended = []
    test_targets_extended = []
    
    for i, series in enumerate(test_series):
        # Standard evaluation (within trained range)
        if len(series) >= input_size + output_size:
            input_series = series[:-output_size]
            target_series = series[-output_size:]
            
            if model_name in ["tide", "nlinear", "linear_regression"]:
                input_series = input_series.with_static_covariates(test_static_cov[i])
            
            test_inputs_standard.append(input_series)
            test_targets_standard.append(target_series)
        
        # Extended evaluation (beyond trained range)
        if len(series) >= input_size + extended_output_size:
            input_series_ext = series[:-extended_output_size]
            target_series_ext = series[-extended_output_size:]
            
            if model_name in ["tide", "nlinear", "linear_regression"]:
                input_series_ext = input_series_ext.with_static_covariates(test_static_cov[i])
            
            test_inputs_extended.append(input_series_ext)
            test_targets_extended.append(target_series_ext)
    
    # Standard predictions
    predictions_standard = model.predict(n=output_size, series=test_inputs_standard)
    mae_val = mae(test_targets_standard, predictions_standard)
    rmse_val = rmse(test_targets_standard, predictions_standard)
    mape_val = mape(test_targets_standard, predictions_standard)
    smape_val = smape(test_targets_standard, predictions_standard)
    mae_standard = float(np.mean(mae_val)) if isinstance(mae_val, (list, np.ndarray)) else float(mae_val)
    rmse_standard = float(np.mean(rmse_val)) if isinstance(rmse_val, (list, np.ndarray)) else float(rmse_val)
    mape_standard = float(np.mean(mape_val)) if isinstance(mape_val, (list, np.ndarray)) else float(mape_val)
    smape_standard = float(np.mean(smape_val)) if isinstance(smape_val, (list, np.ndarray)) else float(smape_val)
    
    # Extended predictions
    predictions_extended = model.predict(n=extended_output_size, series=test_inputs_extended)
    mae_val = mae(test_targets_extended, predictions_extended)
    rmse_val = rmse(test_targets_extended, predictions_extended)
    mape_val = mape(test_targets_extended, predictions_extended)
    smape_val = smape(test_targets_extended, predictions_extended)
    mae_extended = float(np.mean(mae_val)) if isinstance(mae_val, (list, np.ndarray)) else float(mae_val)
    rmse_extended = float(np.mean(rmse_val)) if isinstance(rmse_val, (list, np.ndarray)) else float(rmse_val)
    mape_extended = float(np.mean(mape_val)) if isinstance(mape_val, (list, np.ndarray)) else float(mape_val)
    smape_extended = float(np.mean(smape_val)) if isinstance(smape_val, (list, np.ndarray)) else float(smape_val)
    
    # Run backtests
    backtest_results = run_backtests(
        model=model,
        model_name=model_name,
        test_series_list=test_series,
        test_static_cov=test_static_cov,
        output_size=output_size
    )
    
    return {
        'mae_standard': mae_standard,
        'rmse_standard': rmse_standard,
        'mape_standard': mape_standard,
        'smape_standard': smape_standard,
        'mae_extended': mae_extended,
        'rmse_extended': rmse_extended,
        'mape_extended': mape_extended,
        'smape_extended': smape_extended,
        'predictions_standard': predictions_standard[0] if predictions_standard else None,
        'predictions_extended': predictions_extended[0] if predictions_extended else None,
        'test_input': test_inputs_standard[0] if test_inputs_standard else None,
        'test_target_standard': test_targets_standard[0] if test_targets_standard else None,
        'test_target_extended': test_targets_extended[0] if test_targets_extended else None,
        'full_series': test_series[0] if test_series else None,
        **backtest_results
    }
