"""
Ensemble prediction functionality.

Contains functions for creating and evaluating ensemble predictions
from multiple models.
"""

import logging
import numpy as np
from darts import TimeSeries
from darts.metrics import mae, rmse, mape, smape
from darts.models import AutoARIMA
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def create_simple_ensemble_predictions(
    all_model_predictions: List[List], n_samples: int, output_size: int, time_index
) -> List:
    """Create simple average ensemble predictions.

    Args:
        all_model_predictions: List of prediction lists from different models
        n_samples: Number of samples to create predictions for
        output_size: Output size for predictions
        time_index: Time index for the predictions

    Returns:
        List of ensemble predictions as TimeSeries objects
    """
    ensemble_predictions = []

    for sample_idx in range(n_samples):
        # Average predictions from all models
        sample_preds = []
        for model_preds in all_model_predictions:
            if sample_idx < len(model_preds):
                sample_preds.append(model_preds[sample_idx].values().flatten())

        if len(sample_preds) > 0:
            avg_pred = np.mean(sample_preds, axis=0)
            ensemble_ts = TimeSeries.from_times_and_values(
                times=time_index, values=avg_pred
            )
            ensemble_predictions.append(ensemble_ts)

    return ensemble_predictions


def calculate_ensemble_metrics(
    ensemble_preds_standard: List,
    ensemble_preds_extended: List,
    test_targets_standard: List,
    test_targets_extended: List,
    test_inputs_standard: List,
    test_series: List,
) -> Dict[str, Any]:
    """Calculate metrics for ensemble predictions.

    Args:
        ensemble_preds_standard: Standard ensemble predictions
        ensemble_preds_extended: Extended ensemble predictions
        test_targets_standard: Standard test targets
        test_targets_extended: Extended test targets
        test_inputs_standard: Standard test inputs
        test_series: Full test series

    Returns:
        Dictionary with ensemble metrics
    """
    if not ensemble_preds_standard or not ensemble_preds_extended:
        return {}

    mae_val = mae(test_targets_standard, ensemble_preds_standard)
    rmse_val = rmse(test_targets_standard, ensemble_preds_standard)
    mape_val = mape(test_targets_standard, ensemble_preds_standard)
    smape_val = smape(test_targets_standard, ensemble_preds_standard)
    mae_standard = (
        float(np.mean(mae_val))
        if isinstance(mae_val, (list, np.ndarray))
        else float(mae_val)
    )
    rmse_standard = (
        float(np.mean(rmse_val))
        if isinstance(rmse_val, (list, np.ndarray))
        else float(rmse_val)
    )
    mape_standard = (
        float(np.mean(mape_val))
        if isinstance(mape_val, (list, np.ndarray))
        else float(mape_val)
    )
    smape_standard = (
        float(np.mean(smape_val))
        if isinstance(smape_val, (list, np.ndarray))
        else float(smape_val)
    )

    mae_val = mae(test_targets_extended, ensemble_preds_extended)
    rmse_val = rmse(test_targets_extended, ensemble_preds_extended)
    mape_val = mape(test_targets_extended, ensemble_preds_extended)
    smape_val = smape(test_targets_extended, ensemble_preds_extended)
    mae_extended = (
        float(np.mean(mae_val))
        if isinstance(mae_val, (list, np.ndarray))
        else float(mae_val)
    )
    rmse_extended = (
        float(np.mean(rmse_val))
        if isinstance(rmse_val, (list, np.ndarray))
        else float(rmse_val)
    )
    mape_extended = (
        float(np.mean(mape_val))
        if isinstance(mape_val, (list, np.ndarray))
        else float(mape_val)
    )
    smape_extended = (
        float(np.mean(smape_val))
        if isinstance(smape_val, (list, np.ndarray))
        else float(smape_val)
    )

    logger.info(f"Ensemble MAE (Standard): {mae_standard:.4f}")
    logger.info(f"Ensemble MAE (Extended): {mae_extended:.4f}")

    return {
        "mae_standard": mae_standard,
        "rmse_standard": rmse_standard,
        "mape_standard": mape_standard,
        "smape_standard": smape_standard,
        "mae_extended": mae_extended,
        "rmse_extended": rmse_extended,
        "mape_extended": mape_extended,
        "smape_extended": smape_extended,
        "predictions_standard": ensemble_preds_standard[0]
        if ensemble_preds_standard
        else None,
        "predictions_extended": ensemble_preds_extended[0]
        if ensemble_preds_extended
        else None,
        "test_input": test_inputs_standard[0] if test_inputs_standard else None,
        "test_target_standard": test_targets_standard[0]
        if test_targets_standard
        else None,
        "test_target_extended": test_targets_extended[0]
        if test_targets_extended
        else None,
        "full_series": test_series[0] if test_series else None,
        "backtest_mae": float("nan"),  # Skip backtest for ensemble
        "backtest_rmse": float("nan"),
        "backtest_mape": float("nan"),
        "backtest_smape": float("nan"),
        "backtest_series": None,
        "backtest_forecasts": None,
    }


def evaluate_arima_model(
    test_series: List,
    input_size: int,
    output_size: int,
    extended_output_size: int,
) -> Dict[str, Any]:
    """Evaluate ARIMA model on a single time series (the one used for plotting).
    
    Since ARIMA works on single time series, we use the first test series
    which is also used for plotting.
    
    Args:
        test_series: List of test time series
        input_size: Input size (history to use)
        output_size: Standard output size
        extended_output_size: Extended output size        
    Returns:
        Dictionary with ARIMA evaluation metrics
    """
    logger.info("Evaluating ARIMA model...")
    
    if not test_series:
        logger.error("No test series available for ARIMA")
        return {}
    
    series = test_series[0]
    
    # Prepare standard evaluation
    if len(series) < input_size + output_size:
        logger.error(f"Series too short for ARIMA evaluation: {len(series)}")
        return {}
    
    train_data_standard = series[:-output_size]
    target_standard = series[-output_size:]
    
    # Prepare extended evaluation
    if len(series) < input_size + extended_output_size:
        logger.warning(f"Series too short for extended ARIMA evaluation: {len(series)}")
        train_data_extended = None
        target_extended = None
    else:
        train_data_extended = series[:-extended_output_size]
        target_extended = series[-extended_output_size:]
    
    try:
        # Standard prediction
        logger.info("Training AutoARIMA for standard prediction...")
        arima_model_standard = AutoARIMA()
        arima_model_standard.fit(train_data_standard)
        pred_standard = arima_model_standard.predict(n=output_size)
        
        # Calculate standard metrics
        mae_val = mae(target_standard, pred_standard)
        rmse_val = rmse(target_standard, pred_standard)
        mape_val = mape(target_standard, pred_standard)
        smape_val = smape(target_standard, pred_standard)
        mae_standard = float(np.mean(mae_val)) if isinstance(mae_val, (list, np.ndarray)) else float(mae_val)
        rmse_standard = float(np.mean(rmse_val)) if isinstance(rmse_val, (list, np.ndarray)) else float(rmse_val)
        mape_standard = float(np.mean(mape_val)) if isinstance(mape_val, (list, np.ndarray)) else float(mape_val)
        smape_standard = float(np.mean(smape_val)) if isinstance(smape_val, (list, np.ndarray)) else float(smape_val)
        
        logger.info(f"ARIMA MAE (Standard): {mae_standard:.4f}")
        
        # Extended prediction
        if train_data_extended is not None:
            logger.info("Training AutoARIMA for extended prediction...")
            arima_model_extended = AutoARIMA()
            arima_model_extended.fit(train_data_extended)
            pred_extended = arima_model_extended.predict(n=extended_output_size)
            
            # Calculate extended metrics
            mae_val = mae(target_extended, pred_extended)
            rmse_val = rmse(target_extended, pred_extended)
            mape_val = mape(target_extended, pred_extended)
            smape_val = smape(target_extended, pred_extended)
            mae_extended = float(np.mean(mae_val)) if isinstance(mae_val, (list, np.ndarray)) else float(mae_val)
            rmse_extended = float(np.mean(rmse_val)) if isinstance(rmse_val, (list, np.ndarray)) else float(rmse_val)
            mape_extended = float(np.mean(mape_val)) if isinstance(mape_val, (list, np.ndarray)) else float(mape_val)
            smape_extended = float(np.mean(smape_val)) if isinstance(smape_val, (list, np.ndarray)) else float(smape_val)
            
            logger.info(f"ARIMA MAE (Extended): {mae_extended:.4f}")
        else:
            mae_extended = float("nan")
            rmse_extended = float("nan")
            mape_extended = float("nan")
            smape_extended = float("nan")
            pred_extended = None
        
        return {
            "mae_standard": mae_standard,
            "rmse_standard": rmse_standard,
            "mape_standard": mape_standard,
            "smape_standard": smape_standard,
            "mae_extended": mae_extended,
            "rmse_extended": rmse_extended,
            "mape_extended": mape_extended,
            "smape_extended": smape_extended,
            "predictions_standard": pred_standard,
            "predictions_extended": pred_extended,
            "test_input": train_data_standard,
            "test_target_standard": target_standard,
            "test_target_extended": target_extended,
            "full_series": series,
            "backtest_mae": float("nan"),  # Skip backtest for ARIMA
            "backtest_rmse": float("nan"),
            "backtest_mape": float("nan"),
            "backtest_smape": float("nan"),
            "backtest_series": None,
            "backtest_forecasts": None,
        }
        
    except Exception as e:
        logger.error(f"Failed to evaluate ARIMA: {e}")
        import traceback
        traceback.print_exc()
        return {}
