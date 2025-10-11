"""
Backtesting functionality for model evaluation.

Contains functions for running historical forecasts and calculating
backtest metrics on time series models.
"""

import logging
import numpy as np
from darts.metrics import mae, rmse, mape, smape
from typing import List, Dict, Any
from darts import TimeSeries

logger = logging.getLogger(__name__)


def run_backtests(
    model,
    model_name: str,
    test_series_list: List[TimeSeries],
    test_static_cov: List,
    output_size: int,
) -> Dict[str, Any]:
    """Run comprehensive backtesting on the model.

    Args:
        model: Trained model
        model_name: Name of the model
        test_series: Test time series
        train_static_cov: Training static covariates
        output_size: Output size used by the model

    Returns:
        Dictionary with backtest metrics
    """
    logger.info(f"Running backtests for {model_name.upper()}...")

    # Prepare test series with static covariates
    test_series_with_cov = []
    if model_name in ["tft", "tide", "nlinear", "linear_regression"]:
        for i, series in enumerate(test_series_list):
            series_with_cov = series.with_static_covariates(test_static_cov[i])
            test_series_with_cov.append(series_with_cov)
    else:
        test_series_with_cov = test_series_list

    all_mae = []
    all_rmse = []
    all_mape = []
    all_smape = []

    # Store first series data for plotting
    first_series = None
    first_forecasts = None

    # Test series length for backtesting
    min_length = min(len(series) for series in test_series_with_cov)

    # Set up backtesting parameters
    forecast_horizon = min(
        output_size, min_length // 4
    )  # Use smaller horizon for backtesting
    backtest_start = (
        len(test_series_with_cov[0]) // 4
    )  # Start backtesting at 25% through (extended window)

    # Run backtests on each test series
    for i, test_series in enumerate(test_series_with_cov):
        try:
            historical_forecasts = model.historical_forecasts(
                test_series,
                start=backtest_start,
                forecast_horizon=forecast_horizon,
                stride=forecast_horizon,
                retrain=False,
                verbose=False,
            )

            if len(historical_forecasts) > 0:
                # Store first series for plotting
                if i == 0:
                    first_series = test_series
                    first_forecasts = historical_forecasts

                actual = test_series.slice(
                    historical_forecasts.start_time(), historical_forecasts.end_time()
                )

                mae_val = mae(actual, historical_forecasts)
                rmse_val = rmse(actual, historical_forecasts)
                mape_val = mape(actual, historical_forecasts)
                smape_val = smape(actual, historical_forecasts)

                all_mae.append(mae_val)
                all_rmse.append(rmse_val)
                all_mape.append(mape_val)
                all_smape.append(smape_val)

        except Exception as e:
            logger.warning(f"Error backtesting series {i + 1}: {e}")
            continue

    # Calculate and return average metrics
    if all_mae:
        return {
            "backtest_mae": float(np.mean(all_mae)),
            "backtest_rmse": float(np.mean(all_rmse)),
            "backtest_mape": float(np.mean(all_mape)),
            "backtest_smape": float(np.mean(all_smape)),
            "backtest_series": first_series,
            "backtest_forecasts": first_forecasts,
        }
    else:
        logger.warning(f"No successful backtests completed for {model_name}")
        return {
            "backtest_mae": float("nan"),
            "backtest_rmse": float("nan"),
            "backtest_mape": float("nan"),
            "backtest_smape": float("nan"),
            "backtest_series": None,
            "backtest_forecasts": None,
        }
