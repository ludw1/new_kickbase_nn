"""
Model evaluation and backtesting module.

Contains functions for comprehensive model evaluation, backtesting,
and performance analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from darts.metrics import mae, rmse, mape, smape
from config import Config

logger = logging.getLogger(__name__)


def run_backtests(model, train_series, val_series, test_series, train_static_cov, val_static_cov, test_static_cov):
    """Run comprehensive backtesting on the model."""
    logger.info("="*50)
    logger.info("RUNNING COMPREHENSIVE BACKTESTS")
    logger.info("="*50)

    # Combine train and validation for backtesting
    if Config.MODEL_TYPE in ["tft", "tide", "nlinear", "linear_regression"]:
        # Add static covariates to combined training data
        train_combined = []
        train_static_combined = []

        for i, series in enumerate(train_series):
            series_with_cov = series.with_static_covariates(train_static_cov[i])
            train_combined.append(series_with_cov)
            train_static_combined.append(train_static_cov[i])

        for i, series in enumerate(val_series):
            series_with_cov = series.with_static_covariates(val_static_cov[i])
            train_combined.append(series_with_cov)
            train_static_combined.append(val_static_cov[i])
    else:
        train_combined = train_series + val_series

    # Prepare test series with static covariates
    test_series_with_cov = []
    if Config.MODEL_TYPE in ["tft", "tide", "nlinear", "linear_regression"]:
        for i, series in enumerate(test_series):
            series_with_cov = series.with_static_covariates(test_static_cov[i])
            test_series_with_cov.append(series_with_cov)
    else:
        test_series_with_cov = test_series

    logger.info(f"Backtesting on {len(test_series_with_cov)} test series")

    # Metrics storage
    all_mae = []
    all_rmse = []
    all_mape = []
    all_smape = []

    # Test series length for backtesting
    min_length = min(len(series) for series in test_series_with_cov)
    logger.info(f"Minimum test series length: {min_length}")

    # Set up backtesting parameters
    forecast_horizon = min(Config.OUTPUT_SIZE, min_length // 4)  # Use smaller horizon for backtesting
    backtest_start = len(test_series_with_cov[0]) // 2  # Start backtesting halfway through

    logger.info(f"Forecast horizon: {forecast_horizon}")
    logger.info(f"Backtest start point: {backtest_start}")

    # Run backtests on each test series
    for i, test_series in enumerate(test_series_with_cov[:5]):  # Test on first 5 series for speed
        logger.info(f"\nBacktesting series {i+1}/5...")

        try:
            # Historical forecast backtest
            if Config.MODEL_TYPE in ["tft", "tide", "nlinear", "linear_regression"]:
                historical_forecasts = model.historical_forecasts(
                    test_series,
                    start=backtest_start,
                    forecast_horizon=forecast_horizon,
                    stride=forecast_horizon,
                    retrain=False,
                    verbose=False
                )
            else:
                historical_forecasts = model.historical_forecasts(
                    test_series,
                    start=backtest_start,
                    forecast_horizon=forecast_horizon,
                    stride=forecast_horizon,
                    retrain=False,
                    verbose=False
                )

            if len(historical_forecasts) > 0:
                # Calculate metrics for this series
                actual = test_series.slice(historical_forecasts.start_time(), historical_forecasts.end_time())

                mae_val = mae(actual, historical_forecasts)
                rmse_val = rmse(actual, historical_forecasts)
                mape_val = mape(actual, historical_forecasts)
                smape_val = smape(actual, historical_forecasts)

                all_mae.append(mae_val)
                all_rmse.append(rmse_val)
                all_mape.append(mape_val)
                all_smape.append(smape_val)

                logger.info(f"  Series {i+1} - MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, MAPE: {mape_val:.2f}%, SMAPE: {smape_val:.2f}%")

                # Plot backtest results for first series
                if i == 0:
                    plt.figure(figsize=(12, 6))
                    test_series.plot(label="Actual")
                    historical_forecasts.plot(label="Historical Forecasts")
                    plt.title(f"Backtest Results - {Config.MODEL_TYPE.upper()} Model")
                    plt.legend()
                    plt.savefig(os.path.join(Config.LOG_DIR, f"backtest_{Config.MODEL_TYPE}.png"))
                    plt.close()
            else:
                logger.warning(f"No historical forecasts generated for series {i+1}")

        except Exception as e:
            logger.error(f"Error backtesting series {i+1}: {e}")
            continue

    # Calculate and log average metrics
    if all_mae:
        logger.info("\n" + "="*50)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("="*50)
        logger.info(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
        logger.info(f"Average RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
        logger.info(f"Average MAPE: {np.mean(all_mape):.2f}% ± {np.std(all_mape):.2f}%")
        logger.info(f"Average SMAPE: {np.mean(all_smape):.2f}% ± {np.std(all_smape):.2f}%")

        # Save results to file
        results_df = pd.DataFrame({
            'Series': [f"Test_{i+1}" for i in range(len(all_mae))],
            'MAE': all_mae,
            'RMSE': all_rmse,
            'MAPE': all_mape,
            'SMAPE': all_smape
        })
        results_df.to_csv(os.path.join(Config.LOG_DIR, f"backtest_results_{Config.MODEL_TYPE}.csv"), index=False)
        logger.info(f"Results saved to {Config.LOG_DIR}/backtest_results_{Config.MODEL_TYPE}.csv")
    else:
        logger.warning("No successful backtests completed")