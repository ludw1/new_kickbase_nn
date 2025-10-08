"""
Model evaluation and backtesting module.

Contains functions for comprehensive model evaluation, backtesting,
and performance analysis. Can be run standalone to evaluate a trained model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from darts.metrics import mae, rmse, mape, smape
from darts.models import NHiTSModel, NLinearModel, TiDEModel, LinearRegressionModel
from config import Config
from utils import setup_directories, setup_logging
from data_processing import load_and_preprocess_data
from models import Models
import torch

logger = logging.getLogger(__name__)


def evaluate_model(model, test_series, test_static_cov, input_size, output_size):
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


def run_backtests(model, train_series, val_series, test_series, train_static_cov, val_static_cov, test_static_cov, input_size, output_size):
    """Run comprehensive backtesting on the model.
    
    Args:
        model: Trained model
        train_series: Training time series
        val_series: Validation time series
        test_series: Test time series
        train_static_cov: Training static covariates
        val_static_cov: Validation static covariates
        test_static_cov: Test static covariates
        input_size: Input size used by the model
        output_size: Output size used by the model
    """
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
    forecast_horizon = min(output_size, min_length // 4)  # Use smaller horizon for backtesting
    backtest_start = len(test_series_with_cov[0]) // 2  # Start backtesting halfway through

    logger.info(f"Forecast horizon: {forecast_horizon}")
    logger.info(f"Backtest start point: {backtest_start}")

    # Run backtests on each test series
    for i, test_series in enumerate(test_series_with_cov): 
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


def load_model_from_checkpoint(model_path):
    """Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Loaded model instance
    """
    logger.info(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load based on model type
    if Config.MODEL_TYPE == "nhits":
        model = NHiTSModel.load(model_path)
    elif Config.MODEL_TYPE == "nlinear":
        model = NLinearModel.load(model_path)
    elif Config.MODEL_TYPE == "tide":
        model = TiDEModel.load(model_path, map_location=torch.device('cpu'))
    elif Config.MODEL_TYPE == "linear_regression":
        model = LinearRegressionModel.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {Config.MODEL_TYPE}")
    
    logger.info(f"Model loaded successfully: {Config.MODEL_TYPE}")
    return model


def main():
    """Main evaluation function - loads model from checkpoint and evaluates."""
    setup_directories()
    setup_logging()
    
    logger.info("="*60)
    logger.info("STANDALONE MODEL EVALUATION")
    logger.info("="*60)
    
    # Determine model checkpoint path (prefer BEST model if available)
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_TYPE}_{Config.MODEL_NAME}_BEST.pth")
    final_model_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_TYPE}_{Config.MODEL_NAME}_final.pth")
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
        logger.info("Using BEST model checkpoint")
    elif os.path.exists(final_model_path):
        model_path = final_model_path
        logger.info("Using FINAL model checkpoint")
    else:
        raise FileNotFoundError(
            f"No model checkpoint found. Expected:\n  {best_model_path}\n  or\n  {final_model_path}"
        )
    
    # Load the model
    model = load_model_from_checkpoint(model_path)
    
    # Get input/output sizes from model configuration
    logger.info(f"Setting up {Config.MODEL_TYPE} model configuration to get input/output sizes...")
    if Config.MODEL_TYPE == "nhits":
        _, _, _, input_size, output_size = Models.NHiTSModelConfig().setup_model()
    elif Config.MODEL_TYPE == "nlinear":
        _, _, _, input_size, output_size = Models.NLinearConfig().setup_model()
    elif Config.MODEL_TYPE == "tide":
        _, _, _, input_size, output_size = Models.TiDEConfig().setup_model()
    elif Config.MODEL_TYPE == "linear_regression":
        _, _, _, input_size, output_size = Models.LinearRegressionConfig().setup_model()
    else:
        raise ValueError(f"Unknown model type: {Config.MODEL_TYPE}")
    
    logger.info(f"Using input_size={input_size}, output_size={output_size}")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_series, val_series, test_series, train_static_cov, val_static_cov, test_static_cov = load_and_preprocess_data(
        input_size=input_size, output_size=output_size
    )
    
    logger.info(
        f"Loaded {len(train_series)} training series, {len(val_series)} validation series, and {len(test_series)} test series."
    )
    
    # Evaluate model on test set
    test_mae, test_rmse = evaluate_model(
        model=model,
        test_series=test_series,
        test_static_cov=test_static_cov,
        input_size=input_size,
        output_size=output_size
    )
    
    # Run comprehensive backtesting
    logger.info("\n" + "="*60)
    logger.info("STARTING MODEL BACKTESTING")
    logger.info("="*60)
    
    run_backtests(
        model=model,
        train_series=train_series,
        val_series=val_series,
        test_series=test_series,
        train_static_cov=train_static_cov,
        val_static_cov=val_static_cov,
        test_static_cov=test_static_cov,
        input_size=input_size,
        output_size=output_size
    )
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()