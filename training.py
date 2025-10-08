"""
Main training pipeline module.

Orchestrates the entire training, evaluation, and backtesting process.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from darts.metrics import mae, rmse

# Import from new modular structure
from config import Config
from models import Models
from utils import setup_directories, setup_logging
from data_processing import load_and_preprocess_data
from evaluation import run_backtests

logger = logging.getLogger(__name__)


def main():
    """Main function to load data and preprocess."""
    setup_directories()
    setup_logging()
    
    # Select model configuration based on MODEL_TYPE to get input/output sizes
    logger.info(f"Setting up {Config.MODEL_TYPE} model configuration...")
    if Config.MODEL_TYPE == "nhits":
        model, loss_recorder, model_tracker, input_size, output_size = Models.NHiTSModelConfig().setup_model()
        logger.info("Using NHiTS model")
    elif Config.MODEL_TYPE == "nlinear":
        model, loss_recorder, model_tracker, input_size, output_size = Models.NLinearConfig().setup_model()
        logger.info("Using NLinear model")
    elif Config.MODEL_TYPE == "tide":
        model, loss_recorder, model_tracker, input_size, output_size = Models.TiDEConfig().setup_model()
        logger.info("Using TiDE model")
    elif Config.MODEL_TYPE == "linear_regression":
        model, loss_recorder, model_tracker, input_size, output_size = Models.LinearRegressionConfig().setup_model()
        logger.info("Using Linear Regression model")
    else:
        raise ValueError(f"Unknown model type: {Config.MODEL_TYPE}")

    logger.info(f"Model setup complete: {model}")
    logger.info(f"Using input_size={input_size}, output_size={output_size}")
    
    logger.info("Loading and preprocessing data...")
    train_series, val_series, test_series, train_static_cov, val_static_cov, test_static_cov = load_and_preprocess_data(
        input_size=input_size, output_size=output_size
    )

    # Visualize example series from each set
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    train_series[0].plot(ax=axes[0], label="Example Training Player")
    axes[0].set_title("Training Set: Complete Time Series (Actual Values)")
    axes[0].legend()

    val_series[0].plot(ax=axes[1], label="Example Validation Player")
    axes[1].set_title("Validation Set: Complete Time Series (Actual Values)")
    axes[1].legend()

    test_series[0].plot(ax=axes[2], label="Example Test Player")
    axes[2].set_title("Test Set: Complete Time Series (Actual Values)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(Config.LOG_DIR, "example_series_split.png"))
    plt.close()

    logger.info(
        f"Loaded {len(train_series)} training series, {len(val_series)} validation series, and {len(test_series)} test series."
    )
    logger.info("Training pipeline setup complete.")

    # Add static covariates to time series for models that support them
    if Config.MODEL_TYPE in ["tide", "nlinear", "linear_regression"]:
        # Add static covariates to time series
        train_series_with_cov = []
        val_series_with_cov = []

        for i, series in enumerate(train_series):
            series_with_cov = series.with_static_covariates(train_static_cov[i])
            train_series_with_cov.append(series_with_cov)

        for i, series in enumerate(val_series):
            series_with_cov = series.with_static_covariates(val_static_cov[i])
            val_series_with_cov.append(series_with_cov)

        logger.info("Static covariates added to time series")

        if Config.MODEL_TYPE == "linear_regression":
            # Linear regression doesn't have verbose parameter or epochs
            model.fit(train_series_with_cov, val_series=val_series_with_cov)
            logger.info("Linear Regression training complete (no epochs - direct fitting).")
        else:
            model.fit(train_series_with_cov, val_series=val_series_with_cov, verbose=True)
            logger.info("Model training complete.")
    else:
        model.fit(train_series, val_series=val_series, verbose=True)
        logger.info("Model training complete.")

    # Plot training and validation loss only for models that use PyTorch Lightning
    if Config.MODEL_TYPE != "linear_regression":
        plt.plot(loss_recorder.train_losses, label="Train Loss")
        plt.plot(loss_recorder.val_losses, label="Validation Loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(Config.LOG_DIR, "loss_curves.png"))
        plt.close()
    else:
        logger.info("Linear Regression doesn't use epoch-based training, skipping loss curves.")

    # Evaluate on test set
    logger.info("Evaluating on test set...")

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
    test_inputs[0].plot(label="Test Input")
    test_targets[0].plot(label="Test Target")
    test_predictions[0].plot(label="Test Prediction")
    plt.legend()
    plt.title("Test Input, Target, and Prediction Example")
    plt.show()

    # Calculate test metrics by comparing predictions with actual targets
    test_mae_val = mae(test_targets, test_predictions)
    test_rmse = rmse(test_targets, test_predictions)

    logger.info("Test Set Metrics:")
    logger.info(f"  MAE: {np.mean(test_mae_val)}")
    logger.info(f"  RMSE: {np.mean(test_rmse)}")

    # Save the trained model
    final_model_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_TYPE}_{Config.MODEL_NAME}_final.pth")
    model.save(final_model_path)
    logger.info(
        f"Final model saved to {final_model_path}"
    )

    # Track final model performance and save best model
    if hasattr(model_tracker, 'update_performance'):
        # For linear regression, we don't have validation loss during training
        if Config.MODEL_TYPE == "linear_regression":
            # Calculate a simple validation metric
            if Config.MODEL_TYPE in ["tide", "nlinear", "linear_regression"]:
                train_series_with_cov = []
                val_series_with_cov = []

                for i, series in enumerate(train_series):
                    series_with_cov = series.with_static_covariates(train_static_cov[i])
                    train_series_with_cov.append(series_with_cov)

                for i, series in enumerate(val_series):
                    series_with_cov = series.with_static_covariates(val_static_cov[i])
                    val_series_with_cov.append(series_with_cov)

                # Fit and evaluate on validation set
                model.fit(train_series_with_cov)
                val_predictions = model.predict(n=output_size, series=val_series_with_cov)
                val_mae = mae(val_series_with_cov, val_predictions)
                model_tracker.update_performance(val_mae, 0, final_model_path)

        # Log best model information
        best_info = model_tracker.get_best_model_info()
        if best_info:
            logger.info("\n BEST MODEL PERFORMANCE:")
            logger.info(f"   Best epoch: {best_info['epoch']}")
            logger.info(f"   Best validation loss: {best_info['val_loss']:.4f}")
            logger.info(f"   Timestamp: {best_info['timestamp']}")

            # Save best model with a clear name
            best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_TYPE}_{Config.MODEL_NAME}_BEST.pth")
            if model_tracker.best_model_path and os.path.exists(model_tracker.best_model_path):
                # Copy best model to clearly named file
                import shutil
                shutil.copy2(model_tracker.best_model_path, best_model_path)
                logger.info(f" Best model saved to: {best_model_path}")

    logger.info("Training complete.")

    # Run comprehensive backtesting
    logger.info("\n" + "="*60)
    logger.info("STARTING MODEL BACKTESTING")
    logger.info("="*60)

    run_backtests(model, train_series, val_series, test_series, train_static_cov, val_static_cov, test_static_cov, input_size, output_size)

    logger.info("\n" + "="*60)
    logger.info("FULL PIPELINE COMPLETE - TRAINING + BACKTESTING")
    logger.info("="*60)


if __name__ == "__main__":
    main()