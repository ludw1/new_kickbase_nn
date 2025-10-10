"""
Main training pipeline module.

Orchestrates the entire training, evaluation, and backtesting process.
"""

import os
import logging
import matplotlib.pyplot as plt

# Import from new modular structure
from config import TrainingConfig as Config
from train_model.models import Models
from train_model.utils import setup_directories
from train_model.data_processing import load_and_preprocess_data

logger = logging.getLogger(__name__)


def train_model():
    """Main function to load data and preprocess."""
    setup_directories()

    # Select model configuration based on MODEL_TYPE to get input/output sizes
    logger.info(f"Setting up {Config.MODEL_TYPE} model configuration...")
    if Config.MODEL_TYPE == "nhits":
        model, loss_recorder, model_tracker, input_size, output_size = (
            Models.NHiTSModelConfig().setup_model()
        )
        logger.info("Using NHiTS model")
    elif Config.MODEL_TYPE == "nlinear":
        model, loss_recorder, model_tracker, input_size, output_size = (
            Models.NLinearConfig().setup_model()
        )
        logger.info("Using NLinear model")
    elif Config.MODEL_TYPE == "tide":
        model, loss_recorder, model_tracker, input_size, output_size = (
            Models.TiDEConfig().setup_model()
        )
        logger.info("Using TiDE model")
    elif Config.MODEL_TYPE == "linear_regression":
        model, loss_recorder, model_tracker, input_size, output_size = (
            Models.LinearRegressionConfig().setup_model()
        )
        logger.info("Using Linear Regression model")
    else:
        raise ValueError(f"Unknown model type: {Config.MODEL_TYPE}")

    logger.info(f"Model setup complete: {model}")
    logger.info(f"Using input_size={input_size}, output_size={output_size}")

    logger.info("Loading and preprocessing data...")
    (
        train_series,
        val_series,
        test_series,
        train_static_cov,
        val_static_cov,
        test_static_cov,
    ) = load_and_preprocess_data(input_size=input_size, output_size=output_size)

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
            logger.info(
                "Linear Regression training complete (no epochs - direct fitting)."
            )
        else:
            model.fit(
                train_series_with_cov, val_series=val_series_with_cov, verbose=True
            )
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
        logger.info(
            "Linear Regression doesn't use epoch-based training, skipping loss curves."
        )

    # Save the trained model
    final_model_path = os.path.join(
        Config.CHECKPOINT_DIR, f"{Config.MODEL_TYPE}_{Config.MODEL_NAME}_final.pt"
    )
    
    # Save state dict for PyTorch models, full model for linear regression
    if Config.MODEL_TYPE in ["nhits", "nlinear", "tide"]:
        # Save only the state dict for PyTorch Lightning models
        import torch
        torch.save(model.model.state_dict(), final_model_path)
        logger.info(f"Final model state dict saved to {final_model_path}")
        
        # For TiDE, also save the encoders/scalers separately
        if Config.MODEL_TYPE == "tide" and hasattr(model, 'encoders') and model.encoders is not None:
            encoders_path = final_model_path.replace('.pt', '_encoders.pt')
            torch.save(model.encoders, encoders_path)
            logger.info(f"Final model encoders saved to {encoders_path}")
    else:
        # Linear regression uses the traditional save method
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

    # Track final model performance and save best model
    if Config.MODEL_TYPE != "linear_regression":
        # Log best model information
        best_info = model_tracker.get_best_model_info()
        if best_info:
            logger.info("\n" + "=" * 60)
            logger.info("BEST MODEL PERFORMANCE:")
            logger.info(f"   Best epoch: {best_info['epoch']}")
            logger.info(f"   Best validation loss: {best_info['val_loss']:.4f}")
            logger.info(f"   Timestamp: {best_info['timestamp']}")
            logger.info("=" * 60)

            # For PyTorch Lightning models, the best model is saved during training
            # For other models, copy the final model as the best
            import shutil

            best_model_path = os.path.join(
                Config.CHECKPOINT_DIR,
                f"{Config.MODEL_TYPE}_{Config.MODEL_NAME}_BEST.pt",
            )
            if model_tracker.best_model_path and os.path.exists(
                model_tracker.best_model_path
            ):
                # Copy best model to clearly named file
                shutil.copy2(model_tracker.best_model_path, best_model_path)
                logger.info(f"Best model saved to: {best_model_path}")
            else:
                # If no specific best model was saved during training, use the final model
                logger.info(f"Using final model as best model: {final_model_path}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
