"""
Utility classes and functions for training pipeline.

Contains ModelTracker, LossRecorder, ContinuityLoss, and setup functions.
"""

import os
import torch
import pandas as pd
import logging
from datetime import datetime
from pytorch_lightning.callbacks import Callback
from config import TrainingConfig as Config

logger = logging.getLogger(__name__)


class ModelTracker:
    """Track and save the best models based on validation performance."""

    def __init__(self, model_type, log_dir="logs", checkpoint_dir="checkpoints"):
        self.model_type = model_type
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")
        self.best_model_path = None
        self.tracking_file = os.path.join(
            log_dir, f"model_performance_{model_type}.csv"
        )

        # Initialize tracking CSV
        self.performance_data = []

        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

    def update_performance(
        self, val_loss, epoch, model_path=None, pl_module=None, darts_model=None
    ):
        """Update model performance and save if best.

        Args:
            val_loss: Validation loss for this epoch
            epoch: Current epoch number
            model_path: Optional path where model should be saved
            pl_module: Optional PyTorch Lightning module (the actual trained model)
            darts_model: Optional Darts model wrapper (for accessing encoders/scalers)
        """
        performance_entry = {
            "epoch": epoch,
            "val_loss": val_loss,
            "is_best": val_loss < self.best_val_loss,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.performance_data.append(performance_entry)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

            # Generate best model path if not provided
            if not model_path:
                model_path = os.path.join(
                    self.checkpoint_dir, f"{self.model_type}_best_epoch_{epoch}.pth"
                )

            self.best_model_path = model_path

            # Actually save the model if provided
            if pl_module is not None:
                try:
                    # For PyTorch Lightning models (NHiTS, NLinear, TiDE), save state dict from pl_module
                    if self.model_type in ["nhits", "nlinear", "tide"]:
                        torch.save(pl_module.state_dict(), model_path)
                        logger.info(
                            f"New best model state dict saved to {model_path} with val_loss: {val_loss:.4f}"
                        )

                        # For TiDE, also save the encoders/scalers from the Darts model
                        if (
                            self.model_type == "tide"
                            and darts_model is not None
                            and hasattr(darts_model, "encoders")
                            and darts_model.encoders is not None
                        ):
                            encoders_path = model_path.replace(".pt", "_encoders.pt")
                            torch.save(darts_model.encoders, encoders_path)
                            logger.info(f"Best model encoders saved to {encoders_path}")
                    else:
                        logger.warning(
                            f"Unexpected model type {self.model_type} for pl_module, skipping save"
                        )
                except Exception as e:
                    logger.error(f"Failed to save best model: {e}")
            elif darts_model is not None and hasattr(darts_model, "save"):
                # Fallback for models like Linear Regression that use traditional save
                try:
                    darts_model.save(model_path)
                    logger.info(
                        f"New best model saved to {model_path} with val_loss: {val_loss:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Failed to save best model: {e}")
            else:
                logger.info(
                    f"New best val_loss: {val_loss:.4f} at epoch {epoch} (model not provided for saving)"
                )

        # Save performance tracking
        self._save_performance()

    def _save_performance(self):
        """Save performance data to CSV."""
        df = pd.DataFrame(self.performance_data)
        df.to_csv(self.tracking_file, index=False)

    def get_best_model_info(self):
        """Get information about the best model."""
        if not self.performance_data:
            return None

        best_entry = min(self.performance_data, key=lambda x: x["val_loss"])
        return best_entry


class LossRecorder(Callback):
    """Callback to record training and validation losses."""

    def __init__(self, model_tracker=None, darts_model=None):
        self.train_losses = []
        self.val_losses = []
        self.model_tracker = model_tracker
        self.darts_model = darts_model  # Store reference to the Darts model

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)

        # Update model tracker with validation performance
        if self.model_tracker:
            current_epoch = trainer.current_epoch
            model_path = os.path.join(
                Config.CHECKPOINT_DIR,
                f"{Config.MODEL_TYPE}_{Config.MODEL_NAME}_epoch_{current_epoch}.pt",
            )

            # Pass the PyTorch Lightning module directly for saving
            # pl_module is the actual trained PLForecastingModule
            self.model_tracker.update_performance(
                val_loss,
                current_epoch,
                model_path,
                pl_module=pl_module,
                darts_model=self.darts_model,
            )

    def set_darts_model(self, darts_model):
        """Set the Darts model reference after model initialization."""
        self.darts_model = darts_model


def setup_directories():
    """Setup checkpoint and log directories."""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
