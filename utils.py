"""
Utility classes and functions for training pipeline.

Contains ModelTracker, LossRecorder, ContinuityLoss, and setup functions.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import logging
from datetime import datetime
from pytorch_lightning.callbacks import Callback
from config import Config

logger = logging.getLogger(__name__)


class ModelTracker:
    """Track and save the best models based on validation performance."""

    def __init__(self, model_type, log_dir="logs"):
        self.model_type = model_type
        self.log_dir = log_dir
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.tracking_file = os.path.join(log_dir, f"model_performance_{model_type}.csv")

        # Initialize tracking CSV
        self.performance_data = []

    def update_performance(self, val_loss, epoch, model_path=None):
        """Update model performance and save if best."""
        performance_entry = {
            'epoch': epoch,
            'val_loss': val_loss,
            'is_best': val_loss < self.best_val_loss,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.performance_data.append(performance_entry)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if model_path:
                self.best_model_path = model_path
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")

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

        best_entry = min(self.performance_data, key=lambda x: x['val_loss'])
        return best_entry


class LossRecorder(Callback):
    """Callback to record training and validation losses."""

    def __init__(self, model_tracker=None):
        self.train_losses = []
        self.val_losses = []
        self.model_tracker = model_tracker

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)

        # Update model tracker with validation performance
        if self.model_tracker:
            current_epoch = trainer.current_epoch
            model_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_TYPE}_{Config.MODEL_NAME}_epoch_{current_epoch}.pth")
            self.model_tracker.update_performance(val_loss, current_epoch, model_path)


class ContinuityLoss(nn.Module):
    """Custom loss to enforce continuity between input and output sequences for residual predictions."""

    def __init__(self, continuity_weight: float = 1.5):
        super(ContinuityLoss, self).__init__()
        self.continuity_weight = continuity_weight
        self.mse = nn.MSELoss()

    def forward(self, inputs, predictions, targets):
        # For residual predictions, the first prediction should be close to 0 (continuity)
        # because it represents the change from the last input value
        overall_loss = self.mse(predictions, targets)
        continuity_loss = self.mse(
            predictions[:, 0], torch.zeros_like(predictions[:, 0])
        )
        return overall_loss + self.continuity_weight * continuity_loss


def setup_directories():
    """Setup checkpoint and log directories."""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)


def setup_logging():
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(Config.LOG_DIR, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )