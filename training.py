from nhits import NHiTS
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transform_data import transform_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import logging


# Configuration
class Config:
    INPUT_SIZE = 360
    OUTPUT_SIZE = 3
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    MIN_DELTA = 1e-4
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    MODEL_NAME = "nhits_kickbase"
    SEED = 42

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
        continuity_loss = self.mse(predictions[:, 0], torch.zeros_like(predictions[:, 0]))
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
    return logging.getLogger(__name__)


def save_checkpoint(
    model, optimizer, scheduler, epoch, train_loss, val_loss, is_best=False
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        # "config": Config.__dict__,
    }

    checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR, f"{Config.MODEL_NAME}_epoch_{epoch}.pth"
    )
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_NAME}_best.pth")
        torch.save(checkpoint, best_path)
        logging.info(f"New best model saved with val_loss: {val_loss:.6f}")


def load_checkpoint(
    model: NHiTS,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    checkpoint_path: str,
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["train_loss"], checkpoint["val_loss"]



def load_and_preprocess_data(data_file: str = "all_player_data.json"):
    """Load and preprocess data, creating targets as residuals from the last input step."""
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    raw_data = transform_data(data_file)
    train_val, test = train_test_split(
        raw_data, test_size=0.2, random_state=Config.SEED
    )

    train, val = train_test_split(train_val, test_size=0.25, random_state=Config.SEED)

    # Normalize data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train) # type: ignore
    val_data = scaler.transform(val)
    test_data = scaler.transform(test)

    # --- CREATE RESIDUAL TARGETS ---
    # Get the last value of the input window for each sample
    last_val_train = train_data[:, Config.INPUT_SIZE - 1]
    last_val_val = val_data[:, Config.INPUT_SIZE - 1]
    last_val_test = test_data[:, Config.INPUT_SIZE - 1]

    # The new target is the future value MINUS the last input value
    # np.newaxis is used for correct broadcasting [n_samples, 1]
    y_train_residuals = train_data[:, -Config.OUTPUT_SIZE:] - last_val_train[:, np.newaxis]
    y_val_residuals = val_data[:, -Config.OUTPUT_SIZE:] - last_val_val[:, np.newaxis]
    y_test_residuals = test_data[:, -Config.OUTPUT_SIZE:] - last_val_test[:, np.newaxis]

    # --- CONVERT TO TENSORS ---
    X_train = torch.tensor(train_data[:, :Config.INPUT_SIZE], dtype=torch.float32)
    y_train = torch.tensor(y_train_residuals, dtype=torch.float32) # Use residual targets

    X_val = torch.tensor(val_data[:, :Config.INPUT_SIZE], dtype=torch.float32)
    y_val = torch.tensor(y_val_residuals, dtype=torch.float32) # Use residual targets

    X_test = torch.tensor(test_data[:, :Config.INPUT_SIZE], dtype=torch.float32)
    y_test_residuals = torch.tensor(y_test_residuals, dtype=torch.float32) # Use residual targets

    # Note: We keep the original y_test for final evaluation
    y_test_actuals = torch.tensor(test_data[:, -Config.OUTPUT_SIZE:], dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    # The test dataloader will now provide (X_test, y_test_residuals, y_test_actuals)
    test_dataset = TensorDataset(X_test, y_test_residuals, y_test_actuals)

    train_dataloader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader, X_test, scaler


def setup_model():
    """Setup the model, criterion, and optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NHiTS(
        input_size=Config.INPUT_SIZE,
        output_size=Config.OUTPUT_SIZE,
        num_blocks=2,
        num_stacks=3,
        stack_pooling_kernel_sizes=[10, 4, 2],
        dropout_rate=0.1,
        use_layer_norm=True,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    return model, criterion, optimizer, scheduler, device


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    logger,
) -> tuple[NHiTS, list, list, float]:
    """Train the model with early stopping and checkpointing."""
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(Config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
        )

        # Early stopping check
        if val_loss < best_val_loss - Config.MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            is_best = True
            logger.info("Val loss improved")
        else:
            patience_counter += 1
            is_best = False
            logger.info(
                f"Val loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, train_loss, val_loss, is_best
        )

        # Early stopping
        if patience_counter >= Config.PATIENCE:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model
    best_checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR, f"{Config.MODEL_NAME}_best.pth"
    )
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model with val_loss: {checkpoint['val_loss']:.6f}")

    return model, train_losses, val_losses, best_val_loss


def evaluate_model(model, test_dataloader, criterion, device, logger):
    """Evaluate the model on test data, reconstructing the forecast from residuals."""
    model.eval()
    test_loss = 0.0
    predictions_reconstructed = []
    actuals_list = []

    with torch.no_grad():
        # The dataloader now yields three items
        for X_batch, y_batch_residuals, y_batch_actuals in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch_residuals = y_batch_residuals.to(device)

            # Model predicts the residuals
            outputs_residuals = model(X_batch)

            # Calculate loss on the residuals (this is what the model was trained on)
            loss = criterion(outputs_residuals, y_batch_residuals)
            test_loss += loss.item()

            # --- RECONSTRUCT THE FORECAST ---
            last_input_vals = X_batch[:, -1].unsqueeze(1) # Shape: [batch_size, 1]
            outputs_reconstructed = last_input_vals + outputs_residuals # Add the last value back

            predictions_reconstructed.append(outputs_reconstructed.cpu().numpy())
            actuals_list.append(y_batch_actuals.cpu().numpy())

    test_loss /= len(test_dataloader)
    predictions = np.concatenate(predictions_reconstructed, axis=0)
    actuals = np.concatenate(actuals_list, axis=0)

    # Calculate additional metrics
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)

    # MAPE calculation (avoiding division by zero)
    non_zero_mask = np.abs(actuals) > 1e-8
    mape = (
        np.mean(
            np.abs(
                (actuals[non_zero_mask] - predictions[non_zero_mask])
                / actuals[non_zero_mask]
            )
        )
        * 100
    )

    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test MAE: {mae:.6f}")
    logger.info(f"Test MSE: {mse:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test MAPE: {mape:.2f}%")

    return test_loss, mae, mse, rmse, mape, predictions, actuals


def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training history."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_predictions(predictions, actuals, num_samples=1, save_path=None):
    """Plot predictions vs actuals."""
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(min(num_samples, len(predictions))):
        ax = axes[i]
        ax.plot(actuals[i], label="Actual", marker="o")
        ax.plot(predictions[i], label="Predicted", marker="x")
        ax.set_title(f"Sample {i + 1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def make_prediction_and_plot(model, X_test, device, scaler=None, sample_idx=0, save_path=None):
    """Make a prediction and plot the results with proper denormalization for residual approach."""
    model.eval()
    example_input = X_test[sample_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        # Model predicts residuals
        prediction_residuals = model(example_input).cpu().numpy().flatten()

    # Denormalize the data if scaler is provided
    if scaler is not None:
        # Create dummy array with correct shape (365)
        dummy_array = np.zeros(365)
        dummy_array[:Config.INPUT_SIZE] = example_input.cpu().numpy().flatten()

        # For reconstruction, we need to add the last input value to the residuals
        last_input_val_normalized = example_input.cpu().numpy().flatten()[-1]
        reconstructed_prediction = last_input_val_normalized + prediction_residuals
        dummy_array[Config.INPUT_SIZE:Config.INPUT_SIZE + Config.OUTPUT_SIZE] = reconstructed_prediction

        # Inverse transform using the dummy array
        dummy_denorm = scaler.inverse_transform(dummy_array.reshape(1, -1)).flatten()

        # Extract the relevant parts
        input_denorm = dummy_denorm[:Config.INPUT_SIZE]
        prediction_denorm = dummy_denorm[Config.INPUT_SIZE:Config.INPUT_SIZE + Config.OUTPUT_SIZE]
    else:
        input_denorm = example_input.cpu().numpy().flatten()
        last_input_val = example_input.cpu().numpy().flatten()[-1]
        prediction_denorm = last_input_val + prediction_residuals

    plt.figure(figsize=(12, 6))
    plt.plot(range(Config.INPUT_SIZE), input_denorm, label="Input Data")
    plt.plot(
        range(Config.INPUT_SIZE, Config.INPUT_SIZE + Config.OUTPUT_SIZE),
        prediction_denorm,
        label="Forecast",
        marker="o",
        linestyle='--',
        linewidth=2,
    )

    # Add vertical line to show transition point
    plt.axvline(x=Config.INPUT_SIZE-1, color='red', linestyle=':', alpha=0.7, label='Transition')

    plt.legend()
    plt.title("NHiTS Model Prediction (Residual Approach)")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.grid(True)

    # Print continuity check
    last_input_val = input_denorm[-1]
    first_output_val = prediction_denorm[0]
    continuity_error = abs(last_input_val - first_output_val)
    print(f"Last input value: {last_input_val:.4f}")
    print(f"First output value: {first_output_val:.4f}")
    print(f"Continuity error: {continuity_error:.4f}")
    print(f"Predicted residuals: {prediction_residuals}")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    """Main training pipeline."""
    # Setup
    setup_directories()
    logger = setup_logging()
    logger.info("Starting NHiTS training pipeline")
    logger.info(f"Configuration: {Config.__dict__}")

    # Load and preprocess data
    train_dataloader, val_dataloader, test_dataloader, X_test, scaler = (
        load_and_preprocess_data()
    )
    logger.info(
        f"Data loaded - Train: {len(train_dataloader)}, Val: {len(val_dataloader)}, Test: {len(test_dataloader)}"
    )

    # Setup model
    model, criterion, optimizer, scheduler, device = setup_model()
    model = model.to(device)
    logger.info(f"Model initialized on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    logger.info("Starting training...")
    model, train_losses, val_losses, best_val_loss = train_model(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        scheduler,
        device,
        logger,
    )

    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, mae, mse, rmse, mape, predictions, actuals = evaluate_model(
        model, test_dataloader, criterion, device, logger
    )

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": str(best_val_loss),
        "test_metrics": {
            "loss": str(test_loss),
            "mae": str(mae),
            "mse": str(mse),
            "rmse": str(rmse),
            "mape": str(mape),
        },
    }

    history_path = os.path.join(Config.LOG_DIR, f"{Config.MODEL_NAME}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    # Plot training history
    plot_training_history(
        train_losses,
        val_losses,
        save_path=os.path.join(
            Config.LOG_DIR, f"{Config.MODEL_NAME}_training_history.png"
        ),
    )

    # Plot predictions
    plot_predictions(
        predictions,
        actuals,
        num_samples=1,
        save_path=os.path.join(Config.LOG_DIR, f"{Config.MODEL_NAME}_predictions.png"),
    )

    # Make prediction and plot
    make_prediction_and_plot(
        model,
        X_test,
        device,
        scaler,
        save_path=os.path.join(Config.LOG_DIR, f"{Config.MODEL_NAME}_forecast.png"),
    )

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
