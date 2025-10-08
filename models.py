"""
Model configurations module.

Contains all model configuration classes and their setup methods.
"""

from config import Config
from utils import ModelTracker, LossRecorder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import L1Loss
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.dataprocessing.transformers import Scaler

class Models:
    """Container class for all model configurations."""

    class NHiTSModelConfig:
        def __init__(self):
            # Optuna optimized parameters (Trial 13, Val MAE: 0.0093)
            self.input_size = 10  # input_chunk_length
            self.output_size = 3  # output_chunk_length
            self.learning_rate = 0.00023384895140376912
            self.num_stacks = 2
            self.num_blocks = 3
            self.num_layers = 3
            self.layer_widths = 128
            self.dropout = 0.1
            # Config parameters
            self.patience = Config.PATIENCE
            self.min_delta = Config.MIN_DELTA
            self.scheduler_factor = Config.SCHEDULER_FACTOR
            self.scheduler_patience = Config.SCHEDULER_PATIENCE
            self.checkpoint_dir = Config.CHECKPOINT_DIR
            self.log_dir = Config.LOG_DIR
            self.model_name = Config.MODEL_NAME
            self.seed = Config.SEED
            self.n_epochs = Config.NUM_EPOCHS
            self.batch_size = Config.BATCH_SIZE

        def setup_model(self) -> tuple:
            """Setup the model, criterion, and optimizer."""
            from darts.models import NHiTSModel

            model_tracker = ModelTracker("nhits", self.log_dir)
            loss_recorder = LossRecorder(model_tracker)
            early_stopper = EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                min_delta=self.min_delta,
                mode="min",
            )
            pl_trainer_kwargs = {
                "callbacks": [early_stopper, loss_recorder],
                "enable_checkpointing": True,
                "default_root_dir": self.checkpoint_dir,
            }
            model = NHiTSModel(
                input_chunk_length=self.input_size,
                output_chunk_length=self.output_size,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                num_stacks=self.num_stacks,
                num_blocks=self.num_blocks,
                num_layers=self.num_layers,
                layer_widths=self.layer_widths,
                dropout=self.dropout,
                optimizer_kwargs={"lr": self.learning_rate},
                model_name=self.model_name,
                random_state=self.seed,
                lr_scheduler_cls=ReduceLROnPlateau,
                lr_scheduler_kwargs={
                    "mode": "min",
                    "factor": self.scheduler_factor,
                    "patience": self.scheduler_patience,
                },
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

            return model, loss_recorder, model_tracker, self.input_size, self.output_size

    class NLinearConfig:
        def __init__(self):
            # Optuna optimized parameters (Trial 46, Val MAE: 0.0128)
            self.input_size = 10  # input_chunk_length
            self.output_size = 3  # output_chunk_length
            self.learning_rate = 0.0007898138167929168
            self.shared_weights = True
            self.const_init = True
            self.normalize = False
            # Config parameters
            self.patience = Config.PATIENCE
            self.min_delta = Config.MIN_DELTA
            self.scheduler_factor = Config.SCHEDULER_FACTOR
            self.scheduler_patience = Config.SCHEDULER_PATIENCE
            self.checkpoint_dir = Config.CHECKPOINT_DIR
            self.log_dir = Config.LOG_DIR
            self.model_name = Config.MODEL_NAME
            self.seed = Config.SEED
            self.n_epochs = Config.NUM_EPOCHS
            self.batch_size = Config.BATCH_SIZE

        def setup_model(self):
            """Setup the NLinear model with static covariates support."""
            from darts.models import NLinearModel

            model_tracker = ModelTracker("nlinear", self.log_dir)
            loss_recorder = LossRecorder(model_tracker)
            early_stopper = EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                min_delta=self.min_delta,
                mode="min",
            )
            pl_trainer_kwargs = {
                "callbacks": [early_stopper, loss_recorder],
                "enable_checkpointing": True,
                "default_root_dir": self.checkpoint_dir,
            }
            model = NLinearModel(
                input_chunk_length=self.input_size,
                output_chunk_length=self.output_size,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                shared_weights=self.shared_weights,
                const_init=self.const_init,
                normalize=self.normalize,
                optimizer_kwargs={"lr": self.learning_rate},
                model_name=self.model_name,
                random_state=self.seed,
                lr_scheduler_cls=ReduceLROnPlateau,
                lr_scheduler_kwargs={
                    "mode": "min",
                    "factor": self.scheduler_factor,
                    "patience": self.scheduler_patience,
                },
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

            return model, loss_recorder, model_tracker, self.input_size, self.output_size

    class TiDEConfig:
        def __init__(self):
            # Optuna optimized parameters (Trial 13, Val MAE: 0.0113)
            self.input_size = 60  # input_chunk_length
            self.output_size = 3  # output_chunk_length
            self.learning_rate = 0.00021192213100293766
            self.hidden_size = 512
            self.num_encoder_layers = 3
            self.num_decoder_layers = 2
            self.dropout = 0.2
            # Config parameters
            self.patience = Config.PATIENCE
            self.min_delta = Config.MIN_DELTA
            self.scheduler_factor = Config.SCHEDULER_FACTOR
            self.scheduler_patience = Config.SCHEDULER_PATIENCE
            self.checkpoint_dir = Config.CHECKPOINT_DIR
            self.log_dir = Config.LOG_DIR
            self.model_name = Config.MODEL_NAME
            self.seed = Config.SEED
            self.n_epochs = Config.NUM_EPOCHS
            self.batch_size = Config.BATCH_SIZE

        def setup_model(self):
            """Setup the TiDE model with static covariates support."""
            from darts.models import TiDEModel

            model_tracker = ModelTracker("tide", self.log_dir)
            loss_recorder = LossRecorder(model_tracker)
            early_stopper = EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                min_delta=self.min_delta,
                mode="min",
            )
            pl_trainer_kwargs = {
                "callbacks": [early_stopper, loss_recorder],
                "enable_checkpointing": True,
                "default_root_dir": self.checkpoint_dir,
                "accelerator": "auto",
            }
            encoders = {
                "datetime_attribute": {"future": ["month", "day", "dayofweek"]},
                "transformer": Scaler()
            }
            model = TiDEModel(
                input_chunk_length=self.input_size,
                output_chunk_length=self.output_size,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                model_name=self.model_name,
                random_state=self.seed,
                num_encoder_layers=self.num_encoder_layers,
                num_decoder_layers=self.num_decoder_layers,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                optimizer_kwargs={"lr": self.learning_rate},
                loss_fn=L1Loss(),
                add_encoders=encoders,
                lr_scheduler_cls=ReduceLROnPlateau,
                lr_scheduler_kwargs={
                    "mode": "min",
                    "factor": self.scheduler_factor,
                    "patience": self.scheduler_patience,
                },
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

            return model, loss_recorder, model_tracker, self.input_size, self.output_size

    class LinearRegressionConfig:
        def __init__(self):
            # Optuna optimized parameters (Trial 10, Val MAE: 0.0095)
            self.input_size = 10  # lags
            self.output_size = 3  # output_chunk_length
            # Config parameters
            self.log_dir = Config.LOG_DIR
            self.model_name = Config.MODEL_NAME
            self.seed = Config.SEED

        def setup_model(self):
            """Setup the Linear Regression model with static covariates support."""
            from darts.models import LinearRegressionModel

            model_tracker = ModelTracker("linear_regression", self.log_dir)
            model = LinearRegressionModel(
                lags=self.input_size,
                output_chunk_length=self.output_size,
                use_static_covariates=True,
                random_state=self.seed,
            )

            # Create a simple loss recorder since Linear Regression doesn't use PyTorch Lightning
            class LinearLossRecorder:
                def __init__(self):
                    self.train_losses = []
                    self.val_losses = []

            loss_recorder = LinearLossRecorder()

            return model, loss_recorder, model_tracker, self.input_size, self.output_size