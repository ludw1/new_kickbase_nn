"""
Model configurations module.

Contains all model configuration classes and their setup methods.
"""

from config import Config
from utils import ModelTracker, LossRecorder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.dataprocessing.transformers import Scaler

class Models:
    """Container class for all model configurations."""

    class NHiTSModelConfig:
        def __init__(self):
            self.input_size = Config.INPUT_SIZE
            self.output_size = Config.OUTPUT_SIZE
            self.learning_rate = Config.LEARNING_RATE
            self.weight_decay = Config.WEIGHT_DECAY
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
                num_blocks=2,
                num_stacks=3,
                dropout=0.1,
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

            return model, loss_recorder, model_tracker

    class TFTConfig:
        def __init__(self):
            self.input_size = Config.INPUT_SIZE
            self.output_size = Config.OUTPUT_SIZE
            self.learning_rate = Config.LEARNING_RATE
            self.weight_decay = Config.WEIGHT_DECAY
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
            """Setup the TFT model with static covariates support."""
            from darts.models import TFTModel

            model_tracker = ModelTracker("tft", self.log_dir)
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
            model = TFTModel(
                input_chunk_length=self.input_size,
                output_chunk_length=self.output_size,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                model_name=self.model_name,
                random_state=self.seed,
                hidden_size=32,
                add_relative_index=True,
                lstm_layers=2,
                num_attention_heads=4,
                dropout=0.1,
                lr_scheduler_cls=ReduceLROnPlateau,
                lr_scheduler_kwargs={
                    "mode": "min",
                    "factor": self.scheduler_factor,
                    "patience": self.scheduler_patience,
                },
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

            return model, loss_recorder, model_tracker

    class NLinearConfig:
        def __init__(self):
            self.input_size = Config.INPUT_SIZE
            self.output_size = Config.OUTPUT_SIZE
            self.learning_rate = Config.LEARNING_RATE
            self.weight_decay = Config.WEIGHT_DECAY
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

            return model, loss_recorder, model_tracker

    class TiDEConfig:
        def __init__(self):
            self.input_size = Config.INPUT_SIZE
            self.output_size = Config.OUTPUT_SIZE
            self.learning_rate = Config.LEARNING_RATE
            self.weight_decay = Config.WEIGHT_DECAY
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
                num_encoder_layers=2,
                num_decoder_layers=2,
                add_encoders=encoders,
                hidden_size=128,
                dropout=0.1,
                lr_scheduler_cls=ReduceLROnPlateau,
                lr_scheduler_kwargs={
                    "mode": "min",
                    "factor": self.scheduler_factor,
                    "patience": self.scheduler_patience,
                },
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

            return model, loss_recorder, model_tracker

    class LinearRegressionConfig:
        def __init__(self):
            self.input_size = Config.INPUT_SIZE
            self.output_size = Config.OUTPUT_SIZE
            self.learning_rate = Config.LEARNING_RATE
            self.weight_decay = Config.WEIGHT_DECAY
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

            return model, loss_recorder, model_tracker