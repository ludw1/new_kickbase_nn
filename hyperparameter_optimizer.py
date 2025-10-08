"""
Hyperparameter Optimization using Optuna for Kickbase Time Series Models.

This script implements systematic Bayesian optimization to find optimal hyperparameters
for all model types: TiDE, NHiTS, NLinear, and Linear Regression.

Configuration:
    Set optimization parameters in config.py under OptimizationConfig class:
    - MODEL_TO_OPTIMIZE: Which model to optimize ("tide", "nhits", etc., or "all")
    - N_TRIALS: Number of optimization trials
    - N_EPOCHS_PER_TRIAL: Epochs per trial (lower than full training)
    
Usage:
    python hyperparameter_optimization.py
"""

import os
import numpy as np
import logging
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)
from datetime import datetime
import gc
import torch
# Import from existing modules
from config import Config, OptimizationConfig
from utils import setup_directories, setup_logging
from data_processing import load_and_preprocess_data
from darts.metrics import mae
from darts.models import TiDEModel, NHiTSModel, NLinearModel, LinearRegressionModel
from darts.dataprocessing.transformers import Scaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn import L1Loss

logger = logging.getLogger(__name__)
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            logger.warning(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)

class OptunaOptimizer:
    """Handles hyperparameter optimization for all model types."""

    def __init__(self, model_type: str, n_trials: int = 50, n_epochs: int = 30):
        """
        Initialize the optimizer.

        Args:
            model_type: One of 'tide', 'nhits', 'nlinear', 'linear_regression', or 'all'
            n_trials: Number of optimization trials
            n_epochs: Number of epochs per trial (lower than full training for speed)
        """
        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.n_epochs = n_epochs

        # Load data once
        logger.info("Loading data for optimization...")
        (
            self.train_series,
            self.val_series,
            self.test_series,
            self.train_static_cov,
            self.val_static_cov,
            self.test_static_cov,
        ) = load_and_preprocess_data()

        logger.info(
            f"Loaded {len(self.train_series)} train, {len(self.val_series)} val series"
        )

        # Prepare series with static covariates for models that need them
        self.train_with_cov = self._add_static_covariates(
            self.train_series, self.train_static_cov
        )
        self.val_with_cov = self._add_static_covariates(
            self.val_series, self.val_static_cov
        )

        # Create optimization directory
        self.optuna_dir = os.path.join(Config.LOG_DIR, "optuna_studies")
        os.makedirs(self.optuna_dir, exist_ok=True)

    def _cleanup_memory(self, model=None):
        """
        Clean up memory after each trial to prevent slowdown.
        
        Args:
            model: The model to delete (optional)
        """
        if model is not None:
            # Delete the model
            del model
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _add_static_covariates(self, series_list, static_cov_list):
        """Add static covariates to time series."""
        series_with_cov = []
        for i, series in enumerate(series_list):
            series_with_cov.append(series.with_static_covariates(static_cov_list[i]))
        return series_with_cov

    def optimize(self):
        """Run optimization for the specified model type(s)."""
        if self.model_type == "all":
            models = ["tide", "nhits", "nlinear", "linear_regression"]
            logger.info("Optimizing all models sequentially...")
            results = {}
            for model in models:
                logger.info(f"\n{'='*60}")
                logger.info(f"Starting optimization for {model.upper()}")
                logger.info(f"{'='*60}\n")
                self.model_type = model
                study = self._run_optimization()
                results[model] = study.best_params
            return results
        else:
            return self._run_optimization()

    def _run_optimization(self):
        """Run Optuna optimization study."""
        study_name = f"{self.model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage_name = f"sqlite:///{os.path.join(self.optuna_dir, study_name)}.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="minimize",
            load_if_exists=True,
        )

        logger.info(f"Starting optimization for {self.model_type.upper()}")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Epochs per trial: {self.n_epochs}")

        # Choose the appropriate objective function
        if self.model_type == "tide":
            objective_func = self._objective_tide
        elif self.model_type == "nhits":
            objective_func = self._objective_nhits
        elif self.model_type == "nlinear":
            objective_func = self._objective_nlinear
        elif self.model_type == "linear_regression":
            objective_func = self._objective_linear_regression
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            n_jobs=OptimizationConfig.N_JOBS,
            timeout=OptimizationConfig.TRIAL_TIMEOUT,
            show_progress_bar=OptimizationConfig.SHOW_PROGRESS_BAR,
        )

        # Log best results
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMIZATION COMPLETE: {self.model_type.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best validation MAE: {study.best_value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

        # Save results
        self._save_results(study)
        # self._visualize_study(study)

        return study

    def _objective_tide(self, trial):
        """
        Objective function for TiDE model optimization.

        Key hyperparameters to tune:
        - input_chunk_length: Input sequence length [10, 20, 30, ..., 90]
        - output_chunk_length: Output sequence length [3, 4, 5, ..., 10]
        - hidden_size: Model capacity [64, 128, 256, 512]
        - num_encoder_layers: Encoder depth [1, 2, 3]
        - num_decoder_layers: Decoder depth [1, 2, 3]
        - dropout: Regularization [0.1, 0.2, 0.3, 0.4]
        - learning_rate: [1e-4, 5e-4, 1e-3]
        """
        # Suggest hyperparameters
        input_chunk_length = trial.suggest_int("input_chunk_length", 10, 90, step=10)
        output_chunk_length = trial.suggest_int("output_chunk_length", 3, 10, step=1)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 3)
        num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

        # Build model
        try:
            pruner = PyTorchLightningPruningCallback(trial, "val_loss")
            early_stopper = EarlyStopping(
                monitor="val_loss",
                patience=Config.PATIENCE,
                min_delta=Config.MIN_DELTA,
                mode="min",
            )
            pl_trainer_kwargs = {
                "callbacks": [early_stopper, pruner],
                "enable_checkpointing": False,
                "enable_progress_bar": True,
                "enable_model_summary": True,
                "accelerator": "auto",
            }

            encoders = {
                "datetime_attribute": {"future": ["month", "day", "dayofweek"]},
                "transformer": Scaler(),
            }

            model = TiDEModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_epochs=self.n_epochs,
                batch_size=Config.BATCH_SIZE,
                model_name=f"tide_trial_{trial.number}",
                random_state=Config.SEED,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                hidden_size=hidden_size,
                dropout=dropout,
                loss_fn=L1Loss(),
                add_encoders=encoders,
                optimizer_kwargs={"lr": learning_rate},
                lr_scheduler_cls=ReduceLROnPlateau,
                lr_scheduler_kwargs={
                    "mode": "min",
                    "factor": Config.SCHEDULER_FACTOR,
                    "patience": Config.SCHEDULER_PATIENCE,
                },
                pl_trainer_kwargs=pl_trainer_kwargs,
            )
            # Train model
            model.fit(
                self.train_with_cov,
                val_series=self.val_with_cov,
                verbose=True,

            )

            # Evaluate on validation set
            # Create validation inputs (exclude last output_chunk_length points)
            val_inputs = [series[:-output_chunk_length] for series in self.val_with_cov]
            val_targets = [series[-output_chunk_length:] for series in self.val_with_cov]

            predictions = model.predict(n=output_chunk_length, series=val_inputs)
            val_mae = float(np.mean(mae(val_targets, predictions)))

            logger.info(
                f"Trial {trial.number}: input_len={input_chunk_length}, output_len={output_chunk_length}, "
                f"hidden_size={hidden_size}, enc_layers={num_encoder_layers}, dec_layers={num_decoder_layers}, "
                f"dropout={dropout:.2f}, lr={learning_rate:.5f}, val_MAE={val_mae:.4f}"
            )

            # Clean up memory before returning
            self._cleanup_memory(model)

            return val_mae

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Clean up even on failure
            self._cleanup_memory()
            return float("inf")

    def _objective_nhits(self, trial):
        """
        Objective function for NHiTS model optimization.

        Key hyperparameters:
        - input_chunk_length: Input sequence length [10, 20, 30, ..., 90]
        - output_chunk_length: Output sequence length [3, 4, 5, ..., 10]
        - num_stacks: Number of hierarchical stacks [2, 3, 4]
        - num_blocks: Blocks per stack [1, 2, 3]
        - num_layers: Layers per block [2, 3, 4]
        - layer_widths: Width of layers [128, 256, 512]
        - dropout: [0.1, 0.2, 0.3, 0.4]
        - learning_rate: [1e-4, 5e-4, 1e-3]
        """
        input_chunk_length = trial.suggest_int("input_chunk_length", 10, 90, step=10)
        output_chunk_length = trial.suggest_int("output_chunk_length", 3, 10, step=1)
        num_stacks = trial.suggest_int("num_stacks", 2, 4)
        num_blocks = trial.suggest_int("num_blocks", 1, 3)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        layer_widths = trial.suggest_categorical("layer_widths", [128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

        try:
            early_stopper = EarlyStopping(
                monitor="val_loss",
                patience=Config.PATIENCE,
                min_delta=Config.MIN_DELTA,
                mode="min",
            )
            pruner = PyTorchLightningPruningCallback(trial, "val_loss")
            pl_trainer_kwargs = {
                "callbacks": [early_stopper, pruner],
                "enable_checkpointing": False,
                "enable_progress_bar": False,  # Disable to reduce overhead
                "enable_model_summary": False,  # Disable to reduce overhead
                "accelerator": "auto",
            }

            model = NHiTSModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_epochs=self.n_epochs,
                batch_size=Config.BATCH_SIZE,
                num_stacks=num_stacks,
                num_blocks=num_blocks,
                num_layers=num_layers,
                layer_widths=layer_widths,
                dropout=dropout,
                model_name=f"nhits_trial_{trial.number}",
                random_state=Config.SEED,
                optimizer_kwargs={"lr": learning_rate},
                lr_scheduler_cls=ReduceLROnPlateau,
                lr_scheduler_kwargs={
                    "mode": "min",
                    "factor": Config.SCHEDULER_FACTOR,
                    "patience": Config.SCHEDULER_PATIENCE,
                },
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

            # NHiTS doesn't use static covariates
            model.fit(self.train_series, val_series=self.val_series, verbose=True)

            # Evaluate
            val_inputs = [series[:-output_chunk_length] for series in self.val_series]
            val_targets = [series[-output_chunk_length:] for series in self.val_series]

            predictions = model.predict(n=output_chunk_length, series=val_inputs)
            val_mae = float(np.mean(mae(val_targets, predictions)))

            logger.info(
                f"Trial {trial.number}: input_len={input_chunk_length}, output_len={output_chunk_length}, "
                f"stacks={num_stacks}, blocks={num_blocks}, layers={num_layers}, width={layer_widths}, "
                f"dropout={dropout:.2f}, lr={learning_rate:.5f}, val_MAE={val_mae:.4f}"
            )

            # Clean up memory before returning
            self._cleanup_memory(model)

            return val_mae

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Clean up even on failure
            self._cleanup_memory()
            return float("inf")

    def _objective_nlinear(self, trial):
        """
        Objective function for NLinear model optimization.

        Key hyperparameters:
        - input_chunk_length: Input sequence length [10, 20, 30, ..., 90]
        - output_chunk_length: Output sequence length [3, 4, 5, ..., 10]
        - shared_weights: Whether to share weights across dimensions
        - const_init: Initialize final layer bias to last value
        - learning_rate: [1e-4, 5e-4, 1e-3]
        - normalize: Whether to apply normalization
        """
        input_chunk_length = trial.suggest_int("input_chunk_length", 10, 90, step=10)
        output_chunk_length = trial.suggest_int("output_chunk_length", 3, 10, step=1)
        shared_weights = trial.suggest_categorical("shared_weights", [True, False])
        const_init = trial.suggest_categorical("const_init", [True, False])
        normalize = trial.suggest_categorical("normalize", [True, False])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

        try:
            early_stopper = EarlyStopping(
                monitor="val_loss",
                patience=Config.PATIENCE,
                min_delta=Config.MIN_DELTA,
                mode="min",
            )
            pruner = PyTorchLightningPruningCallback(trial, "val_loss")
            pl_trainer_kwargs = {
                "callbacks": [early_stopper, pruner],
                "enable_checkpointing": False,
                "enable_progress_bar": False,  # Disable to reduce overhead
                "enable_model_summary": False,  # Disable to reduce overhead
                "accelerator": "auto",
            }

            model = NLinearModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_epochs=self.n_epochs,
                batch_size=Config.BATCH_SIZE,
                shared_weights=shared_weights,
                const_init=const_init,
                normalize=normalize,
                model_name=f"nlinear_trial_{trial.number}",
                random_state=Config.SEED,
                optimizer_kwargs={"lr": learning_rate},
                lr_scheduler_cls=ReduceLROnPlateau,
                lr_scheduler_kwargs={
                    "mode": "min",
                    "factor": Config.SCHEDULER_FACTOR,
                    "patience": Config.SCHEDULER_PATIENCE,
                },
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

            # NLinear can use static covariates
            model.fit(
                self.train_with_cov,
                val_series=self.val_with_cov,
                verbose=True,
            )

            # Evaluate
            val_inputs = [series[:-output_chunk_length] for series in self.val_with_cov]
            val_targets = [series[-output_chunk_length:] for series in self.val_with_cov]

            predictions = model.predict(n=output_chunk_length, series=val_inputs)
            val_mae = float(np.mean(mae(val_targets, predictions)))

            logger.info(
                f"Trial {trial.number}: input_len={input_chunk_length}, output_len={output_chunk_length}, "
                f"shared_weights={shared_weights}, const_init={const_init}, normalize={normalize}, "
                f"lr={learning_rate:.5f}, val_MAE={val_mae:.4f}"
            )

            # Clean up memory before returning
            self._cleanup_memory(model)

            return val_mae

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Clean up even on failure
            self._cleanup_memory()
            return float("inf")

    def _objective_linear_regression(self, trial):
        """
        Objective function for Linear Regression model optimization.

        Key hyperparameters:
        - lags: Number of past lags to use [10, 20, 30, ..., 90]
        - output_chunk_length: Output sequence length [3, 4, 5, ..., 10]
        - lags_past_covariates: Past covariate lags [None, 10, 20, 30]
        """
        lags = trial.suggest_int("lags", 10, 90, step=10)
        output_chunk_length = trial.suggest_int("output_chunk_length", 3, 10, step=1)

        try:
            model = LinearRegressionModel(
                lags=lags,
                output_chunk_length=output_chunk_length,
                use_static_covariates=True,
                random_state=Config.SEED,
            )

            # Linear Regression has no epochs - direct fitting
            model.fit(self.train_with_cov)

            # Evaluate
            val_inputs = [series[:-output_chunk_length] for series in self.val_with_cov]
            val_targets = [series[-output_chunk_length:] for series in self.val_with_cov]

            predictions = model.predict(n=output_chunk_length, series=val_inputs)
            val_mae = float(np.mean(mae(val_targets, predictions)))

            logger.info(f"Trial {trial.number}: lags={lags}, output_len={output_chunk_length}, val_MAE={val_mae:.4f}")

            # Clean up memory before returning
            self._cleanup_memory(model)

            return val_mae

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Clean up even on failure
            self._cleanup_memory()
            return float("inf")

    def _save_results(self, study):
        """Save optimization results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save best parameters as text
        params_file = os.path.join(
            self.optuna_dir, f"{self.model_type}_best_params_{timestamp}.txt"
        )
        with open(params_file, "w") as f:
            f.write(f"Best Hyperparameters for {self.model_type.upper()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Best Trial: {study.best_trial.number}\n")
            f.write(f"Best Validation MAE: {study.best_value:.4f}\n\n")
            f.write("Parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")

        # Save trial history as CSV
        df = study.trials_dataframe()
        csv_file = os.path.join(
            self.optuna_dir, f"{self.model_type}_trials_{timestamp}.csv"
        )
        df.to_csv(csv_file, index=False)

        logger.info(f"Results saved to {params_file} and {csv_file}")

    def _visualize_study(self, study):
        """Create visualizations of the optimization study."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Optimization history
            fig1 = plot_optimization_history(study)
            fig1.write_image(
                os.path.join(
                    self.optuna_dir,
                    f"{self.model_type}_optimization_history_{timestamp}.png",
                )
            )

            # Parameter importance
            fig2 = plot_param_importances(study)
            fig2.write_image(
                os.path.join(
                    self.optuna_dir,
                    f"{self.model_type}_param_importance_{timestamp}.png",
                )
            )

            # Parallel coordinate plot
            fig3 = plot_parallel_coordinate(study)
            fig3.write_image(
                os.path.join(
                    self.optuna_dir,
                    f"{self.model_type}_parallel_coordinate_{timestamp}.png",
                )
            )

            logger.info("Visualizations saved successfully")

        except Exception as e:
            logger.warning(f"Could not create some visualizations: {str(e)}")


def main():
    """Main function to run hyperparameter optimization using config values."""
    # Setup
    setup_directories()
    setup_logging()

    # Get configuration from OptimizationConfig
    model_type = OptimizationConfig.MODEL_TO_OPTIMIZE
    n_trials = OptimizationConfig.N_TRIALS
    n_epochs = OptimizationConfig.N_EPOCHS_PER_TRIAL

    logger.info(f"\n{'='*60}")
    logger.info("KICKBASE MODEL HYPERPARAMETER OPTIMIZATION")
    logger.info(f"{'='*60}\n")
    logger.info(f"Model type: {model_type.upper()}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Epochs per trial: {n_epochs}")
    logger.info(f"N_JOBS: {OptimizationConfig.N_JOBS}")
    logger.info(f"Show progress bar: {OptimizationConfig.SHOW_PROGRESS_BAR}")
    logger.info(f"{'='*60}\n")

    # Validate model type
    valid_models = ["tide", "nhits", "nlinear", "linear_regression", "all"]
    if model_type.lower() not in valid_models:
        raise ValueError(
            f"Invalid MODEL_TO_OPTIMIZE: {model_type}. "
            f"Must be one of {valid_models}"
        )

    # Run optimization
    optimizer = OptunaOptimizer(
        model_type=model_type,
        n_trials=n_trials,
        n_epochs=n_epochs,
    )

    results = optimizer.optimize()

    logger.info(f"\n{'='*60}")
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info(f"{'='*60}\n")

    if isinstance(results, dict):
        # Multiple models optimized
        logger.info("Summary of best parameters for all models:")
        for model, params in results.items():
            logger.info(f"\n{model.upper()}:")
            for key, value in params.items():
                logger.info(f"  {key}: {value}")
    else:
        logger.info("Best parameters saved to logs/optuna_studies/")
        logger.info(f"Check the latest {model_type}_best_params_*.txt file")


if __name__ == "__main__":
    main()
