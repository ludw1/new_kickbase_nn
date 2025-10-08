"""
Configuration module for training pipeline.

Contains all hyperparameters and settings for model training, optimization, and evaluation.
"""


class TrainingConfig:
    """Configuration for model training."""
    
    # Series-based split: split players into train/val/test sets
    TRAIN_SPLIT = 0.70  # 70% of players for training
    VAL_SPLIT = 0.15  # 15% of players for validation
    TEST_SPLIT = 0.15  # 15% of players for testing
    
    # Training hyperparameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    MIN_DELTA = 1e-4
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # General settings
    SEED = 42
    MODEL_TYPE = "linear_regression"  # Options: "nhits", "nlinear", "tide", "linear_regression"
    MODEL_NAME = "kickbase_model"


class OptimizationConfig:
    """Configuration for hyperparameter optimization using Optuna."""
    
    # Which model(s) to optimize
    # Options: "tide", "nhits", "nlinear", "linear_regression", "all"
    MODEL_TO_OPTIMIZE = "all"
    
    # Number of Optuna trials to run
    N_TRIALS = 50
    
    # Number of epochs per trial (lower than full training for speed)
    N_EPOCHS_PER_TRIAL = 10
    
    # Whether to enable parallel trials (requires proper setup)
    N_JOBS = 1  # Set to -1 to use all available cores
    
    # Timeout per trial in seconds (None for no timeout)
    TRIAL_TIMEOUT = None
    
    # Whether to show progress bar during optimization
    SHOW_PROGRESS_BAR = True
    
    # Study name prefix (timestamp will be added automatically)
    STUDY_NAME_PREFIX = "kickbase_optimization"


class EvaluationConfig:
    """Configuration for model evaluation and comparison."""
    
    # Model checkpoint paths - update these to point to your trained models
    MODEL_PATHS = {
        "nhits": "checkpoints/nhits_kickbase_model_final.pth",
        "nlinear": "checkpoints/nlinear_kickbase_model_final.pth",
        "tide": "checkpoints/tide_kickbase_model_final.pth",
        "linear_regression": "checkpoints/linear_regression_kickbase_model_final.pth",
    }
    
    # Prediction length multiplier (2x = predict twice the trained output size)
    PREDICTION_LENGTH_MULTIPLIER = 2
    
    # Output paths
    COMPARISON_TABLE_PATH = "figs/model_comparison_table.csv"
    COMPARISON_PLOT_PATH = "figs/model_comparison_plot.png"
    DETAILED_RESULTS_PATH = "figs/detailed_evaluation_results.csv"
    
    # Evaluation settings
    SEED = 42

