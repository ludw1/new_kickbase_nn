"""
Configuration module for training pipeline.

Contains all hyperparameters and settings for model training and evaluation.
"""

# Configuration
class Config:
    # Series-based split: split players into train/val/test sets
    TRAIN_SPLIT = 0.70  # 70% of players for training
    VAL_SPLIT = 0.15  # 15% of players for validation
    TEST_SPLIT = 0.15  # 15% of players for testing
    INPUT_SIZE = 30
    OUTPUT_SIZE = 3
    NUM_EPOCHS = 20
    BATCH_SIZE = 300
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    MIN_DELTA = 0.05
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    MODEL_NAME = "kickbase_model"
    SEED = 42
    MODEL_TYPE = "tide"  # Options: "nhits", "nlinear", "tft", "tide", "linear_regression"