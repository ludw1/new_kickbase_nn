import asyncio
from config import PipelineConfig
import os
from datetime import datetime
from config import TrainingConfig as Config
import logging

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(Config.LOG_DIR, f"training_{timestamp}.log")
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


if __name__ == "__main__":
    setup_logging()
    if PipelineConfig.mode == "get_data":
        from get_data.get_player_data import get_api_data

        asyncio.run(get_api_data())
    elif PipelineConfig.mode == "train":
        from train_model.training import train_model

        train_model()
    elif PipelineConfig.mode == "optimize":
        from train_model.hyperparameter_optimizer import run_optimizer

        run_optimizer()
    elif PipelineConfig.mode == "evaluate":
        from evaluate_model.eval import evaluate_models

        evaluate_models()
