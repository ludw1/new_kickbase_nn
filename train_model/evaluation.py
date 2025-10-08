"""
Model evaluation and backtesting module.

Contains functions for comprehensive model evaluation, backtesting,
and performance analysis. Evaluates all models together and produces comparison
tables and graphs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from darts.metrics import mae, rmse, mape, smape
from darts.models import NHiTSModel, NLinearModel, TiDEModel, LinearRegressionModel
from config import EvaluationConfig, TrainingConfig as Config
from utils import setup_directories, setup_logging
from data_processing import load_and_preprocess_data
from models import Models
import torch
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.rcParams['figure.figsize'] = (14, 8)


def evaluate_model(model, test_series, test_static_cov, input_size, output_size):
    """Evaluate the trained model on the test set.
    
    Args:
        model: Trained model
        test_series: Test time series
        test_static_cov: Test static covariates
        input_size: Input size used by the model
        output_size: Output size used by the model
        
    Returns:
        tuple: (test_mae, test_rmse) - average test metrics
    """
    logger.info("="*50)
    logger.info("EVALUATING MODEL ON TEST SET")
    logger.info("="*50)
    
    # For proper evaluation, we need to split test series into input and target
    # The model uses the last input_size points to predict the next output_size points
    test_inputs = []
    test_targets = []

    for i, series in enumerate(test_series):
        # Use everything except the last output_size points as input
        # Use the last output_size points as target
        if len(series) >= input_size + output_size:
            input_series = series[: -output_size]
            target_series = series[-output_size :]

            # Add static covariates for models that support them
            if Config.MODEL_TYPE in ["tide", "nlinear", "linear_regression"]:
                input_series = input_series.with_static_covariates(test_static_cov[i])

            test_inputs.append(input_series)
            test_targets.append(target_series)

    logger.info(f"Evaluating on {len(test_inputs)} test series...")

    # Predict the next output_size points for each test input
    test_predictions = model.predict(n=output_size, series=test_inputs)
    
    # Plot example prediction
    plt.figure(figsize=(12, 6))
    test_inputs[0].plot(label="Test Input")
    test_targets[0].plot(label="Test Target")
    test_predictions[0].plot(label="Test Prediction")
    plt.legend()
    plt.title("Test Input, Target, and Prediction Example")
    plt.savefig(os.path.join(Config.LOG_DIR, "training_example.png"))
    plt.close()

    # Calculate test metrics by comparing predictions with actual targets
    test_mae_val = mae(test_targets, test_predictions)
    test_rmse_val = rmse(test_targets, test_predictions)

    logger.info("Test Set Metrics:")
    logger.info(f"  MAE: {np.mean(test_mae_val):.4f}")
    logger.info(f"  RMSE: {np.mean(test_rmse_val):.4f}")

    return np.mean(test_mae_val), np.mean(test_rmse_val)


def run_backtests(model, model_name, train_series, val_series, test_series, train_static_cov, val_static_cov, test_static_cov, input_size, output_size):
    """Run comprehensive backtesting on the model.
    
    Args:
        model: Trained model
        model_name: Name of the model
        train_series: Training time series
        val_series: Validation time series
        test_series: Test time series
        train_static_cov: Training static covariates
        val_static_cov: Validation static covariates
        test_static_cov: Test static covariates
        input_size: Input size used by the model
        output_size: Output size used by the model
        
    Returns:
        Dictionary with backtest metrics
    """
    logger.info(f"Running backtests for {model_name.upper()}...")

    # Prepare test series with static covariates
    test_series_with_cov = []
    if model_name in ["tft", "tide", "nlinear", "linear_regression"]:
        for i, series in enumerate(test_series):
            series_with_cov = series.with_static_covariates(test_static_cov[i])
            test_series_with_cov.append(series_with_cov)
    else:
        test_series_with_cov = test_series

    # Metrics storage
    all_mae = []
    all_rmse = []
    all_mape = []
    all_smape = []
    
    # Store first series data for plotting
    first_series = None
    first_forecasts = None

    # Test series length for backtesting
    min_length = min(len(series) for series in test_series_with_cov)

    # Set up backtesting parameters
    forecast_horizon = min(output_size, min_length // 4)  # Use smaller horizon for backtesting
    backtest_start = len(test_series_with_cov[0]) // 4  # Start backtesting at 25% through (extended window)

    # Run backtests on each test series
    for i, test_series in enumerate(test_series_with_cov):
        try:
            # Historical forecast backtest
            historical_forecasts = model.historical_forecasts(
                test_series,
                start=backtest_start,
                forecast_horizon=forecast_horizon,
                stride=forecast_horizon,
                retrain=False,
                verbose=False
            )

            if len(historical_forecasts) > 0:
                # Store first series for plotting
                if i == 0:
                    first_series = test_series
                    first_forecasts = historical_forecasts
                
                # Calculate metrics for this series
                actual = test_series.slice(historical_forecasts.start_time(), historical_forecasts.end_time())

                mae_val = mae(actual, historical_forecasts)
                rmse_val = rmse(actual, historical_forecasts)
                mape_val = mape(actual, historical_forecasts)
                smape_val = smape(actual, historical_forecasts)

                all_mae.append(mae_val)
                all_rmse.append(rmse_val)
                all_mape.append(mape_val)
                all_smape.append(smape_val)

        except Exception as e:
            logger.warning(f"Error backtesting series {i+1}: {e}")
            continue

    # Calculate and return average metrics
    if all_mae:
        return {
            'backtest_mae': float(np.mean(all_mae)),
            'backtest_rmse': float(np.mean(all_rmse)),
            'backtest_mape': float(np.mean(all_mape)),
            'backtest_smape': float(np.mean(all_smape)),
            'backtest_series': first_series,
            'backtest_forecasts': first_forecasts,
        }
    else:
        logger.warning(f"No successful backtests completed for {model_name}")
        return {
            'backtest_mae': float('nan'),
            'backtest_rmse': float('nan'),
            'backtest_mape': float('nan'),
            'backtest_smape': float('nan'),
            'backtest_series': None,
            'backtest_forecasts': None,
        }


def load_model_from_checkpoint(model_name: str, model_path: str) -> Tuple[Any, int, int]:
    """Load a trained model from checkpoint and get its input/output sizes.
    
    Args:
        model_name: Name of the model type (nhits, nlinear, tide, linear_regression)
        model_path: Path to the model checkpoint
        
    Returns:
        Tuple of (loaded model, input_size, output_size)
    """
    logger.info(f"Loading {model_name} model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load model based on type
    if model_name == "nhits":
        model = NHiTSModel.load(model_path)
        _, _, _, input_size, output_size = Models.NHiTSModelConfig().setup_model()
    elif model_name == "nlinear":
        model = NLinearModel.load(model_path)
        _, _, _, input_size, output_size = Models.NLinearConfig().setup_model()
    elif model_name == "tide":
        model = TiDEModel.load(model_path, map_location=torch.device('cpu'))
        _, _, _, input_size, output_size = Models.TiDEConfig().setup_model()
    elif model_name == "linear_regression":
        model = LinearRegressionModel.load(model_path)
        _, _, _, input_size, output_size = Models.LinearRegressionConfig().setup_model()
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    logger.info(f"{model_name} model loaded successfully (input_size={input_size}, output_size={output_size})")
    return model, input_size, output_size


def create_simple_ensemble_predictions(all_model_predictions: List[List], n_samples: int, output_size: int, time_index) -> List:
    """Create simple average ensemble predictions."""
    from darts import TimeSeries
    
    ensemble_predictions = []
    n_models = len(all_model_predictions)
    
    for sample_idx in range(n_samples):
        # Average predictions from all models
        sample_preds = []
        for model_preds in all_model_predictions:
            if sample_idx < len(model_preds):
                sample_preds.append(model_preds[sample_idx].values().flatten())
        
        if len(sample_preds) > 0:
            avg_pred = np.mean(sample_preds, axis=0)
            ensemble_ts = TimeSeries.from_times_and_values(
                times=time_index,
                values=avg_pred
            )
            ensemble_predictions.append(ensemble_ts)
    
    return ensemble_predictions


def evaluate_single_model(
    model_name: str,
    model: Any,
    train_series: List,
    val_series: List,
    test_series: List,
    train_static_cov: List,
    val_static_cov: List,
    test_static_cov: List,
    input_size: int,
    output_size: int,
    extended_output_size: int
) -> Dict[str, float]:
    """Evaluate a single model on the test set with both standard and extended predictions.
    
    Args:
        model_name: Name of the model
        model: Trained model instance
        train_series: Training time series
        val_series: Validation time series
        test_series: Test time series
        train_static_cov: Training static covariates
        val_static_cov: Validation static covariates
        test_static_cov: Test static covariates
        input_size: Input size used by the model
        output_size: Original trained output size
        extended_output_size: Extended output size for out-of-range testing
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name.upper()}...")
    
    # Prepare test inputs and targets
    test_inputs_standard = []
    test_targets_standard = []
    test_inputs_extended = []
    test_targets_extended = []
    
    for i, series in enumerate(test_series):
        # Standard evaluation (within trained range)
        if len(series) >= input_size + output_size:
            input_series = series[:-output_size]
            target_series = series[-output_size:]
            
            if model_name in ["tide", "nlinear", "linear_regression"]:
                input_series = input_series.with_static_covariates(test_static_cov[i])
            
            test_inputs_standard.append(input_series)
            test_targets_standard.append(target_series)
        
        # Extended evaluation (beyond trained range)
        if len(series) >= input_size + extended_output_size:
            input_series_ext = series[:-extended_output_size]
            target_series_ext = series[-extended_output_size:]
            
            if model_name in ["tide", "nlinear", "linear_regression"]:
                input_series_ext = input_series_ext.with_static_covariates(test_static_cov[i])
            
            test_inputs_extended.append(input_series_ext)
            test_targets_extended.append(target_series_ext)
    
    # Standard predictions
    predictions_standard = model.predict(n=output_size, series=test_inputs_standard)
    mae_val = mae(test_targets_standard, predictions_standard)
    rmse_val = rmse(test_targets_standard, predictions_standard)
    mape_val = mape(test_targets_standard, predictions_standard)
    smape_val = smape(test_targets_standard, predictions_standard)
    mae_standard = float(np.mean(mae_val)) if isinstance(mae_val, (list, np.ndarray)) else float(mae_val)
    rmse_standard = float(np.mean(rmse_val)) if isinstance(rmse_val, (list, np.ndarray)) else float(rmse_val)
    mape_standard = float(np.mean(mape_val)) if isinstance(mape_val, (list, np.ndarray)) else float(mape_val)
    smape_standard = float(np.mean(smape_val)) if isinstance(smape_val, (list, np.ndarray)) else float(smape_val)
    
    # Extended predictions
    predictions_extended = model.predict(n=extended_output_size, series=test_inputs_extended)
    mae_val = mae(test_targets_extended, predictions_extended)
    rmse_val = rmse(test_targets_extended, predictions_extended)
    mape_val = mape(test_targets_extended, predictions_extended)
    smape_val = smape(test_targets_extended, predictions_extended)
    mae_extended = float(np.mean(mae_val)) if isinstance(mae_val, (list, np.ndarray)) else float(mae_val)
    rmse_extended = float(np.mean(rmse_val)) if isinstance(rmse_val, (list, np.ndarray)) else float(rmse_val)
    mape_extended = float(np.mean(mape_val)) if isinstance(mape_val, (list, np.ndarray)) else float(mape_val)
    smape_extended = float(np.mean(smape_val)) if isinstance(smape_val, (list, np.ndarray)) else float(smape_val)
    
    # Run backtests
    backtest_results = run_backtests(
        model=model,
        model_name=model_name,
        train_series=train_series,
        val_series=val_series,
        test_series=test_series,
        train_static_cov=train_static_cov,
        val_static_cov=val_static_cov,
        test_static_cov=test_static_cov,
        input_size=input_size,
        output_size=output_size
    )
    
    return {
        'mae_standard': mae_standard,
        'rmse_standard': rmse_standard,
        'mape_standard': mape_standard,
        'smape_standard': smape_standard,
        'mae_extended': mae_extended,
        'rmse_extended': rmse_extended,
        'mape_extended': mape_extended,
        'smape_extended': smape_extended,
        'predictions_standard': predictions_standard[0] if predictions_standard else None,
        'predictions_extended': predictions_extended[0] if predictions_extended else None,
        'test_input': test_inputs_standard[0] if test_inputs_standard else None,
        'test_target_standard': test_targets_standard[0] if test_targets_standard else None,
        'test_target_extended': test_targets_extended[0] if test_targets_extended else None,
        'full_series': test_series[0] if test_series else None,
        **backtest_results
    }


def create_comparison_table(all_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create a comparison table from all model results."""
    logger.info("Creating comparison table...")
    
    data = []
    for model_name, results in all_results.items():
        data.append({
            'Model': model_name.upper(),
            'MAE (Standard)': results['mae_standard'],
            'SMAPE (Standard)': results['smape_standard'],
            'MAE (Extended)': results['mae_extended'],
            'SMAPE (Extended)': results['smape_extended'],
            'MAE (Backtest)': results['backtest_mae'],
            'SMAPE (Backtest)': results['backtest_smape'],
            'MAE Degradation (%)': ((results['mae_extended'] - results['mae_standard']) / results['mae_standard'] * 100),
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('MAE (Standard)')
    
    os.makedirs(os.path.dirname(EvaluationConfig.COMPARISON_TABLE_PATH) or '.', exist_ok=True)
    df.to_csv(EvaluationConfig.COMPARISON_TABLE_PATH, index=False, float_format='%.4f')
    
    print("\n" + "="*120)
    print("MODEL COMPARISON TABLE")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120 + "\n")
    
    return df


def create_metrics_plot(all_results: Dict[str, Dict[str, float]], comparison_df: pd.DataFrame):
    """Create metrics comparison plot (MAE, SMAPE, Backtest)."""
    logger.info("Creating metrics comparison plot...")
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    models = comparison_df['Model'].values
    x = np.arange(len(models))
    width = 0.25
    
    # MAE comparison (Standard, Extended, Backtest)
    ax1 = fig.add_subplot(gs[0, 0])
    mae_std = comparison_df['MAE (Standard)'].values
    mae_ext = comparison_df['MAE (Extended)'].values
    mae_bt = comparison_df['MAE (Backtest)'].values
    ax1.bar(x - width, mae_std, width, label='Standard', alpha=0.8)
    ax1.bar(x, mae_ext, width, label='Extended', alpha=0.8)
    ax1.bar(x + width, mae_bt, width, label='Backtest', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('MAE')
    ax1.set_title('MAE: Standard vs Extended vs Backtest')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # SMAPE comparison (Standard, Extended, Backtest)
    ax2 = fig.add_subplot(gs[0, 1])
    smape_std = comparison_df['SMAPE (Standard)'].values
    smape_ext = comparison_df['SMAPE (Extended)'].values
    smape_bt = comparison_df['SMAPE (Backtest)'].values
    ax2.bar(x - width, smape_std, width, label='Standard', alpha=0.8)
    ax2.bar(x, smape_ext, width, label='Extended', alpha=0.8)
    ax2.bar(x + width, smape_bt, width, label='Backtest', alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('SMAPE (%)')
    ax2.set_title('SMAPE: Standard vs Extended vs Backtest')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # MAE Degradation
    ax3 = fig.add_subplot(gs[0, 2:])
    mae_deg = comparison_df['MAE Degradation (%)'].values
    ax3.bar(x, mae_deg, width*2, alpha=0.8, color='coral')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Degradation (%)')
    ax3.set_title('MAE Degradation (Extended vs Standard)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # Backtest visualization - OVERLAYED (all models in one plot)
    colors = {'nhits': 'blue', 'nlinear': 'green', 'tide': 'red', 'linear_regression': 'purple', 'ensemble': 'orange'}
    
    ax_backtest = fig.add_subplot(gs[1, :])
    
    # Plot actual series once (from first model)
    first_model_results = list(all_results.values())[0]
    if first_model_results.get('backtest_series') is not None:
        first_model_results['backtest_series'].plot(
            ax=ax_backtest, 
            label='Actual', 
            color='black', 
            linewidth=2, 
            alpha=0.8
        )
    
    # Overlay all model forecasts
    for model_name, results in all_results.items():
        if results.get('backtest_forecasts') is not None:
            results['backtest_forecasts'].plot(
                ax=ax_backtest,
                label=f'{model_name.upper()}',
                color=colors.get(model_name, 'gray'),
                linewidth=1.5,
                linestyle='--',
                alpha=0.7
            )
    
    ax_backtest.set_title('Backtest Comparison - All Models', fontsize=11, fontweight='bold')
    ax_backtest.set_xlabel('Time', fontsize=9)
    ax_backtest.set_ylabel('Value', fontsize=9)
    ax_backtest.legend(loc='best', fontsize=9)
    ax_backtest.grid(True, alpha=0.3)
    
    # Save metrics plot
    metrics_path = EvaluationConfig.COMPARISON_PLOT_PATH.replace('.png', '_metrics.png')
    os.makedirs(os.path.dirname(metrics_path) or '.', exist_ok=True)
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    logger.info(f"Metrics plot saved to: {metrics_path}")
    plt.close()


def create_predictions_plot(all_results: Dict[str, Dict[str, float]], extended_output_size: int):
    """Create predictions comparison plot (all models combined in one plot, zoomed to prediction window)."""
    logger.info("Creating predictions comparison plot...")
    
    colors = {'nhits': 'blue', 'nlinear': 'green', 'tide': 'red', 'linear_regression': 'purple', 'ensemble': 'orange'}
    
    # Create single figure with all predictions combined
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    
    # Number of days to show before predictions
    lookback_days = 20
    
    # Get the first valid result to extract the full series
    first_result = None
    for results in all_results.values():
        if results['full_series'] is not None and results['predictions_extended'] is not None:
            first_result = results
            break
    
    if first_result is None:
        logger.warning("No valid results found for predictions plot")
        return
    
    full_series = first_result['full_series']
    
    # Get the actual future values (ground truth)
    actual_future = full_series[-extended_output_size:]
    
    # Get a few days of historical data before predictions
    historical_data = full_series[:-extended_output_size]
    if len(historical_data) > lookback_days:
        historical_context = historical_data[-lookback_days:]
    else:
        historical_context = historical_data
    
    # Plot historical context (last few days before prediction)
    historical_context.plot(ax=ax, label='Historical', color='black', linewidth=2.5, alpha=0.7)
    
    # Plot actual future values
    actual_future.plot(ax=ax, label='Actual', color='black', linestyle='--', linewidth=2.5, alpha=0.9)
    
    # Plot each model's predictions
    for model_name, results in all_results.items():
        if results['predictions_extended'] is not None:
            results['predictions_extended'].plot(
                ax=ax,
                label=f'{model_name.upper()}',
                color=colors.get(model_name, 'gray'),
                linewidth=2,
                alpha=0.8,
                linestyle='-'
            )
    
    # Add vertical line to show prediction start
    if hasattr(actual_future, 'start_time'):
        ax.axvline(x=actual_future.start_time(), color='gray', linestyle=':', linewidth=2, alpha=0.6, label='Prediction Start')
    
    ax.set_title(f'Model Predictions Comparison - {extended_output_size} Day Forecast (Zoomed)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)    # Save predictions plot
    predictions_path = EvaluationConfig.COMPARISON_PLOT_PATH.replace('.png', '_predictions.png')
    os.makedirs(os.path.dirname(predictions_path) or '.', exist_ok=True)
    plt.savefig(predictions_path, dpi=300, bbox_inches='tight')
    logger.info(f"Predictions plot saved to: {predictions_path}")
    plt.close()


def main():
    """Main evaluation function - loads and evaluates all models."""
    setup_directories()
    setup_logging()
    
    logger.info("Starting comprehensive model evaluation...")
    logger.info(f"Prediction length multiplier: {EvaluationConfig.PREDICTION_LENGTH_MULTIPLIER}x")
    
    # Check which models are available
    available_models = {}
    for model_name, model_path in EvaluationConfig.MODEL_PATHS.items():
        if os.path.exists(model_path):
            available_models[model_name] = model_path
            logger.info(f"Found {model_name} model")
        else:
            logger.warning(f"Model not found: {model_name}")

    if not available_models:
        logger.error("No model checkpoints found! Please train models first.")
        return
    
    # Load all models and their configurations
    models_dict = {}
    for model_name, model_path in available_models.items():
        try:
            model, input_size, output_size = load_model_from_checkpoint(model_name, model_path)
            models_dict[model_name] = {
                'model': model,
                'input_size': input_size,
                'output_size': output_size
            }
        except Exception as e:
            logger.error(f"Failed to load {model_name} model: {e}")
            continue
    
    if not models_dict:
        logger.error("Failed to load any models!")
        return
    
    # Get maximum input/output sizes for data loading
    max_input_size = max(m['input_size'] for m in models_dict.values())
    max_output_size = max(m['output_size'] for m in models_dict.values())
    extended_output_size = int(max_output_size * EvaluationConfig.PREDICTION_LENGTH_MULTIPLIER)
    
    logger.info(f"Loading data (extended output size: {extended_output_size} days)...")
    
    # Load and preprocess data
    train_series, val_series, test_series, train_static_cov, val_static_cov, test_static_cov = load_and_preprocess_data(
        input_size=max_input_size,
        output_size=extended_output_size  # Use extended size for data loading
    )
    
    # Evaluate all models and collect their predictions
    all_results = {}
    all_standard_predictions = []
    all_extended_predictions = []
    
    for model_name, model_info in models_dict.items():
        try:
            results = evaluate_single_model(
                model_name=model_name,
                model=model_info['model'],
                train_series=train_series,
                val_series=val_series,
                test_series=test_series,
                train_static_cov=train_static_cov,
                val_static_cov=val_static_cov,
                test_static_cov=test_static_cov,
                input_size=model_info['input_size'],
                output_size=model_info['output_size'],
                extended_output_size=extended_output_size
            )
            all_results[model_name] = results
            
            # Store predictions for ensemble
            # Get all predictions (not just first one)
            test_inputs_standard = []
            test_inputs_extended = []
            test_targets_standard = []
            test_targets_extended = []
            
            for i, series in enumerate(test_series):
                if len(series) >= model_info['input_size'] + model_info['output_size']:
                    input_series = series[:-model_info['output_size']]
                    target_series = series[-model_info['output_size']:]
                    if model_name in ["tide", "nlinear", "linear_regression"]:
                        input_series = input_series.with_static_covariates(test_static_cov[i])
                    test_inputs_standard.append(input_series)
                    test_targets_standard.append(target_series)
                
                if len(series) >= model_info['input_size'] + extended_output_size:
                    input_series_ext = series[:-extended_output_size]
                    target_series_ext = series[-extended_output_size:]
                    if model_name in ["tide", "nlinear", "linear_regression"]:
                        input_series_ext = input_series_ext.with_static_covariates(test_static_cov[i])
                    test_inputs_extended.append(input_series_ext)
                    test_targets_extended.append(target_series_ext)
            
            # Get all predictions
            preds_standard = model_info['model'].predict(n=model_info['output_size'], series=test_inputs_standard)
            preds_extended = model_info['model'].predict(n=extended_output_size, series=test_inputs_extended)
            
            all_standard_predictions.append(preds_standard)
            all_extended_predictions.append(preds_extended)
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        logger.error("Failed to evaluate any models!")
        return
    
    # Create simple ensemble (average of all models)
    logger.info("\nCreating simple ensemble (average of all models)...")
    
    # Prepare test data for ensemble
    test_inputs_standard = []
    test_targets_standard = []
    test_inputs_extended = []
    test_targets_extended = []
    
    for i, series in enumerate(test_series):
        if len(series) >= max_input_size + max_output_size:
            test_inputs_standard.append(series[:-max_output_size])
            test_targets_standard.append(series[-max_output_size:])
        
        if len(series) >= max_input_size + extended_output_size:
            test_inputs_extended.append(series[:-extended_output_size])
            test_targets_extended.append(series[-extended_output_size:])
    
    # Create ensemble predictions
    ensemble_preds_standard = create_simple_ensemble_predictions(
        all_standard_predictions,
        len(test_targets_standard),
        max_output_size,
        test_targets_standard[0].time_index if test_targets_standard else None
    )
    
    ensemble_preds_extended = create_simple_ensemble_predictions(
        all_extended_predictions,
        len(test_targets_extended),
        extended_output_size,
        test_targets_extended[0].time_index if test_targets_extended else None
    )
    
    # Calculate ensemble metrics
    if ensemble_preds_standard and ensemble_preds_extended:
        mae_val = mae(test_targets_standard, ensemble_preds_standard)
        rmse_val = rmse(test_targets_standard, ensemble_preds_standard)
        mape_val = mape(test_targets_standard, ensemble_preds_standard)
        smape_val = smape(test_targets_standard, ensemble_preds_standard)
        mae_standard = float(np.mean(mae_val)) if isinstance(mae_val, (list, np.ndarray)) else float(mae_val)
        rmse_standard = float(np.mean(rmse_val)) if isinstance(rmse_val, (list, np.ndarray)) else float(rmse_val)
        mape_standard = float(np.mean(mape_val)) if isinstance(mape_val, (list, np.ndarray)) else float(mape_val)
        smape_standard = float(np.mean(smape_val)) if isinstance(smape_val, (list, np.ndarray)) else float(smape_val)
        
        mae_val = mae(test_targets_extended, ensemble_preds_extended)
        rmse_val = rmse(test_targets_extended, ensemble_preds_extended)
        mape_val = mape(test_targets_extended, ensemble_preds_extended)
        smape_val = smape(test_targets_extended, ensemble_preds_extended)
        mae_extended = float(np.mean(mae_val)) if isinstance(mae_val, (list, np.ndarray)) else float(mae_val)
        rmse_extended = float(np.mean(rmse_val)) if isinstance(rmse_val, (list, np.ndarray)) else float(rmse_val)
        mape_extended = float(np.mean(mape_val)) if isinstance(mape_val, (list, np.ndarray)) else float(mape_val)
        smape_extended = float(np.mean(smape_val)) if isinstance(smape_val, (list, np.ndarray)) else float(smape_val)
        
        all_results['ensemble'] = {
            'mae_standard': mae_standard,
            'rmse_standard': rmse_standard,
            'mape_standard': mape_standard,
            'smape_standard': smape_standard,
            'mae_extended': mae_extended,
            'rmse_extended': rmse_extended,
            'mape_extended': mape_extended,
            'smape_extended': smape_extended,
            'predictions_standard': ensemble_preds_standard[0] if ensemble_preds_standard else None,
            'predictions_extended': ensemble_preds_extended[0] if ensemble_preds_extended else None,
            'test_input': test_inputs_standard[0] if test_inputs_standard else None,
            'test_target_standard': test_targets_standard[0] if test_targets_standard else None,
            'test_target_extended': test_targets_extended[0] if test_targets_extended else None,
            'full_series': test_series[0] if test_series else None,
            'backtest_mae': float('nan'),  # Skip backtest for ensemble
            'backtest_rmse': float('nan'),
            'backtest_mape': float('nan'),
            'backtest_smape': float('nan'),
            'backtest_series': None,
            'backtest_forecasts': None,
        }
        
        logger.info(f"Ensemble MAE (Standard): {mae_standard:.4f}")
        logger.info(f"Ensemble MAE (Extended): {mae_extended:.4f}")
    
    # Create comparison table
    comparison_df = create_comparison_table(all_results)
    
    # Create comparison plots (split into two)
    create_metrics_plot(all_results, comparison_df)
    create_predictions_plot(all_results, extended_output_size)
    
    # Print summary
    best_model = comparison_df.iloc[0]['Model']
    best_mae = comparison_df.iloc[0]['MAE (Standard)']
    logger.info(f"\nEvaluation complete. Best model: {best_model} (MAE: {best_mae:.4f})")
    logger.info(f"Results saved to {EvaluationConfig.COMPARISON_TABLE_PATH}")


if __name__ == "__main__":
    main()