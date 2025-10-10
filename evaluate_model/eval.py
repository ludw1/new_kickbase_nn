"""
Model evaluation and backtesting module.

Main orchestration module that coordinates model evaluation, backtesting,
and performance analysis. Evaluates all models together and produces comparison
tables and graphs.
"""

import os
import logging
from config import EvaluationConfig
from train_model.utils import setup_directories
from train_model.data_processing import load_and_preprocess_data

from .model_loading import load_model_from_checkpoint
from .metrics import evaluate_single_model
from .ensemble import create_simple_ensemble_predictions, calculate_ensemble_metrics
from .visualization import create_comparison_table, create_metrics_plot, create_predictions_plot

logger = logging.getLogger(__name__)


def evaluate_models():
    """Main evaluation function - loads and evaluates all models."""
    setup_directories()
    
    logger.info("Starting comprehensive model evaluation...")
    logger.info(f"Prediction length multiplier: {EvaluationConfig.PREDICTION_LENGTH_MULTIPLIER}x")
    tide_encoder_path = None
    # Check which models are available
    available_models = {}
    for model_name, model_path in EvaluationConfig.MODEL_PATHS.items():
        if os.path.exists(model_path):
            if "encoder" in model_path:
                tide_encoder_path = model_path
            else:
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
            model, input_size, output_size = load_model_from_checkpoint(model_name, model_path, tide_encoder_path)
            models_dict[model_name] = {
                'model': model,
                'input_size': input_size,
                'output_size': output_size
            }
        except Exception as e:
            logger.exception(f"Failed to load {model_name} model: {e}")
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
        output_size=extended_output_size
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
            test_inputs_standard = []
            test_inputs_extended = []
            
            for i, series in enumerate(test_series):
                if len(series) >= model_info['input_size'] + model_info['output_size']:
                    input_series = series[:-model_info['output_size']]
                    if model_name in ["tide", "nlinear", "linear_regression"]:
                        input_series = input_series.with_static_covariates(test_static_cov[i])
                    test_inputs_standard.append(input_series)
                
                if len(series) >= model_info['input_size'] + extended_output_size:
                    input_series_ext = series[:-extended_output_size]
                    if model_name in ["tide", "nlinear", "linear_regression"]:
                        input_series_ext = input_series_ext.with_static_covariates(test_static_cov[i])
                    test_inputs_extended.append(input_series_ext)
            
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
    logger.info("\\nCreating simple ensemble (average of all models)...")
    
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
    
    # Calculate ensemble metrics using the ensemble module
    ensemble_results = calculate_ensemble_metrics(
        ensemble_preds_standard,
        ensemble_preds_extended,
        test_targets_standard,
        test_targets_extended,
        test_inputs_standard,
        test_series
    )
    
    if ensemble_results:
        all_results['ensemble'] = ensemble_results
    
    # Create comparison table
    comparison_df = create_comparison_table(all_results)
    
    # Create comparison plots
    create_metrics_plot(all_results, comparison_df)
    create_predictions_plot(all_results, extended_output_size)
    
    # Print summary
    best_model = comparison_df.iloc[0]['Model']
    best_mae = comparison_df.iloc[0]['MAE (Standard)']
    logger.info(f"\\nEvaluation complete. Best model: {best_model} (MAE: {best_mae:.4f})")
    logger.info(f"Results saved to {EvaluationConfig.COMPARISON_TABLE_PATH}")


