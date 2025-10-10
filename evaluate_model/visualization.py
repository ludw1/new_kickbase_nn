"""
Visualization functionality for model evaluation.

Contains functions for creating comparison plots, metrics visualizations,
and prediction graphs.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import EvaluationConfig
from typing import Dict, Any

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (14, 8)


def create_comparison_table(all_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create a comparison table from all model results.

    Args:
        all_results: Dictionary containing results from all models

    Returns:
        DataFrame with comparison metrics
    """
    logger.info("Creating comparison table...")

    data = []
    for model_name, results in all_results.items():
        data.append(
            {
                "Model": model_name.upper(),
                "MAE (Standard)": results["mae_standard"],
                "SMAPE (Standard)": results["smape_standard"],
                "MAE (Extended)": results["mae_extended"],
                "SMAPE (Extended)": results["smape_extended"],
                "MAE (Backtest)": results["backtest_mae"],
                "SMAPE (Backtest)": results["backtest_smape"],
                "MAE Degradation (%)": (
                    (results["mae_extended"] - results["mae_standard"])
                    / results["mae_standard"]
                    * 100
                ),
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("MAE (Standard)")

    os.makedirs(
        os.path.dirname(EvaluationConfig.COMPARISON_TABLE_PATH) or ".", exist_ok=True
    )
    df.to_csv(EvaluationConfig.COMPARISON_TABLE_PATH, index=False, float_format="%.4f")

    print("\n" + "=" * 120)
    print("MODEL COMPARISON TABLE")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120 + "\n")

    return df


def create_metrics_plot(
    all_results: Dict[str, Dict[str, Any]], comparison_df: pd.DataFrame
):
    """Create metrics comparison plot (MAE, SMAPE, Backtest).

    Args:
        all_results: Dictionary containing results from all models
        comparison_df: DataFrame with comparison metrics
    """
    logger.info("Creating metrics comparison plot...")

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    models = comparison_df["Model"].values
    x = np.arange(len(models))
    width = 0.25

    # MAE comparison (Standard, Extended, Backtest)
    ax1 = fig.add_subplot(gs[0, 0])
    mae_std = comparison_df["MAE (Standard)"].values
    mae_ext = comparison_df["MAE (Extended)"].values
    mae_bt = comparison_df["MAE (Backtest)"].values
    ax1.bar(x - width, mae_std, width, label="Standard", alpha=0.8)
    ax1.bar(x, mae_ext, width, label="Extended", alpha=0.8)
    ax1.bar(x + width, mae_bt, width, label="Backtest", alpha=0.8)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("MAE")
    ax1.set_title("MAE: Standard vs Extended vs Backtest")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # SMAPE comparison (Standard, Extended, Backtest)
    ax2 = fig.add_subplot(gs[0, 1])
    smape_std = comparison_df["SMAPE (Standard)"].values
    smape_ext = comparison_df["SMAPE (Extended)"].values
    smape_bt = comparison_df["SMAPE (Backtest)"].values
    ax2.bar(x - width, smape_std, width, label="Standard", alpha=0.8)
    ax2.bar(x, smape_ext, width, label="Extended", alpha=0.8)
    ax2.bar(x + width, smape_bt, width, label="Backtest", alpha=0.8)
    ax2.set_xlabel("Model")
    ax2.set_ylabel("SMAPE (%)")
    ax2.set_title("SMAPE: Standard vs Extended vs Backtest")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # MAE Degradation
    ax3 = fig.add_subplot(gs[0, 2:])
    mae_deg = comparison_df["MAE Degradation (%)"].values
    ax3.bar(x, mae_deg, width * 2, alpha=0.8, color="coral")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Degradation (%)")
    ax3.set_title("MAE Degradation (Extended vs Standard)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha="right")
    ax3.grid(axis="y", alpha=0.3)
    ax3.axhline(y=0, color="black", linestyle="--", linewidth=0.5)

    # Backtest visualization - OVERLAYED (all models in one plot)
    colors = {
        "nhits": "blue",
        "nlinear": "green",
        "tide": "red",
        "linear_regression": "purple",
        "ensemble": "orange",
    }

    ax_backtest = fig.add_subplot(gs[1, :])

    # Plot actual series once (from first model)
    first_model_results = list(all_results.values())[0]
    if first_model_results.get("backtest_series") is not None:
        first_model_results["backtest_series"].plot(
            ax=ax_backtest, label="Actual", color="black", linewidth=2, alpha=0.8
        )

    # Overlay all model forecasts
    for model_name, results in all_results.items():
        if results.get("backtest_forecasts") is not None:
            results["backtest_forecasts"].plot(
                ax=ax_backtest,
                label=f"{model_name.upper()}",
                color=colors.get(model_name, "gray"),
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
            )

    ax_backtest.set_title(
        "Backtest Comparison - All Models", fontsize=11, fontweight="bold"
    )
    ax_backtest.set_xlabel("Time", fontsize=9)
    ax_backtest.set_ylabel("Value", fontsize=9)
    ax_backtest.legend(loc="best", fontsize=9)
    ax_backtest.grid(True, alpha=0.3)

    # Save metrics plot
    metrics_path = EvaluationConfig.COMPARISON_PLOT_PATH.replace(".png", "_metrics.png")
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
    logger.info(f"Metrics plot saved to: {metrics_path}")
    plt.close()


def create_predictions_plot(
    all_results: Dict[str, Dict[str, Any]], extended_output_size: int
):
    """Create predictions comparison plot (all models combined in one plot, zoomed to prediction window).

    Args:
        all_results: Dictionary containing results from all models
        extended_output_size: Extended output size for predictions
    """
    logger.info("Creating predictions comparison plot...")

    colors = {
        "nhits": "blue",
        "nlinear": "green",
        "tide": "red",
        "linear_regression": "purple",
        "ensemble": "orange",
        "arima": "brown",
    }

    # Create single figure with all predictions combined
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)

    # Number of days to show before predictions
    lookback_days = 20

    # Get the first valid result to extract the full series
    first_result = None
    for results in all_results.values():
        if (
            results["full_series"] is not None
            and results["predictions_extended"] is not None
        ):
            first_result = results
            break

    if first_result is None:
        logger.warning("No valid results found for predictions plot")
        return

    full_series = first_result["full_series"]

    # Get the actual future values (ground truth)
    actual_future = full_series[-extended_output_size:]

    # Get a few days of historical data before predictions
    historical_data = full_series[:-extended_output_size]
    if len(historical_data) > lookback_days:
        historical_context = historical_data[-lookback_days:]
    else:
        historical_context = historical_data

    historical_context.plot(
        ax=ax, label="Historical", color="black", linewidth=2.5, alpha=0.7
    )

    actual_future.plot(
        ax=ax, label="Actual", color="black", linestyle="--", linewidth=2.5, alpha=0.9
    )

    # Plot each model's predictions
    for model_name, results in all_results.items():
        if results["predictions_extended"] is not None:
            results["predictions_extended"].plot(
                ax=ax,
                label=f"{model_name.upper()}",
                color=colors.get(model_name, "gray"),
                linewidth=2,
                alpha=0.8,
                linestyle="-",
            )

    # Add vertical line to show prediction start
    if hasattr(actual_future, "start_time"):
        ax.axvline(
            x=actual_future.start_time(),
            color="gray",
            linestyle=":",
            linewidth=2,
            alpha=0.6,
            label="Prediction Start",
        )

    ax.set_title(
        f"Model Predictions Comparison - {extended_output_size} Day Forecast (Zoomed)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    # Save predictions plot
    predictions_path = EvaluationConfig.COMPARISON_PLOT_PATH.replace(
        ".png", "_predictions.png"
    )
    os.makedirs(os.path.dirname(predictions_path) or ".", exist_ok=True)
    plt.savefig(predictions_path, dpi=300, bbox_inches="tight")
    logger.info(f"Predictions plot saved to: {predictions_path}")
    plt.close()
