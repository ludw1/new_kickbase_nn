"""
Analyze the predictability of market value differences.
This script helps diagnose why the model isn't learning.
"""
import numpy as np
from transform_data import transform_data
import matplotlib.pyplot as plt
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_predictability():
    """Analyze if market value differences are predictable."""
    
    # Load data
    raw_data = transform_data("all_player_data.json")
    logger.info(f"Loaded {len(raw_data)} player time series")
    
    all_diffs = []
    autocorrelations = []
    
    for player_values in raw_data:
        if len(player_values) < 10:
            continue
            
        values = np.array(player_values)
        diffs = values[1:] - values[:-1]
        all_diffs.extend(diffs)
        
        # Calculate autocorrelation at lag 1
        if len(diffs) > 2:
            autocorr = np.corrcoef(diffs[:-1], diffs[1:])[0, 1]
            if not np.isnan(autocorr):
                autocorrelations.append(autocorr)
    
    all_diffs = np.array(all_diffs)
    autocorrelations = np.array(autocorrelations)
    
    # Statistics
    logger.info("\n" + "="*60)
    logger.info("DIFFERENCE STATISTICS")
    logger.info("="*60)
    logger.info(f"Total differences: {len(all_diffs)}")
    logger.info(f"Mean: {np.mean(all_diffs):.4f}")
    logger.info(f"Std: {np.std(all_diffs):.4f}")
    logger.info(f"Min: {np.min(all_diffs):.4f}")
    logger.info(f"Max: {np.max(all_diffs):.4f}")
    logger.info(f"Median: {np.median(all_diffs):.4f}")
    
    # Check if centered around zero
    logger.info(f"\nPercentage of zero changes: {(all_diffs == 0).sum() / len(all_diffs) * 100:.2f}%")
    logger.info(f"Percentage of positive changes: {(all_diffs > 0).sum() / len(all_diffs) * 100:.2f}%")
    logger.info(f"Percentage of negative changes: {(all_diffs < 0).sum() / len(all_diffs) * 100:.2f}%")
    
    # Autocorrelation analysis
    logger.info("\n" + "="*60)
    logger.info("AUTOCORRELATION ANALYSIS (Lag-1)")
    logger.info("="*60)
    logger.info(f"Mean autocorrelation: {np.mean(autocorrelations):.4f}")
    logger.info(f"Median autocorrelation: {np.median(autocorrelations):.4f}")
    logger.info(f"Std autocorrelation: {np.std(autocorrelations):.4f}")
    
    if abs(np.mean(autocorrelations)) < 0.1:
        logger.warning("⚠️  WARNING: Very low autocorrelation suggests differences are close to random walk!")
        logger.warning("⚠️  This means past differences have almost no predictive power for future differences.")
    
    # Stationarity test (Augmented Dickey-Fuller)
    from statsmodels.tsa.stattools import adfuller
    
    # Test a few long series
    p_values = []
    for player_values in raw_data[:50]:  # Test first 50 players
        if len(player_values) > 30:
            values = np.array(player_values)
            diffs = values[1:] - values[:-1]
            try:
                result = adfuller(diffs)
                p_values.append(result[1])
            except:
                pass
    
    logger.info("\n" + "="*60)
    logger.info("STATIONARITY TEST (Augmented Dickey-Fuller)")
    logger.info("="*60)
    logger.info(f"Mean p-value: {np.mean(p_values):.4f}")
    logger.info(f"% series that are stationary (p < 0.05): {(np.array(p_values) < 0.05).sum() / len(p_values) * 100:.1f}%")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution of differences
    axes[0, 0].hist(all_diffs, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0, 0].set_xlabel('Market Value Difference')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Market Value Differences')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Autocorrelation distribution
    axes[0, 1].hist(autocorrelations, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='No correlation')
    axes[0, 1].axvline(np.mean(autocorrelations), color='green', linestyle='--', linewidth=2, label=f'Mean = {np.mean(autocorrelations):.3f}')
    axes[0, 1].set_xlabel('Lag-1 Autocorrelation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Autocorrelations Across Players')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Example time series - raw values
    for i in range(min(5, len(raw_data))):
        if len(raw_data[i]) > 30:
            axes[1, 0].plot(raw_data[i], alpha=0.6, label=f'Player {i+1}')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Market Value')
    axes[1, 0].set_title('Example Raw Market Value Time Series')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Example time series - differences
    for i in range(min(5, len(raw_data))):
        if len(raw_data[i]) > 30:
            values = np.array(raw_data[i])
            diffs = values[1:] - values[:-1]
            axes[1, 1].plot(diffs, alpha=0.6, label=f'Player {i+1}')
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Market Value Difference')
    axes[1, 1].set_title('Example Market Value Difference Time Series')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/data_predictability_analysis.png', dpi=150)
    logger.info("\nSaved analysis plot to logs/data_predictability_analysis.png")
    plt.close()
    
    # RECOMMENDATION
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)
    
    if abs(np.mean(autocorrelations)) < 0.05:
        logger.info("❌ The differences appear to be nearly RANDOM (white noise).")
        logger.info("   Recommendation: Consider these alternatives:")
        logger.info("   1. Use PERCENTAGE changes instead: (new - old) / old")
        logger.info("   2. Predict ABSOLUTE values with temporal split (not player split)")
        logger.info("   3. Add EXTERNAL FEATURES (team performance, position, etc.)")
        logger.info("   4. Use LONGER input windows to capture seasonal patterns")
        logger.info("   5. Predict CUMULATIVE changes over longer horizons")
    elif abs(np.mean(autocorrelations)) < 0.15:
        logger.info("⚠️  The differences show WEAK but non-zero autocorrelation.")
        logger.info("   The current approach might work with:")
        logger.info("   1. Much longer input sequences (60-90 days)")
        logger.info("   2. Additional features beyond just past values")
        logger.info("   3. Ensemble methods")
    else:
        logger.info("✓ The differences show meaningful autocorrelation.")
        logger.info("  The model should be able to learn. Check:")
        logger.info("  1. Learning rate might be too low")
        logger.info("  2. Model might be too simple")
        logger.info("  3. Training data might be too limited")

if __name__ == "__main__":
    analyze_predictability()
