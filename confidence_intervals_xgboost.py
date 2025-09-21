#!/usr/bin/env python3
"""
Confidence Intervals Visualization on Imbalanced Data - XGBoost Version

This simulation demonstrates how bootstrap confidence intervals behave with
XGBoost models when dealing with asymmetrically distributed data points.
Compares XGBoost vs polynomial regression approaches for uncertainty quantification.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
BOOTSTRAP_ITERATIONS = 5000
CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL
POLYNOMIAL_DEGREE = 2
FIGURE_SIZE = (12, 8)
DPI = 100

# XGBoost hyperparameters
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'random_state': 42,
    'verbosity': 0
}

# Color scheme
COLOR_DATA = 'black'
COLOR_FIT = 'blue'
COLOR_POLYNOMIAL = 'green'
COLOR_XGBOOST = 'red'
COLOR_OUT_OF_SAMPLE = 'purple'


def generate_asymmetric_dataset():
    """
    Generate an asymmetric dataset with dense and sparse regions.
    
    Part 1: Generate the Asymmetrical Dataset
    - Dense: 100 points in [-1, 0]
    - Sparse: 10 points in [0, 1]
    - Function: y = x^2 + noise
    
    Returns:
        tuple: (X, y) where X is features and y is targets
    """
    # Generate dense region: 100 points in [-1, 0]
    x_dense = np.linspace(-1, 0, 100)
    noise_dense = np.random.normal(0, 1, 100)
    y_dense = x_dense**2 + noise_dense
    
    # Generate sparse region: 10 points in [0, 1]
    x_sparse = np.linspace(0, 1, 10)
    noise_sparse = np.random.normal(0, 1, 10)
    y_sparse = x_sparse**2 + noise_sparse
    
    # Combine datasets
    X = np.concatenate([x_dense, x_sparse])
    y = np.concatenate([y_dense, y_sparse])
    
    print(f"Dataset created: {len(X)} points total")
    print(f"  - Dense region [-1, 0]: {len(x_dense)} points")
    print(f"  - Sparse region [0, 1]: {len(x_sparse)} points")
    
    return X.reshape(-1, 1), y


def bootstrap_xgboost_ci(X, y, n_iterations=BOOTSTRAP_ITERATIONS):
    """
    Calculate confidence intervals using bootstrap method with XGBoost.
    
    Part 2A: XGBoost Bootstrap Method
    - Resample data with replacement
    - Fit XGBoost regressor to each sample
    - Calculate percentile-based confidence intervals
    - Track feature importance and performance metrics
    
    Args:
        X: Feature array
        y: Target array
        n_iterations: Number of bootstrap iterations
        
    Returns:
        dict: Contains x_pred, y_pred, lower_bound, upper_bound, metrics, importance_history
    """
    print(f"\nCalculating XGBoost bootstrap confidence intervals ({n_iterations} iterations)...")
    
    # Create prediction points
    x_pred = np.linspace(X.min(), X.max(), 300)
    X_pred = x_pred.reshape(-1, 1)
    
    # Store predictions and metrics from each bootstrap iteration
    bootstrap_predictions = []
    feature_importance_history = []
    performance_metrics = {'rmse': [], 'r2': []}
    
    # Bootstrap iterations
    for i in range(n_iterations):
        if i % 1000 == 0:
            print(f"  Bootstrap iteration {i}/{n_iterations}")
        
        # Resample with replacement
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
        
        # Fit XGBoost model
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_sample, y_sample)
        
        # Make predictions
        y_pred = model.predict(X_pred)
        bootstrap_predictions.append(y_pred)
        
        # Track feature importance (for single feature, this is just the importance value)
        feature_importance_history.append(model.feature_importances_[0])
        
        # Calculate performance metrics on bootstrap sample
        y_sample_pred = model.predict(X_sample)
        rmse = np.sqrt(mean_squared_error(y_sample, y_sample_pred))
        r2 = r2_score(y_sample, y_sample_pred)
        performance_metrics['rmse'].append(rmse)
        performance_metrics['r2'].append(r2)
    
    # Convert to array for easier manipulation
    bootstrap_predictions = np.array(bootstrap_predictions)
    
    # Calculate percentiles for confidence intervals
    lower_percentile = (ALPHA / 2) * 100
    upper_percentile = (1 - ALPHA / 2) * 100
    
    lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
    
    # Get mean prediction
    y_pred_mean = np.mean(bootstrap_predictions, axis=0)
    
    print(f"  XGBoost bootstrap CI calculation complete")
    print(f"  Mean RMSE: {np.mean(performance_metrics['rmse']):.4f} ± {np.std(performance_metrics['rmse']):.4f}")
    print(f"  Mean R²: {np.mean(performance_metrics['r2']):.4f} ± {np.std(performance_metrics['r2']):.4f}")
    print(f"  Feature importance: {np.mean(feature_importance_history):.4f} ± {np.std(feature_importance_history):.4f}")
    
    return {
        'x_pred': x_pred,
        'y_pred': y_pred_mean,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'performance_metrics': performance_metrics,
        'importance_history': feature_importance_history,
        'predictions_history': bootstrap_predictions
    }


def bootstrap_polynomial_ci(X, y, n_iterations=BOOTSTRAP_ITERATIONS):
    """
    Calculate confidence intervals using bootstrap method with polynomial regression.
    (For comparison with XGBoost)
    
    Args:
        X: Feature array
        y: Target array
        n_iterations: Number of bootstrap iterations
        
    Returns:
        dict: Contains x_pred, y_pred, lower_bound, upper_bound, metrics
    """
    print(f"\nCalculating polynomial bootstrap confidence intervals ({n_iterations} iterations)...")
    
    # Create prediction points
    x_pred = np.linspace(X.min(), X.max(), 300)
    X_pred = x_pred.reshape(-1, 1)
    
    # Store predictions and metrics from each bootstrap iteration
    bootstrap_predictions = []
    performance_metrics = {'rmse': [], 'r2': []}
    
    # Create polynomial regression model
    model = make_pipeline(
        PolynomialFeatures(degree=POLYNOMIAL_DEGREE),
        LinearRegression()
    )
    
    # Bootstrap iterations
    for i in range(n_iterations):
        if i % 1000 == 0:
            print(f"  Bootstrap iteration {i}/{n_iterations}")
        
        # Resample with replacement
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
        
        # Fit model and predict
        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_pred)
        bootstrap_predictions.append(y_pred)
        
        # Calculate performance metrics on bootstrap sample
        y_sample_pred = model.predict(X_sample)
        rmse = np.sqrt(mean_squared_error(y_sample, y_sample_pred))
        r2 = r2_score(y_sample, y_sample_pred)
        performance_metrics['rmse'].append(rmse)
        performance_metrics['r2'].append(r2)
    
    # Convert to array for easier manipulation
    bootstrap_predictions = np.array(bootstrap_predictions)
    
    # Calculate percentiles for confidence intervals
    lower_percentile = (ALPHA / 2) * 100
    upper_percentile = (1 - ALPHA / 2) * 100
    
    lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
    
    # Get mean prediction
    y_pred_mean = np.mean(bootstrap_predictions, axis=0)
    
    print(f"  Polynomial bootstrap CI calculation complete")
    print(f"  Mean RMSE: {np.mean(performance_metrics['rmse']):.4f} ± {np.std(performance_metrics['rmse']):.4f}")
    print(f"  Mean R²: {np.mean(performance_metrics['r2']):.4f} ± {np.std(performance_metrics['r2']):.4f}")
    
    return {
        'x_pred': x_pred,
        'y_pred': y_pred_mean,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'performance_metrics': performance_metrics,
        'predictions_history': bootstrap_predictions
    }


def xgboost_quantile_regression(X, y, quantiles=[0.025, 0.975]):
    """
    Alternative confidence interval estimation using XGBoost quantile regression.
    
    Args:
        X: Feature array
        y: Target array
        quantiles: List of quantiles to predict
        
    Returns:
        dict: Contains x_pred, quantile predictions
    """
    print(f"\nCalculating XGBoost quantile regression intervals...")
    
    # Create prediction points
    x_pred = np.linspace(X.min(), X.max(), 300)
    X_pred = x_pred.reshape(-1, 1)
    
    quantile_predictions = {}
    
    for q in quantiles:
        # Create XGBoost quantile regressor
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            **{k: v for k, v in XGB_PARAMS.items() if k != 'random_state'},
            random_state=42
        )
        
        # Fit and predict
        model.fit(X, y)
        y_pred = model.predict(X_pred)
        quantile_predictions[q] = y_pred
        
        print(f"  Quantile {q:.3f} regression complete")
    
    return {
        'x_pred': x_pred,
        'quantile_predictions': quantile_predictions
    }


def plot_xgboost_ci(X, y, xgb_results, title_suffix=""):
    """
    Plot XGBoost confidence intervals
    
    Args:
        X, y: Original data
        xgb_results: Results from bootstrap_xgboost_ci
        title_suffix: Additional text for title
    """
    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Extract results
    x_pred = xgb_results['x_pred']
    y_pred = xgb_results['y_pred']
    lower = xgb_results['lower_bound']
    upper = xgb_results['upper_bound']
    
    # Plot data points
    plt.scatter(X, y, color=COLOR_DATA, alpha=0.5, s=20, label='Data points')
    
    # Plot fitted curve
    plt.plot(x_pred, y_pred, color=COLOR_FIT, linewidth=2, label='XGBoost fitted curve')
    
    # Plot confidence interval
    plt.fill_between(x_pred, lower, upper, color=COLOR_XGBOOST, alpha=0.3, 
                     label='95% XGBoost Bootstrap CI')
    plt.plot(x_pred, lower, color=COLOR_XGBOOST, linestyle='--', alpha=0.5, linewidth=1)
    plt.plot(x_pred, upper, color=COLOR_XGBOOST, linestyle='--', alpha=0.5, linewidth=1)
    
    # Add vertical lines to show region boundaries
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.text(-0.5, plt.ylim()[1]*0.9, 'Dense region\n(100 points)', 
             ha='center', fontsize=10, color='gray')
    plt.text(0.5, plt.ylim()[1]*0.9, 'Sparse region\n(10 points)', 
             ha='center', fontsize=10, color='gray')
    
    # Add performance metrics
    metrics = xgb_results['performance_metrics']
    perf_text = (f"XGBoost Performance:\n"
                f"RMSE: {np.mean(metrics['rmse']):.3f} ± {np.std(metrics['rmse']):.3f}\n"
                f"R²: {np.mean(metrics['r2']):.3f} ± {np.std(metrics['r2']):.3f}")
    
    plt.text(0.02, 0.98, perf_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(f'XGBoost Bootstrap Confidence Intervals{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_model_comparison(X, y, poly_results, xgb_results):
    """
    Plot comparison between polynomial and XGBoost confidence intervals
    
    Args:
        X, y: Original data
        poly_results: Results from bootstrap_polynomial_ci
        xgb_results: Results from bootstrap_xgboost_ci
    """
    plt.figure(figsize=(16, 8), dpi=DPI)
    
    # Plot data points
    plt.scatter(X, y, color=COLOR_DATA, alpha=0.6, s=30, label='Data points', zorder=5)
    
    # Plot polynomial results
    plt.plot(poly_results['x_pred'], poly_results['y_pred'], color=COLOR_POLYNOMIAL, 
             linewidth=2.5, label='Polynomial fit', zorder=4)
    plt.fill_between(poly_results['x_pred'], poly_results['lower_bound'], poly_results['upper_bound'], 
                     color=COLOR_POLYNOMIAL, alpha=0.2, label='Polynomial Bootstrap CI')
    plt.plot(poly_results['x_pred'], poly_results['lower_bound'], color=COLOR_POLYNOMIAL, 
             linestyle='--', alpha=0.7, linewidth=1.5)
    plt.plot(poly_results['x_pred'], poly_results['upper_bound'], color=COLOR_POLYNOMIAL, 
             linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Plot XGBoost results
    plt.plot(xgb_results['x_pred'], xgb_results['y_pred'], color=COLOR_XGBOOST, 
             linewidth=2.5, label='XGBoost fit', zorder=4)
    plt.fill_between(xgb_results['x_pred'], xgb_results['lower_bound'], xgb_results['upper_bound'], 
                     color=COLOR_XGBOOST, alpha=0.2, label='XGBoost Bootstrap CI')
    plt.plot(xgb_results['x_pred'], xgb_results['lower_bound'], color=COLOR_XGBOOST, 
             linestyle='-.', alpha=0.7, linewidth=1.5)
    plt.plot(xgb_results['x_pred'], xgb_results['upper_bound'], color=COLOR_XGBOOST, 
             linestyle='-.', alpha=0.7, linewidth=1.5)
    
    # Add vertical line and region labels
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Add text annotations
    y_max = max(xgb_results['upper_bound'].max(), poly_results['upper_bound'].max())
    plt.text(-0.5, y_max*0.85, 'Dense region\n(100 points)', 
             ha='center', fontsize=11, color='gray', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.5, y_max*0.85, 'Sparse region\n(10 points)', 
             ha='center', fontsize=11, color='gray',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('X', fontsize=13)
    plt.ylabel('Y', fontsize=13)
    plt.title('Comparison: Polynomial vs XGBoost Bootstrap Confidence Intervals', 
              fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add comparison metrics
    dense_mask = poly_results['x_pred'] <= 0
    sparse_mask = poly_results['x_pred'] > 0
    
    poly_width_dense = np.mean(poly_results['upper_bound'][dense_mask] - poly_results['lower_bound'][dense_mask])
    poly_width_sparse = np.mean(poly_results['upper_bound'][sparse_mask] - poly_results['lower_bound'][sparse_mask])
    xgb_width_dense = np.mean(xgb_results['upper_bound'][dense_mask] - xgb_results['lower_bound'][dense_mask])
    xgb_width_sparse = np.mean(xgb_results['upper_bound'][sparse_mask] - xgb_results['lower_bound'][sparse_mask])
    
    poly_metrics = poly_results['performance_metrics']
    xgb_metrics = xgb_results['performance_metrics']
    
    info_text = (f"Average CI Width:\n"
                f"Polynomial - Dense: {poly_width_dense:.3f}, Sparse: {poly_width_sparse:.3f}\n"
                f"XGBoost - Dense: {xgb_width_dense:.3f}, Sparse: {xgb_width_sparse:.3f}\n\n"
                f"Mean Performance (RMSE):\n"
                f"Polynomial: {np.mean(poly_metrics['rmse']):.3f} ± {np.std(poly_metrics['rmse']):.3f}\n"
                f"XGBoost: {np.mean(xgb_metrics['rmse']):.3f} ± {np.std(xgb_metrics['rmse']):.3f}")
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return plt.gcf()


def plot_performance_comparison(poly_results, xgb_results):
    """
    Plot performance metrics comparison between models
    
    Args:
        poly_results: Results from bootstrap_polynomial_ci
        xgb_results: Results from bootstrap_xgboost_ci
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=DPI)
    
    # RMSE comparison
    ax1.hist(poly_results['performance_metrics']['rmse'], bins=50, alpha=0.7, 
             color=COLOR_POLYNOMIAL, label='Polynomial', density=True)
    ax1.hist(xgb_results['performance_metrics']['rmse'], bins=50, alpha=0.7, 
             color=COLOR_XGBOOST, label='XGBoost', density=True)
    ax1.set_xlabel('RMSE')
    ax1.set_ylabel('Density')
    ax1.set_title('Bootstrap RMSE Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R² comparison
    ax2.hist(poly_results['performance_metrics']['r2'], bins=50, alpha=0.7, 
             color=COLOR_POLYNOMIAL, label='Polynomial', density=True)
    ax2.hist(xgb_results['performance_metrics']['r2'], bins=50, alpha=0.7, 
             color=COLOR_XGBOOST, label='XGBoost', density=True)
    ax2.set_xlabel('R²')
    ax2.set_ylabel('Density')
    ax2.set_title('Bootstrap R² Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Model Performance Comparison Across Bootstrap Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def analyze_feature_importance(importance_history):
    """
    Analyze XGBoost feature importance distribution across bootstrap samples
    
    Args:
        importance_history: List of feature importance values from bootstrap
    """
    plt.figure(figsize=(10, 6), dpi=DPI)
    
    plt.hist(importance_history, bins=50, color=COLOR_XGBOOST, alpha=0.7, density=True)
    plt.axvline(np.mean(importance_history), color='black', linestyle='--', 
                label=f'Mean: {np.mean(importance_history):.4f}')
    plt.axvline(np.percentile(importance_history, 2.5), color='red', linestyle=':', 
                label=f'95% CI: [{np.percentile(importance_history, 2.5):.4f}, {np.percentile(importance_history, 97.5):.4f}]')
    plt.axvline(np.percentile(importance_history, 97.5), color='red', linestyle=':')
    
    plt.xlabel('Feature Importance')
    plt.ylabel('Density')
    plt.title('XGBoost Feature Importance Distribution Across Bootstrap Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def out_of_sample_comparison(X_train, y_train):
    """
    Out-of-sample test comparing polynomial and XGBoost models
    
    Args:
        X_train, y_train: Training data used to fit the models
        
    Returns:
        matplotlib figure
    """
    print("\n" + "="*60)
    print("Out-of-Sample Comparison: Polynomial vs XGBoost")
    print("="*60)
    
    # Train both models on original data
    poly_model = make_pipeline(
        PolynomialFeatures(degree=POLYNOMIAL_DEGREE),
        LinearRegression()
    )
    poly_model.fit(X_train, y_train)
    
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    
    # Generate new data from a linear function
    np.random.seed(123)  # Different seed for variety
    x_test = np.linspace(-1, 1, 20)
    noise_test = np.random.normal(0, 0.5, 20)  # Smaller noise for clarity
    y_test_true = x_test + noise_test  # Linear function: y = x + noise
    X_test = x_test.reshape(-1, 1)
    
    # Make predictions with both models
    y_test_poly = poly_model.predict(X_test)
    y_test_xgb = xgb_model.predict(X_test)
    
    # Calculate metrics for both models
    poly_mse = mean_squared_error(y_test_true, y_test_poly)
    poly_r2 = r2_score(y_test_true, y_test_poly)
    xgb_mse = mean_squared_error(y_test_true, y_test_xgb)
    xgb_r2 = r2_score(y_test_true, y_test_xgb)
    
    print(f"Out-of-sample performance on linear data (y = x + noise):")
    print(f"  Polynomial - MSE: {poly_mse:.4f}, R²: {poly_r2:.4f}")
    print(f"  XGBoost - MSE: {xgb_mse:.4f}, R²: {xgb_r2:.4f}")
    
    # Create visualization
    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Plot test data and true function
    x_smooth = np.linspace(-1, 1, 100)
    plt.plot(x_smooth, x_smooth, 'g--', linewidth=2, 
             label='True function (y = x)', alpha=0.7)
    plt.scatter(x_test, y_test_true, color=COLOR_OUT_OF_SAMPLE, 
               s=100, label='Test data (linear)', zorder=5, edgecolors='black')
    
    # Plot model predictions
    X_smooth = x_smooth.reshape(-1, 1)
    y_poly_pred = poly_model.predict(X_smooth)
    y_xgb_pred = xgb_model.predict(X_smooth)
    
    plt.plot(x_smooth, y_poly_pred, color=COLOR_POLYNOMIAL, linewidth=2, 
             label='Polynomial prediction (quadratic)')
    plt.plot(x_smooth, y_xgb_pred, color=COLOR_XGBOOST, linewidth=2, 
             label='XGBoost prediction')
    
    # Plot prediction points
    plt.scatter(x_test, y_test_poly, color=COLOR_POLYNOMIAL, s=50, 
               label='Polynomial predictions', marker='s', zorder=4)
    plt.scatter(x_test, y_test_xgb, color=COLOR_XGBOOST, s=50, 
               label='XGBoost predictions', marker='^', zorder=4)
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Out-of-Sample Test: Model Generalization to Linear Data', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add performance annotation
    perf_text = (f"Performance Metrics:\n"
                f"Polynomial: MSE = {poly_mse:.3f}, R² = {poly_r2:.3f}\n"
                f"XGBoost: MSE = {xgb_mse:.3f}, R² = {xgb_r2:.3f}")
    plt.text(0.02, 0.98, perf_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    
    return plt.gcf()


def main():
    """
    Main execution function to run the complete XGBoost simulation
    """
    print("="*60)
    print("CONFIDENCE INTERVALS ON IMBALANCED DATA - XGBOOST VERSION")
    print("="*60)
    
    # Part 1: Generate asymmetric dataset
    print("\n" + "="*60)
    print("Part 1: Generating Asymmetric Dataset")
    print("="*60)
    X, y = generate_asymmetric_dataset()
    
    # Part 2A: XGBoost bootstrap confidence intervals
    print("\n" + "="*60)
    print("Part 2A: XGBoost Bootstrap Confidence Intervals")
    print("="*60)
    xgb_results = bootstrap_xgboost_ci(X, y)
    
    # Part 2B: Polynomial bootstrap confidence intervals (for comparison)
    print("\n" + "="*60)
    print("Part 2B: Polynomial Bootstrap Confidence Intervals")
    print("="*60)
    poly_results = bootstrap_polynomial_ci(X, y)
    
    # Part 3: Create visualizations
    print("\n" + "="*60)
    print("Part 3: Creating Visualizations")
    print("="*60)
    
    # Plot 1: XGBoost CI
    fig1 = plot_xgboost_ci(X, y, xgb_results, " on Imbalanced Data")
    fig1.savefig('plot1_xgboost_ci.png', dpi=150, bbox_inches='tight')
    print("  Saved: plot1_xgboost_ci.png")
    
    # Plot 2: Model comparison
    fig2 = plot_model_comparison(X, y, poly_results, xgb_results)
    fig2.savefig('plot2_polynomial_vs_xgboost.png', dpi=150, bbox_inches='tight')
    print("  Saved: plot2_polynomial_vs_xgboost.png")
    
    # Plot 3: Performance comparison
    fig3 = plot_performance_comparison(poly_results, xgb_results)
    fig3.savefig('plot3_performance_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: plot3_performance_comparison.png")
    
    # Plot 4: Feature importance analysis
    fig4 = analyze_feature_importance(xgb_results['importance_history'])
    fig4.savefig('plot4_feature_importance.png', dpi=150, bbox_inches='tight')
    print("  Saved: plot4_feature_importance.png")
    
    # Plot 5: Out-of-sample comparison
    fig5 = out_of_sample_comparison(X, y)
    fig5.savefig('plot5_out_of_sample_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: plot5_out_of_sample_comparison.png")
    
    # Optional: XGBoost quantile regression (if time permits)
    print("\n" + "="*60)
    print("Optional: XGBoost Quantile Regression")
    print("="*60)
    try:
        quantile_results = xgboost_quantile_regression(X, y)
        print("  XGBoost quantile regression completed successfully")
        
        # Quick plot of quantile results
        plt.figure(figsize=(12, 8))
        plt.scatter(X, y, color=COLOR_DATA, alpha=0.5, s=20, label='Data points')
        plt.plot(quantile_results['x_pred'], quantile_results['quantile_predictions'][0.025], 
                'r--', label='2.5% quantile')
        plt.plot(quantile_results['x_pred'], quantile_results['quantile_predictions'][0.975], 
                'r--', label='97.5% quantile')
        plt.fill_between(quantile_results['x_pred'], 
                        quantile_results['quantile_predictions'][0.025],
                        quantile_results['quantile_predictions'][0.975], 
                        color=COLOR_XGBOOST, alpha=0.2, label='XGBoost Quantile CI')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('XGBoost Quantile Regression Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plot6_xgboost_quantile.png', dpi=150, bbox_inches='tight')
        print("  Saved: plot6_xgboost_quantile.png")
    except Exception as e:
        print(f"  Warning: XGBoost quantile regression failed: {e}")
        print("  Continuing without quantile regression...")
    
    # Display all plots
    plt.show()
    
    print("\n" + "="*60)
    print("XGBOOST SIMULATION COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("1. XGBoost bootstrap CI captures model uncertainty through ensemble variance")
    print("2. Tree-based models may show different uncertainty patterns vs polynomial")
    print("3. Both methods show increased uncertainty in the sparse region [0, 1]")
    print("4. XGBoost may better capture non-linear patterns but lacks analytical CI")
    print("5. Feature importance analysis provides additional model interpretability")
    print("\nAll plots have been saved to the current directory.")
    
    # Print final comparison summary
    poly_metrics = poly_results['performance_metrics']
    xgb_metrics = xgb_results['performance_metrics']
    
    print(f"\nFinal Performance Summary:")
    print(f"Polynomial Bootstrap - RMSE: {np.mean(poly_metrics['rmse']):.4f}, R²: {np.mean(poly_metrics['r2']):.4f}")
    print(f"XGBoost Bootstrap - RMSE: {np.mean(xgb_metrics['rmse']):.4f}, R²: {np.mean(xgb_metrics['r2']):.4f}")


if __name__ == "__main__":
    main()