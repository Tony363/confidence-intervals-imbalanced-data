#!/usr/bin/env python3
"""
Confidence Intervals Visualization on Imbalanced Data

This simulation demonstrates how bootstrap and analytical confidence intervals
behave differently when dealing with asymmetrically distributed data points.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
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

# Color scheme
COLOR_DATA = 'black'
COLOR_FIT = 'blue'
COLOR_BOOTSTRAP = 'red'
COLOR_ANALYTICAL = 'green'
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


def bootstrap_confidence_interval(X, y, n_iterations=BOOTSTRAP_ITERATIONS):
    """
    Calculate confidence intervals using bootstrap method.
    
    Part 2A: Bootstrap Method
    - Resample data with replacement
    - Fit polynomial regression to each sample
    - Calculate percentile-based confidence intervals
    
    Args:
        X: Feature array
        y: Target array
        n_iterations: Number of bootstrap iterations
        
    Returns:
        tuple: (x_pred, y_pred, lower_bound, upper_bound)
    """
    print(f"\nCalculating bootstrap confidence intervals ({n_iterations} iterations)...")
    
    # Create prediction points
    x_pred = np.linspace(X.min(), X.max(), 300)
    X_pred = x_pred.reshape(-1, 1)
    
    # Store predictions from each bootstrap iteration
    bootstrap_predictions = []
    
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
    
    # Convert to array for easier manipulation
    bootstrap_predictions = np.array(bootstrap_predictions)
    
    # Calculate percentiles for confidence intervals
    lower_percentile = (ALPHA / 2) * 100
    upper_percentile = (1 - ALPHA / 2) * 100
    
    lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
    
    # Get mean prediction
    y_pred_mean = np.mean(bootstrap_predictions, axis=0)
    
    print(f"  Bootstrap CI calculation complete")
    
    return x_pred, y_pred_mean, lower_bound, upper_bound


def analytical_confidence_interval(X, y):
    """
    Calculate confidence intervals using analytical regression method.
    
    Part 2B: Analytical Method
    - Fit single polynomial regression model
    - Calculate standard errors
    - Use t-distribution for confidence intervals
    
    Args:
        X: Feature array
        y: Target array
        
    Returns:
        tuple: (x_pred, y_pred, lower_bound, upper_bound)
    """
    print("\nCalculating analytical confidence intervals...")
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=POLYNOMIAL_DEGREE)
    X_poly = poly.fit_transform(X)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Create prediction points
    x_pred = np.linspace(X.min(), X.max(), 300)
    X_pred = x_pred.reshape(-1, 1)
    X_pred_poly = poly.transform(X_pred)
    
    # Make predictions
    y_pred = model.predict(X_pred_poly)
    
    # Calculate residuals and MSE
    y_train_pred = model.predict(X_poly)
    residuals = y - y_train_pred
    n = len(X)
    p = X_poly.shape[1]  # Number of parameters
    dof = n - p  # Degrees of freedom
    mse = np.sum(residuals**2) / dof
    
    # Calculate standard errors for predictions
    # SE = sqrt(MSE * (1 + x'(X'X)^-1x))
    XtX_inv = np.linalg.inv(X_poly.T @ X_poly)
    
    se_pred = []
    for x_p in X_pred_poly:
        leverage = x_p @ XtX_inv @ x_p.T
        se = np.sqrt(mse * (1 + leverage))
        se_pred.append(se)
    se_pred = np.array(se_pred)
    
    # Calculate t-value for confidence interval
    t_value = stats.t.ppf(1 - ALPHA/2, dof)
    
    # Calculate confidence intervals
    lower_bound = y_pred - t_value * se_pred
    upper_bound = y_pred + t_value * se_pred
    
    print(f"  Analytical CI calculation complete")
    print(f"  Degrees of freedom: {dof}")
    print(f"  MSE: {mse:.4f}")
    print(f"  t-value: {t_value:.4f}")
    
    return x_pred, y_pred, lower_bound, upper_bound


def plot_bootstrap_ci(X, y, x_pred, y_pred, lower, upper):
    """
    Plot #1: Bootstrap confidence intervals
    
    Args:
        X, y: Original data
        x_pred, y_pred: Prediction points and values
        lower, upper: Confidence interval bounds
    """
    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Plot data points
    plt.scatter(X, y, color=COLOR_DATA, alpha=0.5, s=20, label='Data points')
    
    # Plot fitted curve
    plt.plot(x_pred, y_pred, color=COLOR_FIT, linewidth=2, label='Fitted curve')
    
    # Plot confidence interval
    plt.fill_between(x_pred, lower, upper, color=COLOR_BOOTSTRAP, alpha=0.3, 
                     label='95% Bootstrap CI')
    plt.plot(x_pred, lower, color=COLOR_BOOTSTRAP, linestyle='--', alpha=0.5, linewidth=1)
    plt.plot(x_pred, upper, color=COLOR_BOOTSTRAP, linestyle='--', alpha=0.5, linewidth=1)
    
    # Add vertical lines to show region boundaries
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.text(-0.5, plt.ylim()[1]*0.9, 'Dense region\n(100 points)', 
             ha='center', fontsize=10, color='gray')
    plt.text(0.5, plt.ylim()[1]*0.9, 'Sparse region\n(10 points)', 
             ha='center', fontsize=10, color='gray')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Plot #1: Bootstrap Confidence Intervals on Imbalanced Data', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_analytical_ci(X, y, x_pred, y_pred, lower, upper):
    """
    Plot #2: Analytical confidence intervals
    
    Args:
        X, y: Original data
        x_pred, y_pred: Prediction points and values
        lower, upper: Confidence interval bounds
    """
    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Plot data points
    plt.scatter(X, y, color=COLOR_DATA, alpha=0.5, s=20, label='Data points')
    
    # Plot fitted curve
    plt.plot(x_pred, y_pred, color=COLOR_FIT, linewidth=2, label='Fitted curve')
    
    # Plot confidence interval
    plt.fill_between(x_pred, lower, upper, color=COLOR_ANALYTICAL, alpha=0.3, 
                     label='95% Analytical CI')
    plt.plot(x_pred, lower, color=COLOR_ANALYTICAL, linestyle='--', alpha=0.5, linewidth=1)
    plt.plot(x_pred, upper, color=COLOR_ANALYTICAL, linestyle='--', alpha=0.5, linewidth=1)
    
    # Add vertical lines to show region boundaries
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.text(-0.5, plt.ylim()[1]*0.9, 'Dense region\n(100 points)', 
             ha='center', fontsize=10, color='gray')
    plt.text(0.5, plt.ylim()[1]*0.9, 'Sparse region\n(10 points)', 
             ha='center', fontsize=10, color='gray')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Plot #2: Analytical Confidence Intervals on Imbalanced Data', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_comparison(X, y, x_pred_boot, y_pred_boot, lower_boot, upper_boot,
                   x_pred_anal, y_pred_anal, lower_anal, upper_anal):
    """
    Plot #3: Combined comparison of both methods
    
    Args:
        All parameters from both bootstrap and analytical methods
    """
    plt.figure(figsize=(14, 8), dpi=DPI)
    
    # Plot data points
    plt.scatter(X, y, color=COLOR_DATA, alpha=0.6, s=30, label='Data points', zorder=5)
    
    # Plot fitted curve (should be the same for both)
    plt.plot(x_pred_boot, y_pred_boot, color=COLOR_FIT, linewidth=2.5, 
             label='Fitted curve', zorder=4)
    
    # Plot bootstrap confidence interval
    plt.fill_between(x_pred_boot, lower_boot, upper_boot, 
                     color=COLOR_BOOTSTRAP, alpha=0.2, label='Bootstrap CI')
    plt.plot(x_pred_boot, lower_boot, color=COLOR_BOOTSTRAP, 
             linestyle='--', alpha=0.7, linewidth=1.5)
    plt.plot(x_pred_boot, upper_boot, color=COLOR_BOOTSTRAP, 
             linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Plot analytical confidence interval
    plt.fill_between(x_pred_anal, lower_anal, upper_anal, 
                     color=COLOR_ANALYTICAL, alpha=0.2, label='Analytical CI')
    plt.plot(x_pred_anal, lower_anal, color=COLOR_ANALYTICAL, 
             linestyle='-.', alpha=0.7, linewidth=1.5)
    plt.plot(x_pred_anal, upper_anal, color=COLOR_ANALYTICAL, 
             linestyle='-.', alpha=0.7, linewidth=1.5)
    
    # Add vertical line and region labels
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Add text annotations
    y_max = max(upper_boot.max(), upper_anal.max())
    plt.text(-0.5, y_max*0.85, 'Dense region\n(100 points)', 
             ha='center', fontsize=11, color='gray', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.5, y_max*0.85, 'Sparse region\n(10 points)', 
             ha='center', fontsize=11, color='gray',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('X', fontsize=13)
    plt.ylabel('Y', fontsize=13)
    plt.title('Plot #3: Comparison of Bootstrap vs Analytical Confidence Intervals', 
              fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add width comparison annotation
    # Calculate average CI widths in each region
    dense_mask = x_pred_boot <= 0
    sparse_mask = x_pred_boot > 0
    
    boot_width_dense = np.mean(upper_boot[dense_mask] - lower_boot[dense_mask])
    boot_width_sparse = np.mean(upper_boot[sparse_mask] - lower_boot[sparse_mask])
    anal_width_dense = np.mean(upper_anal[dense_mask] - lower_anal[dense_mask])
    anal_width_sparse = np.mean(upper_anal[sparse_mask] - lower_anal[sparse_mask])
    
    info_text = (f"Average CI Width:\n"
                f"Bootstrap - Dense: {boot_width_dense:.3f}, Sparse: {boot_width_sparse:.3f}\n"
                f"Analytical - Dense: {anal_width_dense:.3f}, Sparse: {anal_width_sparse:.3f}")
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return plt.gcf()


def out_of_sample_test(X_train, y_train):
    """
    Part 4 (Optional): Out-of-sample test
    
    Generate new data from a different function (y = x + noise)
    and test the model's poor generalization
    
    Args:
        X_train, y_train: Training data used to fit the model
        
    Returns:
        matplotlib figure
    """
    print("\n" + "="*60)
    print("Part 4: Out-of-Sample Test")
    print("="*60)
    
    # Train the polynomial model on original data
    model = make_pipeline(
        PolynomialFeatures(degree=POLYNOMIAL_DEGREE),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    
    # Generate new data from a linear function
    np.random.seed(123)  # Different seed for variety
    x_test = np.linspace(-1, 1, 10)
    noise_test = np.random.normal(0, 0.5, 10)  # Smaller noise for clarity
    y_test_true = x_test + noise_test  # Linear function: y = x + noise
    X_test = x_test.reshape(-1, 1)
    
    # Make predictions with the quadratic model
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test_true - y_test_pred)**2)
    r2 = 1 - (np.sum((y_test_true - y_test_pred)**2) / 
              np.sum((y_test_true - y_test_true.mean())**2))
    
    print(f"Out-of-sample performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Note: Poor performance demonstrates limited generalization")
    
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
    y_smooth_pred = model.predict(X_smooth)
    plt.plot(x_smooth, y_smooth_pred, color=COLOR_FIT, linewidth=2, 
             label='Model prediction (quadratic)')
    
    # Plot prediction errors
    for i in range(len(x_test)):
        plt.plot([x_test[i], x_test[i]], [y_test_true[i], y_test_pred[i]], 
                'r-', alpha=0.5, linewidth=1)
    plt.scatter(x_test, y_test_pred, color='red', s=50, 
               label='Model predictions on test', marker='x', zorder=4)
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Part 4: Out-of-Sample Test - Poor Generalization to Linear Data', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add performance annotation
    perf_text = (f"Performance Metrics:\n"
                f"MSE = {mse:.3f}\n"
                f"R² = {r2:.3f}")
    plt.text(0.02, 0.98, perf_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    
    return plt.gcf()


def main():
    """
    Main execution function to run the complete simulation
    """
    print("="*60)
    print("CONFIDENCE INTERVALS ON IMBALANCED DATA SIMULATION")
    print("="*60)
    
    # Part 1: Generate asymmetric dataset
    print("\n" + "="*60)
    print("Part 1: Generating Asymmetric Dataset")
    print("="*60)
    X, y = generate_asymmetric_dataset()
    
    # Part 2A: Bootstrap confidence intervals
    print("\n" + "="*60)
    print("Part 2A: Bootstrap Confidence Intervals")
    print("="*60)
    x_pred_boot, y_pred_boot, lower_boot, upper_boot = bootstrap_confidence_interval(X, y)
    
    # Part 2B: Analytical confidence intervals
    print("\n" + "="*60)
    print("Part 2B: Analytical Confidence Intervals")
    print("="*60)
    x_pred_anal, y_pred_anal, lower_anal, upper_anal = analytical_confidence_interval(X, y)
    
    # Part 3: Create visualizations
    print("\n" + "="*60)
    print("Part 3: Creating Visualizations")
    print("="*60)
    
    # Plot 1: Bootstrap CI
    fig1 = plot_bootstrap_ci(X, y, x_pred_boot, y_pred_boot, lower_boot, upper_boot)
    fig1.savefig('plot1_bootstrap_ci.png', dpi=150, bbox_inches='tight')
    print("  Saved: plot1_bootstrap_ci.png")
    
    # Plot 2: Analytical CI
    fig2 = plot_analytical_ci(X, y, x_pred_anal, y_pred_anal, lower_anal, upper_anal)
    fig2.savefig('plot2_analytical_ci.png', dpi=150, bbox_inches='tight')
    print("  Saved: plot2_analytical_ci.png")
    
    # Plot 3: Comparison
    fig3 = plot_comparison(X, y, x_pred_boot, y_pred_boot, lower_boot, upper_boot,
                          x_pred_anal, y_pred_anal, lower_anal, upper_anal)
    fig3.savefig('plot3_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: plot3_comparison.png")
    
    # Part 4: Out-of-sample test
    fig4 = out_of_sample_test(X, y)
    fig4.savefig('plot4_out_of_sample.png', dpi=150, bbox_inches='tight')
    print("  Saved: plot4_out_of_sample.png")
    
    # Display all plots
    plt.show()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("1. Bootstrap CI tends to be wider in sparse regions due to sampling variability")
    print("2. Analytical CI reflects theoretical uncertainty based on model assumptions")
    print("3. Both methods show increased uncertainty in the sparse region [0, 1]")
    print("4. Out-of-sample test demonstrates poor generalization to different functions")
    print("\nAll plots have been saved to the current directory.")


if __name__ == "__main__":
    main()