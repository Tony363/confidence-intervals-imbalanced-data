#!/usr/bin/env python3
"""
Demo script to run both polynomial and XGBoost confidence interval simulations.
This script provides a quick way to compare both approaches with reduced iterations.
"""

import numpy as np
import matplotlib.pyplot as plt
from confidence_intervals_simulation import main as run_polynomial_sim
from confidence_intervals_xgboost import main as run_xgboost_sim
from confidence_intervals_xgboost import (
    generate_asymmetric_dataset, 
    bootstrap_polynomial_ci, 
    bootstrap_xgboost_ci,
    plot_model_comparison
)

def quick_comparison_demo():
    """
    Run a quick comparison demo with reduced bootstrap iterations
    """
    print("="*60)
    print("QUICK COMPARISON DEMO")
    print("Polynomial vs XGBoost Confidence Intervals")
    print("="*60)
    
    # Generate the same asymmetric dataset
    X, y = generate_asymmetric_dataset()
    
    # Run bootstrap with fewer iterations for speed
    n_iterations = 1000
    print(f"\nRunning bootstrap comparison with {n_iterations} iterations each...")
    
    # Bootstrap with polynomial regression
    print("\n--- Polynomial Bootstrap ---")
    poly_results = bootstrap_polynomial_ci(X, y, n_iterations=n_iterations)
    
    # Bootstrap with XGBoost
    print("\n--- XGBoost Bootstrap ---")
    xgb_results = bootstrap_xgboost_ci(X, y, n_iterations=n_iterations)
    
    # Create comparison plot
    print("\n--- Creating Comparison Plot ---")
    fig = plot_model_comparison(X, y, poly_results, xgb_results)
    fig.savefig('quick_comparison_demo.png', dpi=150, bbox_inches='tight')
    print("  Saved: quick_comparison_demo.png")
    
    # Print summary statistics
    poly_metrics = poly_results['performance_metrics']
    xgb_metrics = xgb_results['performance_metrics']
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'RMSE':<12} {'R²':<12} {'CI Width (Dense)':<18} {'CI Width (Sparse)':<18}")
    print(f"{'-'*80}")
    
    # Calculate CI widths
    dense_mask = poly_results['x_pred'] <= 0
    sparse_mask = poly_results['x_pred'] > 0
    
    poly_width_dense = np.mean(poly_results['upper_bound'][dense_mask] - poly_results['lower_bound'][dense_mask])
    poly_width_sparse = np.mean(poly_results['upper_bound'][sparse_mask] - poly_results['lower_bound'][sparse_mask])
    xgb_width_dense = np.mean(xgb_results['upper_bound'][dense_mask] - xgb_results['lower_bound'][dense_mask])
    xgb_width_sparse = np.mean(xgb_results['upper_bound'][sparse_mask] - xgb_results['lower_bound'][sparse_mask])
    
    print(f"{'Polynomial':<15} {np.mean(poly_metrics['rmse']):<12.4f} {np.mean(poly_metrics['r2']):<12.4f} {poly_width_dense:<18.4f} {poly_width_sparse:<18.4f}")
    print(f"{'XGBoost':<15} {np.mean(xgb_metrics['rmse']):<12.4f} {np.mean(xgb_metrics['r2']):<12.4f} {xgb_width_dense:<18.4f} {xgb_width_sparse:<18.4f}")
    
    print(f"\nKey Observations:")
    print(f"1. XGBoost RMSE is {'lower' if np.mean(xgb_metrics['rmse']) < np.mean(poly_metrics['rmse']) else 'higher'} than polynomial")
    print(f"2. XGBoost R² is {'higher' if np.mean(xgb_metrics['r2']) > np.mean(poly_metrics['r2']) else 'lower'} than polynomial")
    print(f"3. Both models show wider confidence intervals in sparse regions")
    print(f"4. XGBoost feature importance: {np.mean(xgb_results['importance_history']):.4f} (single feature gets full importance)")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")

def run_full_simulations():
    """
    Run both full simulations (this may take several minutes)
    """
    print("="*60)
    print("RUNNING FULL SIMULATIONS")
    print("="*60)
    print("Warning: This will take several minutes due to 5000 bootstrap iterations each")
    
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("Full simulation cancelled.")
        return
    
    print("\n--- Running Polynomial Simulation ---")
    run_polynomial_sim()
    
    print("\n--- Running XGBoost Simulation ---")
    run_xgboost_sim()
    
    print("\n" + "="*60)
    print("BOTH SIMULATIONS COMPLETE")
    print("="*60)

def main():
    """
    Main demo function with user choice
    """
    print("Confidence Intervals Demo - Polynomial vs XGBoost")
    print("="*60)
    print("Choose an option:")
    print("1. Quick Comparison Demo (1000 iterations, ~30 seconds)")
    print("2. Full Simulations (5000 iterations each, ~5-10 minutes)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            quick_comparison_demo()
            break
        elif choice == '2':
            run_full_simulations()
            break
        elif choice == '3':
            print("Exiting demo.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()