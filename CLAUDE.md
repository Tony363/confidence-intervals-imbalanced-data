# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science project focused on visualizing confidence intervals on imbalanced data using Python. The project demonstrates the differences between bootstrapped and analytical confidence intervals when dealing with asymmetrically distributed data points.

## Project Structure

The project implements a statistical simulation with four main components:

1. **Part 1: Generate Asymmetrical Dataset**
   - Creates a parabolic function (y = x² + noise)
   - 100 dense points in range [-1, 0]
   - 10 sparse points in range [0, 1]
   - Total of 110 points with intentional imbalance

2. **Part 2: Calculate Confidence Intervals**
   - Method A: Bootstrap resampling with polynomial regression
   - Method B: Analytical regression with standard formulas

3. **Part 3: Comparison Visualization**
   - Combines both methods on a single plot for direct comparison

4. **Part 4 (Optional): Out-of-Sample Testing**
   - Tests model generalization on linear data (y = x + noise)

## Current Implementation

The project is fully implemented in `confidence_intervals_simulation.py` with all four parts completed and tested.

### File Structure
```
sims_temp/
├── confidence_intervals_simulation.py  # Main implementation (all parts)
├── requirements.txt                    # Dependencies (NumPy <2.0 required)
├── README.md                           # Comprehensive documentation
├── CLAUDE.md                           # This file
├── plot1_bootstrap_ci.png             # Bootstrap CI visualization
├── plot2_analytical_ci.png            # Analytical CI visualization
├── plot3_comparison.png               # Combined comparison plot
└── plot4_out_of_sample.png           # Out-of-sample test results
```

### Dependencies and Compatibility

```bash
# Install with version constraints
pip install -r requirements.txt
```

**Important**: NumPy 2.x causes compatibility issues with matplotlib. The requirements.txt specifies NumPy <2.0.

### Running the Simulation

```bash
python confidence_intervals_simulation.py
```

Execution takes approximately 15-20 seconds (primarily due to 5000 bootstrap iterations).

## Key Implementation Decisions

### Technical Choices Made

1. **Bootstrap Parameters**:
   - **Iterations**: 5000 (provides stable CI estimates)
   - **Resampling**: With replacement, maintaining n=110
   - **Progress Reporting**: Every 1000 iterations for user feedback

2. **Polynomial Regression**:
   - **Degree**: 2 (matches true quadratic function)
   - **Implementation**: scikit-learn's `PolynomialFeatures` + `LinearRegression`
   - **Pipeline**: Used `make_pipeline` for clean workflow

3. **Analytical CI Calculation**:
   - **Standard Error Formula**: SE = √(MSE × (1 + leverage))
   - **Leverage Calculation**: x'(X'X)^(-1)x for each prediction point
   - **t-distribution**: Used for proper CI with 107 degrees of freedom

4. **Visualization Strategy**:
   - **Consistent Color Scheme**: Red (bootstrap), Green (analytical), Blue (fit)
   - **Transparency**: Alpha=0.3 for CI bands, 0.5 for data points
   - **Annotations**: Region labels, CI width metrics, performance statistics
   - **Resolution**: 150 DPI for saved figures, 300 prediction points for smooth curves

### Key Functions

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `generate_asymmetric_dataset()` | Creates imbalanced quadratic data | Returns (X, y) with 110 points |
| `bootstrap_confidence_interval()` | Bootstrap CI calculation | n_iterations=5000, returns bounds |
| `analytical_confidence_interval()` | Analytical CI using regression theory | Uses t-distribution, returns bounds |
| `plot_comparison()` | Combined visualization | Overlays both CI methods with metrics |
| `out_of_sample_test()` | Tests generalization failure | Linear data, returns MSE and R² |

### Performance Optimizations

1. **Vectorized Operations**: All calculations use NumPy arrays
2. **Pre-allocated Arrays**: Bootstrap predictions stored in pre-sized array
3. **Efficient Percentile Calculation**: Single `np.percentile` call across all iterations
4. **Memory Management**: Reuse model objects across iterations

## Known Issues and Solutions

### Issue 1: NumPy Version Compatibility
**Problem**: NumPy 2.x breaks matplotlib compatibility
**Solution**: Downgrade to NumPy 1.26.4 (handled in requirements.txt)
```bash
pip install numpy==1.26.4
```

### Issue 2: Display Backend Warnings
**Problem**: Qt/Wayland warnings when displaying plots
**Solution**: These are cosmetic warnings and don't affect functionality

### Issue 3: Bootstrap Stability
**Observation**: CI stabilizes after ~2000 iterations
**Implication**: 5000 iterations provides margin of safety

## Statistical Insights from Implementation

### Bootstrap vs Analytical CI Behavior

1. **Width Differences**:
   - Bootstrap CI consistently wider (more conservative)
   - Difference more pronounced in sparse region
   - Bootstrap captures sampling uncertainty analytical method misses

2. **Computational Trade-offs**:
   - Bootstrap: Computationally intensive but distribution-free
   - Analytical: Fast but assumes normality and correct model

3. **Edge Effects**:
   - Both methods show increased uncertainty at boundaries
   - Bootstrap handles asymmetry better than analytical

### Out-of-Sample Results

- **MSE**: 1.116 (indicates poor fit)
- **R²**: -0.418 (worse than mean prediction)
- **Insight**: Quadratic model completely fails on linear data

## Potential Improvements

### Code Structure
1. **Modularization**: Could separate into multiple files for larger projects
2. **Configuration**: Could use YAML/JSON for parameters
3. **Caching**: Could cache bootstrap results for interactive exploration

### Statistical Enhancements
1. **BCa Bootstrap**: Bias-corrected and accelerated intervals
2. **Prediction Intervals**: Different from confidence intervals
3. **Cross-Validation**: For polynomial degree selection
4. **Heteroscedasticity**: Model variance as function of x

### Visualization Enhancements
1. **Interactive Plots**: Using Plotly or Bokeh
2. **Animation**: Show bootstrap sampling process
3. **3D Visualization**: CI surface over parameter space

## Testing Recommendations

```python
# Quick validation tests
assert len(generate_asymmetric_dataset()[0]) == 110
assert bootstrap_ci_width > analytical_ci_width  # Generally true
assert out_of_sample_r2 < 0  # Model fails on linear data
```

## Common Modifications

### Change Bootstrap Iterations
```python
BOOTSTRAP_ITERATIONS = 10000  # Line 21
```

### Adjust Polynomial Degree
```python
POLYNOMIAL_DEGREE = 3  # Line 23, affects model complexity
```

### Modify Data Imbalance
```python
x_dense = np.linspace(-1, 0, 200)  # More dense points
x_sparse = np.linspace(0, 1, 5)    # Fewer sparse points
```

## Debugging Tips

1. **Bootstrap Taking Too Long**: Reduce iterations to 1000 for testing
2. **Memory Issues**: Process bootstrap in batches
3. **Plot Not Showing**: Check matplotlib backend with `plt.get_backend()`
4. **CI Too Narrow/Wide**: Verify confidence level (default 95%)

## References for Future Development

- Bootstrap Methods: Efron & Tibshirani (1993)
- Regression CIs: Montgomery & Peck (1992)
- scikit-learn polynomial regression documentation
- matplotlib confidence band examples

## Mathematical Background

The project explores how confidence intervals behave differently when:
- Data density varies across the domain
- Using resampling (bootstrap) vs analytical approaches
- The true confidence varies due to data sparsity

This is particularly relevant for understanding model uncertainty in regions with limited data.