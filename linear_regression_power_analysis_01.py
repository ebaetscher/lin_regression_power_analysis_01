import numpy as np
from scipy import stats
import pandas as pd

def regression_power_analysis(n, r_squared, n_predictors=1, alpha=0.05):
    """
    Perform power analysis for linear regression
    
    Parameters:
    -----------
    n : int
        Sample size
    r_squared : float
        Observed R-squared value
    n_predictors : int
        Number of predictors (default=1)
    alpha : float
        Significance level (default=0.05)
    
    Returns:
    --------
    dict with power analysis results
    """
    
    # Convert R² to Cohen's f²
    if r_squared >= 1:
        f_squared = np.inf
    else:
        f_squared = r_squared / (1 - r_squared)
    
    # Degrees of freedom
    df1 = n_predictors
    df2 = n - n_predictors - 1
    
    # Calculate observed power (post-hoc)
    # Non-centrality parameter
    lambda_ncp = f_squared * n
    
    # Critical F-value
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    
    # Power = P(F > f_crit | lambda_ncp)
    observed_power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_ncp)
    
    # Calculate minimum detectable effect size for 80% power
    # This requires solving for f² that gives power = 0.80
    def power_for_f_squared(f2):
        ncp = f2 * n
        return 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
    
    # Binary search for f² that gives 80% power
    f2_range = np.logspace(-3, 1, 1000)
    powers = [power_for_f_squared(f2) for f2 in f2_range]
    
    # Find f² closest to 80% power
    idx_80 = np.argmin(np.abs(np.array(powers) - 0.80))
    f2_mde_80 = f2_range[idx_80]
    r2_mde_80 = f2_mde_80 / (1 + f2_mde_80)
    
    # Similarly for 50% power (helps understand sensitivity)
    idx_50 = np.argmin(np.abs(np.array(powers) - 0.50))
    f2_mde_50 = f2_range[idx_50]
    r2_mde_50 = f2_mde_50 / (1 + f2_mde_50)
    
    results = {
        'n': n,
        'observed_r_squared': r_squared,
        'observed_f_squared': f_squared,
        'observed_power': observed_power,
        'mde_r2_80pct_power': r2_mde_80,
        'mde_f2_80pct_power': f2_mde_80,
        'mde_r2_50pct_power': r2_mde_50,
        'interpretation': interpret_power(observed_power, r_squared, r2_mde_80)
    }
    
    return results

def interpret_power(observed_power, observed_r2, mde_r2):
    """Generate interpretation of power analysis results"""
    
    interp = []
    
    if observed_power < 0.50:
        interp.append("Very low power: High risk of Type II error")
    elif observed_power < 0.80:
        interp.append("Underpowered: Moderate risk of Type II error")
    else:
        interp.append("Adequate power achieved")
    
    if observed_r2 < mde_r2:
        interp.append(f"Observed effect (R²={observed_r2:.3f}) is smaller than "
                     f"minimum detectable effect with 80% power (R²={mde_r2:.3f})")
        interp.append("This suggests the study was underpowered to detect the observed effect size")
    else:
        interp.append(f"Observed effect (R²={observed_r2:.3f}) exceeds minimum "
                     f"detectable effect (R²={mde_r2:.3f}), yet was not significant")
        interp.append("This suggests a true null or very small effect")
    
    return " | ".join(interp)

def print_power_results(results):
    """Pretty print power analysis results"""
    
    print("="*70)
    print("POWER ANALYSIS FOR LINEAR REGRESSION")
    print("="*70)
    print(f"Sample size (n): {results['n']}")
    print(f"Observed R²: {results['observed_r_squared']:.4f}")
    print(f"Observed f²: {results['observed_f_squared']:.4f}")
    print(f"\nPost-hoc power: {results['observed_power']:.1%}")
    print(f"\nMinimum Detectable Effect (80% power):")
    print(f"  R²: {results['mde_r2_80pct_power']:.4f}")
    print(f"  f²: {results['mde_f2_80pct_power']:.4f}")
    print(f"\nMinimum Detectable Effect (50% power):")
    print(f"  R²: {results['mde_r2_50pct_power']:.4f}")
    print(f"\n{results['interpretation']}")
    print("="*70)

# Example
results = regression_power_analysis(n=45, r_squared=0.043)
print_power_results(results)