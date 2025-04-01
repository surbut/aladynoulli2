import torch
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
from scipy.interpolate import interp1d

# Load saved model
model_path = '/Users/sarahurbut/Dropbox (Personal)/model_with_poptrajectory_lr1e-4_grad.pt'
saved_model = torch.load(model_path)

# Extract predictions and actual values
Y = saved_model['Y']
predicted = model.forward()
pi_pred = predicted[0] if isinstance(predicted, tuple) else predicted
pi_pred = pi_pred.cpu().detach().numpy()

# Calculate marginal risks
observed_risk = Y.mean(axis=0).flatten()
predicted_risk = pi_pred.mean(axis=0).flatten()

# Create plots for presentation
plt.style.use('seaborn-paper')

def create_calibration_plots():
    # 1. Original vs LOESS Calibration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original predictions
    ax1.scatter(observed_risk, predicted_risk, alpha=0.5, s=20)
    ax1.plot([0, max(observed_risk)], [0, max(observed_risk)], 'r--', label='Perfect calibration')
    ax1.set_title('Original Predictions')
    ax1.set_xlabel('Observed Risk')
    ax1.set_ylabel('Predicted Risk')
    
    # LOESS calibration
    calibrated_risk = lowess(observed_risk, predicted_risk, frac=0.3, return_sorted=False)
    ax2.scatter(observed_risk, calibrated_risk, alpha=0.5, s=20)
    ax2.plot([0, max(observed_risk)], [0, max(observed_risk)], 'r--', label='Perfect calibration')
    ax2.set_title('LOESS Calibrated')
    ax2.set_xlabel('Observed Risk')
    ax2.set_ylabel('Calibrated Risk')
    
    plt.tight_layout()
    plt.savefig('calibration_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. LOESS Calibration Curve
    plt.figure(figsize=(8, 6))
    # Sort for smooth curve
    sort_idx = np.argsort(predicted_risk)
    pred_sorted = predicted_risk[sort_idx]
    obs_sorted = observed_risk[sort_idx]
    
    # Compute LOESS curve
    smoothed = lowess(obs_sorted, pred_sorted, frac=0.3)
    
    plt.scatter(predicted_risk, observed_risk, alpha=0.3, s=20, label='Original data')
    plt.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='LOESS curve')
    plt.plot([0, max(predicted_risk)], [0, max(predicted_risk)], 'k--', label='Perfect calibration')
    plt.xlabel('Predicted Risk')
    plt.ylabel('Observed Risk')
    plt.title('LOESS Calibration Curve')
    plt.legend()
    plt.savefig('loess_curve.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Error Analysis
    residuals_original = observed_risk - predicted_risk
    residuals_calibrated = observed_risk - calibrated_risk
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(residuals_original, ax=ax1, bins=50)
    ax1.set_title('Original Residuals')
    ax1.set_xlabel('Error')
    
    sns.histplot(residuals_calibrated, ax=ax2, bins=50)
    ax2.set_title('Calibrated Residuals')
    ax2.set_xlabel('Error')
    
    plt.tight_layout()
    plt.savefig('error_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    return calibrated_risk

# Generate plots and get calibrated predictions
calibrated_risk = create_calibration_plots(saved_model)

# Print performance metrics
from sklearn.metrics import r2_score, mean_squared_error
print("\nPerformance Metrics:")
print(f"Original R²: {r2_score(observed_risk, predicted_risk):.3f}")
print(f"LOESS Calibrated R²: {r2_score(observed_risk, calibrated_risk):.3f}")
print(f"Original RMSE: {np.sqrt(mean_squared_error(observed_risk, predicted_risk)):.6f}")
print(f"LOESS Calibrated RMSE: {np.sqrt(mean_squared_error(observed_risk, calibrated_risk)):.6f}")