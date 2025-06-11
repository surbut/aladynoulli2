from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any 

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering  # Add this import

def load_model_essentials(base_path='/Users/sarahurbut/Dropbox/data_for_running/'):
    """
    Load all essential components
    """
    print("Loading components...")
    
    # Load large matrices
    Y = torch.load(base_path + 'Y_tensor.pt')
    E = torch.load(base_path + 'E_matrix.pt')
    G = torch.load(base_path + 'G_matrix.pt')
    
    # Load other components
    essentials = torch.load(base_path + 'model_essentials.pt')
    
    print("Loaded all components successfully!")
    
    return Y, E, G, essentials



def plot_roc_curve(y_true, y_pred, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=label)

def create_traditional_calibration_plot(predictions=None, Y=None, checkpoint_path=None, cov_df=None, mu_dt=None, n_bins=10, use_log_scale=True, min_bin_count=1000, save_path=None):
    """Create a calibration plot comparing predicted vs observed event rates at enrollment.
    
    Args:
        predictions: Predicted probabilities at enrollment (N,) - optional if using checkpoint
        Y: Observed events at enrollment (N,) - optional if using checkpoint
        checkpoint_path: Path to model checkpoint - optional if providing predictions directly
        cov_df: DataFrame containing enrollment ages - required if using checkpoint
        mu_dt: Smoothed prevalence tensor (D, T) - optional, will use raw events if not provided
        n_bins: Number of bins for calibration
        use_log_scale: Whether to use log-scale binning (recommended for rare events)
        min_bin_count: Minimum number of samples per bin
        save_path: Path to save the plot as PDF (optional)
    """
    # Load predictions from checkpoint if provided
    if checkpoint_path is not None:
        if cov_df is None:
            raise ValueError("cov_df is required when using checkpoint_path")
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        
        # Get parameters from state dict
        lambda_ = state_dict['lambda_']  # Shape: (N, K, T)
        phi = state_dict['phi']  # Shape: (K, D, T)
        kappa = state_dict['kappa']  # Shape: scalar
        
        # Calculate theta (normalized lambda)
        theta = torch.zeros_like(lambda_)
        for i in range(lambda_.shape[0]):
            for t in range(lambda_.shape[2]):
                exp_lambda = torch.exp(lambda_[i, :, t])
                sum_exp_lambda = torch.sum(exp_lambda)
                theta[i, :, t] = exp_lambda / sum_exp_lambda if sum_exp_lambda > 0 else torch.zeros_like(exp_lambda)
        
        # Calculate phi probabilities (sigmoid)
        phi_prob = torch.sigmoid(phi)
        
        # Calculate pi (disease probabilities)
        pi = torch.zeros((lambda_.shape[0], phi.shape[1], lambda_.shape[2]))
        for d in range(phi.shape[1]):
            for t in range(lambda_.shape[2]):
                pi[:, d, t] = kappa * torch.sum(theta[:, :, t] * phi_prob[:, d, t], dim=1)
        
        # Convert to numpy for analysis
        pi_np = pi.detach().numpy()  # Shape: (N, D, T)
        if mu_dt is not None and torch.is_tensor(mu_dt):
            mu_dt = mu_dt.detach().numpy()
        
        # Flatten predictions and observations to 1D arrays
        all_predictions = []
        all_observations = []
        
        n_patients, n_diseases, n_timepoints = pi_np.shape
        
        # For each disease, collect predictions and observations at enrollment time
        for d in range(n_diseases):
            for i, row in enumerate(cov_df.itertuples()):
                enroll_age = row.age
                enroll_time = int(enroll_age - 30)  # Convert age to time index
                
                if enroll_time < 0 or enroll_time >= n_timepoints:
                    continue
                    
                # Get prediction and observation at enrollment
                pred = pi_np[i, d, enroll_time]
                
                # Use mu_dt directly as the observed rate
                obs = mu_dt[d, enroll_time]
                
                # Skip if observation is NaN
                if np.isnan(obs):
                    continue
                    
                all_predictions.append(pred)
                all_observations.append(obs)
        
        predictions = np.array(all_predictions)
        Y = np.array(all_observations)
    
    # Convert to numpy if needed
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    if hasattr(Y, 'numpy'):
        Y = Y.numpy()
    
    # Create bins in log or linear space
    if use_log_scale:
        bin_edges = np.logspace(np.log10(max(1e-7, predictions.min())), 
                              np.log10(predictions.max()), 
                              n_bins + 1)
    else:
        bin_edges = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
    
    # Initialize arrays for storing bin statistics
    bin_means = np.zeros(n_bins)
    observed_rates = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    # Calculate statistics for each bin
    for i in range(n_bins):
        bin_mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if np.sum(bin_mask) >= min_bin_count:
            bin_means[i] = np.mean(predictions[bin_mask])
            observed_rates[i] = np.mean(Y[bin_mask])
            counts[i] = np.sum(bin_mask)
    
    # Create figure with specific size and high DPI
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Plot perfect calibration line
    if use_log_scale:
        plt.plot([1e-7, 1], [1e-7, 1], '--', color='gray', alpha=0.5, label='Perfect calibration')
    else:
        plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Perfect calibration')
    
    # Plot calibration points
    valid_mask = counts >= min_bin_count
    plt.plot(bin_means[valid_mask], 
            observed_rates[valid_mask],
            'o-', 
            color='#1f77b4',
            markersize=8,
            linewidth=2,
            label='Observed rates')

    # Customize plot appearance
    if use_log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(max(1e-7, predictions.min()/2), min(1, predictions.max()*2))
        plt.ylim(max(1e-7, predictions.min()/2), min(1, predictions.max()*2))
    
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.xlabel('Predicted Event Rate at Enrollment', fontsize=12)
    plt.ylabel('Observed Event Rate at Enrollment', fontsize=12)
    plt.title('Model Calibration at Enrollment', fontsize=14, pad=20)
    
    # Add summary statistics to plot
    mse = np.mean((bin_means[valid_mask] - observed_rates[valid_mask])**2)
    mean_pred = np.mean(predictions)
    mean_obs = np.mean(Y)
    
    stats_text = f'MSE: {mse:.2e}\n'
    stats_text += f'Mean Predicted: {mean_pred:.2e}\n'
    stats_text += f'Mean Observed: {mean_obs:.2e}\n'
    stats_text += f'N per bin: {int(np.mean(counts[valid_mask])):,}'
    
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add bin counts as small annotations
    for i in range(n_bins):
        if counts[i] >= min_bin_count:
            plt.annotate(f'n={int(counts[i]):,}',
                        (bin_means[i], observed_rates[i]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def analyze_internal_kappa_calibration(model, mu_dt):
    """
    Analyze calibration for a model with internal kappa implementation
    
    Args:
        model: The loaded model with internal kappa
        mu_dt: The observed prevalence matrix
    
    Returns:
        Calibration plot figure
    """
    # Get predictions from the model
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
    
    # Convert to numpy for analysis
    pi_np = pi.detach().numpy()
    
    # Aggregate predictions and observations
    all_pi_values = []
    all_mu_dt_values = []
    
    n_diseases, n_timepoints = mu_dt.shape
    print(f"Number of diseases: {n_diseases}, Number of timepoints: {n_timepoints}")
    
    # Aggregate pi values for each disease and timepoint
    for d in range(n_diseases):
        for t in range(n_timepoints):
            # Get valid entries (non-NaN) for this disease and timepoint
            valid_mask = ~np.isnan(pi_np[:, d, t])
            n_valid = np.sum(valid_mask)
            
            if n_valid > 0:
                avg_pi_dt = np.mean(pi_np[valid_mask, d, t])
                all_pi_values.append(avg_pi_dt)
                all_mu_dt_values.append(mu_dt[d, t])
    
    # Convert to numpy arrays
    all_pi_values = np.array(all_pi_values)
    all_mu_dt_values = np.array(all_mu_dt_values)
    
    # Calculate calibration factor
    valid_mask = (all_pi_values > 0) & (all_mu_dt_values > 0)
    pi_valid = all_pi_values[valid_mask]
    mu_dt_valid = all_mu_dt_values[valid_mask]
    
    mean_pred = np.mean(pi_valid)
    mean_obs = np.mean(mu_dt_valid)
    calib_factor = mean_obs / mean_pred
    
    print(f"Overall mean predicted risk: {mean_pred:.6f}")
    print(f"Overall mean observed risk: {mean_obs:.6f}")
    print(f"Overall calibration factor: {calib_factor:.4f}")
    
    # Create figure with two plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Original predictions vs. smoothed prevalence
    ax1.scatter(mu_dt_valid, pi_valid, alpha=0.4, s=5, color='blue', edgecolor='none')
    ax1.plot([1e-8, 2e-2], [1e-8, 2e-2], 'k--', linewidth=1.5)  # Perfect calibration line
    
    ax1.set_xlabel('Smoothed Prevalence (μ_dt)')
    ax1.set_ylabel('Predicted Risk (Internal Kappa)')
    ax1.set_title('Predictions with Internal Kappa vs Smoothed Prevalence')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e-8, 2e-2)
    ax1.set_ylim(1e-8, 2e-2)
    ax1.grid(True, alpha=0.3)
    
    # Add text with calibration metrics
    ax1.text(0.05, 0.95, 
             f"Mean predicted: {mean_pred:.6f}\nMean observed: {mean_obs:.6f}\nCalib factor: {calib_factor:.4f}", 
             transform=ax1.transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Post-hoc calibrated predictions
    scaled_pi = pi_valid * calib_factor
    
    ax2.scatter(mu_dt_valid, scaled_pi, alpha=0.4, s=5, color='green', edgecolor='none')
    ax2.plot([1e-8, 2e-2], [1e-8, 2e-2], 'k--', linewidth=1.5)  # Perfect calibration line
    
    ax2.set_xlabel('Smoothed Prevalence (μ_dt)')
    ax2.set_ylabel('Post-hoc Calibrated Risk')
    ax2.set_title(f'Post-hoc Calibration (factor={calib_factor:.4f})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1e-8, 2e-2)
    ax2.set_ylim(1e-8, 2e-2)
    ax2.grid(True, alpha=0.3)
    
    # Add text with calibration improvement metrics
    mse_orig = np.mean((pi_valid - mu_dt_valid)**2)
    mse_calib = np.mean((scaled_pi - mu_dt_valid)**2)
    improvement = (mse_orig - mse_calib) / mse_orig * 100
    
    ax2.text(0.05, 0.95, 
             f"Original MSE: {mse_orig:.6f}\nCalibrated MSE: {mse_calib:.6f}\nImprovement: {improvement:.2f}%", 
             transform=ax2.transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nCalibration with Internal Kappa:")
    print(f"Mean predicted risk: {mean_pred:.6f}")
    print(f"Mean observed prevalence: {mean_obs:.6f}")
    print(f"Calibration factor needed: {calib_factor:.4f}")
    print(f"MSE before post-hoc calibration: {mse_orig:.6f}")
    print(f"MSE after post-hoc calibration: {mse_calib:.6f}")
    print(f"Improvement from post-hoc calibration: {improvement:.2f}%")
    
    return fig, all_pi_values, all_mu_dt_values, calib_factor

# Usage example:
# Assuming you have your model and mu_dt (prevalence matrix) available
#mu_dt = essentials['prevalence_t'] # Shape: (348, 52)
#fig, all_pi, all_mu_dt, calib_factor = analyze_internal_kappa_calibration(model, mu_dt)


def plot_10year_with_isotonic_1year_calibration(model, pce_df, calibrator_1yr, ascvd_indices=[111,112,113,114,115,116], n_bins=10):
    """
    Create a 10-year calibration plot using 1-year isotonic calibration
    
    Args:
        model: The loaded model with internal kappa
        pce_df: DataFrame containing PCE scores
        calibrator_1yr: Trained isotonic regression model for 1-year calibration
        ascvd_indices: List of indices for ASCVD diseases
        n_bins: Number of bins for calibration curves
        
    Returns:
        Figure and calibration statistics
    """
    # Get model predictions
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
    
    # Convert to numpy
    pi_np = pi.detach().numpy()
    Y_np = model.Y.detach().numpy()
    
    # Storage for results
    all_model_risks = []
    all_calibrated_risks = []
    all_actual_outcomes = []
    all_pce_risks = []
    patient_ids = []
    
    # Process each patient
    for i, row in enumerate(pce_df.itertuples()):
        if i >= pi_np.shape[0] or np.isnan(row.pce):
            continue
            
        age = row.age
        enroll_time = int(age - 30)
        
        if enroll_time < 0 or enroll_time + 10 >= Y_np.shape[2]:
            continue
        
        # Get ASCVD predictions for next 10 years
        pi_ascvd = pi_np[i, ascvd_indices, enroll_time:enroll_time+10]
        
        # Apply 1-year isotonic calibration to each 1-year prediction
        pi_ascvd_calibrated = np.zeros_like(pi_ascvd)
        for t in range(pi_ascvd.shape[1]):
            for d in range(pi_ascvd.shape[0]):
                pi_ascvd_calibrated[d, t] = calibrator_1yr.transform([pi_ascvd[d, t]])[0]
        
        # Calculate 10-year risk using survival probability
        survival_prob = np.prod(1 - pi_ascvd)
        ten_year_risk = 1 - survival_prob
        
        survival_prob_calibrated = np.prod(1 - pi_ascvd_calibrated)
        ten_year_risk_calibrated = 1 - survival_prob_calibrated
        
        # Get actual outcome (any ASCVD event in 10 years)
        Y_ascvd = Y_np[i, ascvd_indices, enroll_time:enroll_time+10]
        had_event = np.any(Y_ascvd)
        
        # Store results
        all_model_risks.append(ten_year_risk)
        all_calibrated_risks.append(ten_year_risk_calibrated)
        all_actual_outcomes.append(had_event)
        all_pce_risks.append(row.pce)
        patient_ids.append(row.Index)
    
    # Convert to arrays
    all_model_risks = np.array(all_model_risks)
    all_calibrated_risks = np.array(all_calibrated_risks)
    all_actual_outcomes = np.array(all_actual_outcomes)
    all_pce_risks = np.array(all_pce_risks)
    
    # Print overall statistics
    print("\nOverall 10-Year ASCVD Risk Statistics:")
    print(f"Total patients: {len(all_model_risks):,}")
    print(f"Mean original model risk: {np.mean(all_model_risks):.4f}")
    print(f"Mean calibrated risk: {np.mean(all_calibrated_risks):.4f}")
    print(f"Mean PCE risk: {np.mean(all_pce_risks):.4f}")
    print(f"Observed event rate: {np.mean(all_actual_outcomes):.4f}")
    
    # Create calibration curves using equal-sized bins
    def calculate_calibration_with_ci(risks, outcomes):
        # Sort by risk
        sorted_idx = np.argsort(risks)
        sorted_risks = risks[sorted_idx]
        sorted_outcomes = outcomes[sorted_idx]
        
        # Create bins with equal number of patients
        bin_size = len(sorted_risks) // n_bins
        bin_pred = []
        bin_obs = []
        bin_lower = []
        bin_upper = []
        bin_counts = []
        
        for i in range(n_bins):
            start = i * bin_size
            end = (i+1) * bin_size if i < n_bins-1 else len(sorted_risks)
            
            if end > start:
                bin_pred.append(np.mean(sorted_risks[start:end]))
                obs_rate = np.mean(sorted_outcomes[start:end])
                bin_obs.append(obs_rate)
                bin_counts.append(end - start)
                
                # Wilson score interval for confidence intervals
                n = end - start
                z = 1.96  # 95% confidence
                
                denominator = 1 + z**2/n
                center = (obs_rate + z**2/(2*n))/denominator
                halfwidth = z * np.sqrt(obs_rate*(1-obs_rate)/n + z**2/(4*n**2))/denominator
                
                bin_lower.append(max(0, center - halfwidth))
                bin_upper.append(min(1, center + halfwidth))
        
        return bin_pred, bin_obs, bin_lower, bin_upper, bin_counts
    
    # Calculate calibration curves
    model_pred, model_obs, model_lower, model_upper, model_counts = calculate_calibration_with_ci(
        all_model_risks, all_actual_outcomes)
    calib_pred, calib_obs, calib_lower, calib_upper, _ = calculate_calibration_with_ci(
        all_calibrated_risks, all_actual_outcomes)
    pce_pred, pce_obs, pce_lower, pce_upper, _ = calculate_calibration_with_ci(
        all_pce_risks, all_actual_outcomes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot calibration curves
    ax.plot(model_pred, model_obs, 'o-', color='blue', label='Original Model')
    ax.fill_between(model_pred, model_lower, model_upper, color='blue', alpha=0.1)
    
    ax.plot(calib_pred, calib_obs, 'o-', color='green', label='Isotonic Calibrated Model')
    ax.fill_between(calib_pred, calib_lower, calib_upper, color='green', alpha=0.1)
    
    ax.plot(pce_pred, pce_obs, 'o-', color='orange', label='PCE')
    ax.fill_between(pce_pred, pce_lower, pce_upper, color='orange', alpha=0.1)
    
    ax.plot([0, 0.3], [0, 0.3], 'k--', label='Perfect Calibration')
    
    # Add bin counts as annotations
    for i, (x, y) in enumerate(zip(model_pred, model_obs)):
        ax.annotate(f'n={model_counts[i]:,}', (x, y), xytext=(0, 7), 
                    textcoords='offset points', ha='center', fontsize=8)
    
    # Calculate MSE for each approach
    model_mse = np.mean((np.array(model_pred) - np.array(model_obs))**2)
    calib_mse = np.mean((np.array(calib_pred) - np.array(calib_obs))**2)
    pce_mse = np.mean((np.array(pce_pred) - np.array(pce_obs))**2)
    
    # Add metrics to plot
    ax.text(0.05, 0.95, 
             f"Original Model MSE: {model_mse:.6f}\n"
             f"Isotonic Calibrated MSE: {calib_mse:.6f}\n"
             f"PCE MSE: {pce_mse:.6f}\n"
             f"Improvement: {(model_mse - calib_mse)/model_mse*100:.1f}%",
             transform=ax.transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlabel('Predicted 10-year ASCVD Risk')
    ax.set_ylabel('Observed 10-year ASCVD Rate')
    ax.set_title('10-Year ASCVD Risk with Isotonic 1-Year Calibration')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Set limits with some padding
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Return results for further analysis
    results = {
        'model_risks': all_model_risks,
        'calibrated_risks': all_calibrated_risks,
        'pce_risks': all_pce_risks,
        'outcomes': all_actual_outcomes,
        'patient_ids': patient_ids,
        'model_mse': model_mse,
        'calib_mse': calib_mse,
        'pce_mse': pce_mse
    }
    
    return fig, results

# Usage (assuming calibrator_1yr is already trained):
# fig, results = plot_10year_with_isotonic_1year_calibration(model, pce_df, calibrator_1yr)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score

def plot_ascvd_risk_comparison(n_batches=40, batch_size=10000, ascvd_indices=[111, 112, 113, 114, 115, 116]):
    """
    Create a plot comparing model predictions to smoothed prevalence-based risk by age
    """
    # Initialize arrays to store aggregated data
    all_pi_values = []
    
    # First, try to get prevalence from the first model
    first_model_path = f'/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_{batch_size}/model.pt'
    try:
        first_model = torch.load(first_model_path)
        if 'prevalence_t' in first_model:
            # Check if it's already a numpy array
            if isinstance(first_model['prevalence_t'], np.ndarray):
                mu_dt = first_model['prevalence_t']
            else:
                mu_dt = first_model['prevalence_t'].numpy()
            print(f"Loaded prevalence_t from first model, shape: {mu_dt.shape}")
        else:
            print("prevalence_t not found in first model")
            return None
    except FileNotFoundError:
        print(f"Warning: Could not find first model file")
        return None
    
    print("Processing batches...")
    
    # Process all batches to get predictions
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size
        model_path = f'/Users/sarahurbut/Dropbox/resultshighamp/results/output_{start_idx}_{end_idx}/model.pt'
        
        try:
            # Load model
            model = torch.load(model_path)
            phi_prob = torch.sigmoid(model['model_state_dict']['phi']).detach().numpy()
            theta = torch.softmax(model['model_state_dict']['lambda_'], dim=1).detach().numpy()
            kappa = model['model_state_dict']['kappa'].item()
            
            # Calculate predictions with kappa
            pi = np.einsum('nkt,kdt->ndt', theta, phi_prob) * kappa
            pi = np.clip(pi, 1e-8, 1-1e-8)
            
            all_pi_values.append(pi)
            
            print(f"Processed batch {batch} for predictions")
            
        except FileNotFoundError:
            print(f"Warning: Could not find model file for batch {batch}")
            continue
    
    # Combine all predictions
    all_pi = np.concatenate(all_pi_values, axis=0)
    print(f"Combined predictions shape: {all_pi.shape}")
    
    # Get mean predictions across patients for each ASCVD condition and timepoint
    pred_risk = all_pi[:, ascvd_indices, :].mean(axis=0)  # Shape: [6 diseases, T timepoints]
    print(f"Mean ASCVD predictions shape: {pred_risk.shape}")
    
    # Get prevalence for ASCVD conditions
    if torch.is_tensor(first_model['prevalence_t']):
        mu_dt_ascvd = first_model['prevalence_t'][ascvd_indices].cpu().numpy()
    else:
        mu_dt_ascvd = first_model['prevalence_t'][ascvd_indices]
    
    # Smoothed scaling factor
    ascvd_scale_smooth = np.mean(mu_dt_ascvd) / np.mean(pred_risk)
    print(f"Smoothed ASCVD scaling factor: {ascvd_scale_smooth:.4f}")
    
    # Apply scaling to get calibrated predictions
    ascvd_preds = all_pi[:, ascvd_indices, :]
    ascvd_preds_calibrated = ascvd_preds * ascvd_scale_smooth
    
    # Get patient ages
    # For simplicity, we'll create a range of ages
    n_patients = all_pi.shape[0]
    common_ids = np.arange(n_patients)
    
    # Calculate 10-year risks
    model_risks = calculate_ten_year_risks(ascvd_preds_calibrated, common_ids, max_age=70)
    
    # Calculate mean, 25th, and 75th percentiles by age
    mean_risks = model_risks.mean(axis=0)
    p25_risks = np.percentile(model_risks.values, 25, axis=0)
    p75_risks = np.percentile(model_risks.values, 75, axis=0)
    
    # Calculate prevalence-based 10-year risk
    prevalence_risks = calculate_prevalence_10yr_risk(mu_dt_ascvd)
    
    # Calculate R-squared between mean model predictions and prevalence-based risk
    r2 = r2_score(prevalence_risks, mean_risks)
    print(f"R² between model predictions and prevalence-based risk: {r2:.4f}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot model predictions
    plt.plot(range(40, 71), mean_risks, color='blue', linestyle='-', linewidth=2, label='Model predictions (mean)')
    plt.plot(range(40, 71), p25_risks, color='orange', linestyle='--', linewidth=1.5, label='Model 25.0th percentile')
    plt.plot(range(40, 71), p75_risks, color='green', linestyle='--', linewidth=1.5, label='Model 75.0th percentile')
    
    # Plot prevalence-based risk
    plt.plot(range(40, 71), prevalence_risks, color='red', linestyle='-', linewidth=2, label='Prevalence-based risk')
    
    plt.xlabel('Age')
    plt.ylabel('10-year Risk (%)')
    plt.title(f'Model 10-year Risk Predictions vs Prevalence-based Risk (ASCVD calib={ascvd_scale_smooth:.2f}, R²={r2:.4f})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 15)
    
    plt.tight_layout()
    plt.savefig(f'/Users/sarahurbut/aladynoulli2/pyScripts/figures_for_science/figure5/ascvd_risk_comparison.pdf',dpi=300)
    
    return mean_risks, p25_risks, p75_risks, prevalence_risks, ascvd_scale_smooth, r2

def calculate_ten_year_risks(ascvd_preds, common_ids, max_age=70):
    """
    Calculate 10-year ASCVD risks for each patient at each age
    ascvd_preds starts at age 30, we want risks for ages 40-max_age
    """
    n_patients = len(common_ids)
    n_ages = max_age - 40 + 1  # e.g., ages 40-70 inclusive
    ten_year_risks = np.zeros((n_patients, n_ages))
    
    age_offset = 10  # Offset because predictions start at age 30
    
    for age_idx in range(n_ages):
        # For each starting age (40-70), look at next 10 years
        start_idx = age_idx + age_offset
        time_window = slice(start_idx, start_idx + 10)
        
        # Get all predictions for this 10-year window
        window_preds = ascvd_preds[:, :, time_window]
        
        # Probability of surviving (no events)
        survival_probs = 1 - window_preds
        
        # Probability of surviving all diseases for all years
        total_survival = np.prod(survival_probs, axis=(1,2))
        
        # 10-year risk is probability of not surviving
        ten_year_risks[:, age_idx] = 1 - total_survival
    
    risk_df = pd.DataFrame(ten_year_risks * 100,
                          index=common_ids, 
                          columns=range(40, max_age + 1))
    
    return risk_df

def calculate_prevalence_10yr_risk(mu_dt_ascvd):
    """
    Calculate 10-year risk from smoothed prevalence rates
    """
    n_ages = 31  # Ages 40-70
    ten_year_risks = np.zeros(n_ages)
    
    age_offset = 10  # Offset because predictions start at age 30
    
    for age_idx in range(n_ages):
        # For each starting age (40-70), look at next 10 years
        start_idx = age_idx + age_offset
        time_window = slice(start_idx, start_idx + 10)
        
        # Get prevalence for this 10-year window
        window_prev = mu_dt_ascvd[:, time_window]
        
        # For each year, calculate P(any ASCVD)
        yearly_any_ascvd = np.zeros(10)
        for year in range(10):
            # Probability of no disease for each condition this year
            no_disease_probs = 1 - window_prev[:, year]
            # Probability of no ASCVD this year
            no_ascvd = np.prod(no_disease_probs)
            # Probability of any ASCVD this year
            yearly_any_ascvd[year] = 1 - no_ascvd
        
        # Calculate 10-year survival probability
        ten_year_survival = np.prod(1 - yearly_any_ascvd)
        
        # 10-year risk
        ten_year_risks[age_idx] = (1 - ten_year_survival) * 100
    
    return ten_year_risks

# Usage:
# mean_risks, p25_risks, p75_risks, prevalence_risks, ascvd_scale, r2 = plot_ascvd_risk_comparison(n_batches=40)



def compare_with_pce(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using single timepoint prediction
    """
    our_10yr_risks = []
    actual_10yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Get mean risks across patients for calibration
    predicted_risk_2d = pi.mean(axis=0)  # Shape: [D, T]
    observed_risk_2d = model.Y.numpy().mean(axis=0)  # Shape: [D, T]
    
    # Sort and get LOESS calibration curve
    pred_flat = predicted_risk_2d.flatten()
    obs_flat = observed_risk_2d.flatten()
    sort_idx = np.argsort(pred_flat)
    smoothed = lowess(obs_flat[sort_idx], pred_flat[sort_idx], frac=0.3)
    
    # Apply calibration to all predictions using interpolation
    pi_calibrated = np.interp(pi.flatten(), smoothed[:, 0], smoothed[:, 1]).reshape(pi.shape)
    
    # Calculate 10-year risks using only enrollment time prediction
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 10 >= model.T:
            continue
            
        # Only use predictions at enrollment time
        pi_ascvd = pi_calibrated[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk first
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        
        # Convert to 10-year risk
        risk = 1 - (1 - yearly_risk)**10
        our_10yr_risks.append(risk)
        
        # Still look at actual events over 10 years
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+10]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_10yr.append(actual.item())
   
    # Rest of the function remains the same
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_risks = pce_df['pce_goff_fuull'].values[:len(our_10yr_risks)]
    
    # Calculate ROC AUCs
    our_auc = roc_auc_score(actual_10yr, our_10yr_risks)
    pce_auc = roc_auc_score(actual_10yr, pce_risks)
    
    print(f"\nROC AUC Comparison (10-year prediction from enrollment):")
    print(f"Our model: {our_auc:.3f}")
    print(f"PCE: {pce_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_10yr, our_10yr_risks, label=f'Our Model (AUC={our_auc:.3f})')
    plot_roc_curve(actual_10yr, pce_risks, label=f'PCE (AUC={pce_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()



def compare_with_prevent(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using single timepoint prediction
    """
    our_10yr_risks = []
    actual_10yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Get mean risks across patients for calibration
    predicted_risk_2d = pi.mean(axis=0)  # Shape: [D, T]
    observed_risk_2d = model.Y.numpy().mean(axis=0)  # Shape: [D, T]
    
    # Sort and get LOESS calibration curve
    pred_flat = predicted_risk_2d.flatten()
    obs_flat = observed_risk_2d.flatten()
    sort_idx = np.argsort(pred_flat)
    smoothed = lowess(obs_flat[sort_idx], pred_flat[sort_idx], frac=0.3)
    
    # Apply calibration to all predictions using interpolation
    pi_calibrated = np.interp(pi.flatten(), smoothed[:, 0], smoothed[:, 1]).reshape(pi.shape)
    
    # Calculate 10-year risks using only enrollment time prediction
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 10 >= model.T:
            continue
            
        # Only use predictions at enrollment time
        pi_ascvd = pi_calibrated[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk first
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        
        # Convert to 10-year risk
        risk = 1 - (1 - yearly_risk)**10
        our_10yr_risks.append(risk)
        
        # Still look at actual events over 10 years
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+10]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_10yr.append(actual.item())
   
    # Rest of the function remains the same
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_risks = pce_df['prevent_impute'].values[:len(our_10yr_risks)]
    
    # Calculate ROC AUCs
    our_auc = roc_auc_score(actual_10yr, our_10yr_risks)
    pce_auc = roc_auc_score(actual_10yr, pce_risks)
    
    print(f"\nROC AUC Comparison (10-year prediction from enrollment):")
    print(f"Our model: {our_auc:.3f}")
    print(f"PREVENT: {pce_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_10yr, our_10yr_risks, label=f'Our Model (AUC={our_auc:.3f})')
    plot_roc_curve(actual_10yr, pce_risks, label=f'PREVENT (AUC={pce_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_observed_rates(Y, disease_indices=None, disease_names=None):
    """
    Analyze and plot observed event rates from Y matrix
    
    Args:
        Y: Observed events tensor/array (shape: N x D x T)
        disease_indices: List of disease indices to analyze (optional)
        disease_names: List of disease names corresponding to indices (optional)
    """
    # Convert to numpy if tensor
    Y_np = Y.detach().numpy() if torch.is_tensor(Y) else Y
    
    n_patients, n_diseases, n_timepoints = Y_np.shape
    print(f"Data shape: {n_patients} patients, {n_diseases} diseases, {n_timepoints} timepoints")
    
    # If no indices provided, look at first few diseases
    if disease_indices is None:
        disease_indices = list(range(min(5, n_diseases)))
    
    if disease_names is None:
        disease_names = [f"Disease {i}" for i in disease_indices]
    
    # Calculate event rates for each disease and timepoint
    event_rates = np.nanmean(Y_np, axis=0)  # Shape: D x T
    
    # Create plot
    plt.figure(figsize=(15, 8))
    
    # Plot event rates for selected diseases
    for idx, disease_idx in enumerate(disease_indices):
        rates = event_rates[disease_idx]
        plt.plot(range(30, 30 + n_timepoints), rates, 
                label=f"{disease_names[idx]} (mean={np.nanmean(rates):.6f})")
    
    plt.xlabel('Age')
    plt.ylabel('Event Rate')
    plt.title('Observed Event Rates by Age')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"{'Disease':30} {'Mean Rate':>12} {'Total Events':>12} {'Non-zero Ages':>12}")
    print("-" * 70)
    
    for idx, disease_idx in enumerate(disease_indices):
        rates = event_rates[disease_idx]
        total_events = np.nansum(Y_np[:, disease_idx])
        non_zero_ages = np.sum(rates > 0)
        
        print(f"{disease_names[idx]:30} {np.nanmean(rates):12.6f} {int(total_events):12d} {non_zero_ages:12d}")
    
    plt.show()
    
    return event_rates[disease_indices]



def create_traditional_calibration_plot2(model, mu_dt, n_bins=10):
    """
    Create a traditional calibration plot with binned risk predictions
    for 1-year risks.
    
    Args:
        model: The loaded model with internal kappa
        mu_dt: The observed prevalence matrix (shape: D x T)
        n_bins: Number of bins to use (default: 10 for deciles)
        
    Returns:
        Calibration plot figure
    """
    # Get predictions from the model
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
    
    # Convert to numpy for analysis
    pi_np = pi.detach().numpy()  # Shape: (N, D, T)
    
    # Flatten predictions and observations to 1D arrays
    all_predictions = []
    all_observations = []
    
    n_diseases, n_timepoints = mu_dt.shape
    
    # For each disease and timepoint, collect predictions and observations
    for d in range(n_diseases):
        for t in range(n_timepoints):
            # Skip if prevalence is NaN
            if np.isnan(mu_dt[d, t]):
                continue
                
            # Get predictions for this disease-timepoint
            pred_dt = pi_np[:, d, t]
            
            # Get actual observations from Y if available, otherwise use mu_dt
            if hasattr(model, 'Y'):
                obs_dt = model.Y[:, d, t].detach().numpy()
            else:
                # Use prevalence as proxy for observations
                obs_dt = np.random.binomial(1, mu_dt[d, t], size=len(pred_dt))
            
            # Add to our collections
            all_predictions.extend(pred_dt)
            all_observations.extend(obs_dt)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_observations = np.array(all_observations)
    
    # Create bins based on predictions
    bin_edges = np.percentile(all_predictions, np.linspace(0, 100, n_bins+1))
    
    # Ensure unique bin edges (can happen with very skewed distributions)
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1
    
    # Initialize arrays for bin statistics
    bin_pred_means = np.zeros(n_bins)
    bin_obs_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_lower_bounds = np.zeros(n_bins)
    bin_upper_bounds = np.zeros(n_bins)
    
    # Calculate statistics for each bin
    for i in range(n_bins):
        if i == n_bins - 1:
            # Include the right edge in the last bin
            mask = (all_predictions >= bin_edges[i]) & (all_predictions <= bin_edges[i+1])
        else:
            mask = (all_predictions >= bin_edges[i]) & (all_predictions < bin_edges[i+1])
        
        bin_pred_means[i] = np.mean(all_predictions[mask])
        bin_obs_means[i] = np.mean(all_observations[mask])
        bin_counts[i] = np.sum(mask)
        
        # Calculate 95% confidence intervals using Wilson score interval
        if bin_counts[i] > 0:
            n = bin_counts[i]
            p = bin_obs_means[i]
            z = 1.96  # 95% confidence
            
            # Wilson score interval
            denominator = 1 + z**2/n
            center = (p + z**2/(2*n))/denominator
            halfwidth = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))/denominator
            
            bin_lower_bounds[i] = max(0, center - halfwidth)
            bin_upper_bounds[i] = min(1, center + halfwidth)
        else:
            bin_lower_bounds[i] = 0
            bin_upper_bounds[i] = 0
    
    # Create the calibration plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the calibration curve
    ax.plot(bin_pred_means, bin_obs_means, 'o-', markersize=8, label='Calibration curve')
    
    # Add error bars for 95% confidence intervals
    ax.errorbar(bin_pred_means, bin_obs_means, 
                yerr=[bin_obs_means - bin_lower_bounds, bin_upper_bounds - bin_obs_means],
                fmt='none', capsize=5, color='blue', alpha=0.5)
    
    # Add perfect calibration line
    ax.plot([0, max(bin_pred_means)*1.1], [0, max(bin_pred_means)*1.1], 'k--', label='Perfect calibration')
    
    # Add bin sizes as text
    for i in range(n_bins):
        ax.annotate(f'n={int(bin_counts[i]):,}', 
                   (bin_pred_means[i], bin_obs_means[i]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
    
    # Calculate metrics
    mse = np.mean((bin_pred_means - bin_obs_means)**2)
    
    # Add metrics to plot
    ax.text(0.05, 0.95, 
            f"MSE: {mse:.6f}\nNumber of bins: {n_bins}\nTotal observations: {len(all_observations)}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlabel('Predicted Risk')
    ax.set_ylabel('Observed Event Rate')
    ax.set_title('Traditional Calibration Plot (1-year risks)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Set limits with some padding
    max_val = max(max(bin_pred_means), max(bin_obs_means)) * 1.1
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Calibration plot created with {n_bins} bins")
    print(f"MSE between predicted and observed: {mse:.6f}")
    
    return fig, bin_pred_means, bin_obs_means, bin_counts



# utils.py
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import pandas as pd # Make sure pandas is imported

# ... (other functions) ...

# ADDED follow_up_duration_years parameter
def evaluate_major_diseases_wsex(model, Y_100k, E_100k, disease_names, pce_df, follow_up_duration_years=10):
    """
    Evaluate model performance on major diseases for OUTCOMES within a specified follow-up duration, 
    using 1-YEAR RISK at enrollment as the score for AUC calculation.
    Handles sex-specific diseases correctly using pce_df.
    Calculates n_events and event_rate based *only* on the evaluated cohort (pce_df/Y_100k).
    Uses integer positional indices consistently after filtering.
    
    Parameters:
    - model: trained model
    - Y_100k: disease status matrix (PyTorch tensor) used for AUC calculation cohort
    - E_100k: event times matrix (PyTorch tensor) - currently unused but kept for signature consistency
    - disease_names: list of disease names
    - pce_df: DataFrame with patient characteristics including 'age' and 'Sex' (must match Y_100k rows)
    - follow_up_duration_years: Integer, the number of years after enrollment to check for outcomes (default: 10)
    """
    major_diseases = {
        'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                  'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus', 'Cancer of bronchus; lung', 'Breast cancer [female]', 'Malignant neoplasm of female breast', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver', 'Secondary malignancy of bone'],
        'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
        'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'], # Sex-specific
        'Prostate_Cancer': ['Cancer of prostate'], # Sex-specific
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
    }

    # --- Input Validation ---
    if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")
    if not isinstance(follow_up_duration_years, int) or follow_up_duration_years <= 0:
        raise ValueError("'follow_up_duration_years' must be a positive integer.")

    with torch.no_grad():
        pi, _, _ = model.forward()
        
    N_pi = pi.shape[0]
    N_pce = len(pce_df)
    N_y100k = Y_100k.shape[0]
    
    # --- Ensure alignment and reset index ---
    if not (N_pi == N_pce == N_y100k):
        print(f"Warning: Size mismatch for evaluation cohort. pi: {N_pi}, pce_df: {N_pce}, Y_100k: {N_y100k}. Using minimum size.")
        min_N = min(N_pi, N_pce, N_y100k)
        pi = pi[:min_N]
        pce_df = pce_df.iloc[:min_N]
        Y_100k = Y_100k[:min_N]
    pce_df = pce_df.reset_index(drop=True) 

    results = {}
    
    # --- Main Loop ---
    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group} ({follow_up_duration_years}-Year Outcome, 1-Year Score)...") # Updated printout
        
        # --- Get Disease Indices (same logic, including bounds check) ---
        disease_indices = []
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                 if idx not in unique_indices:
                      disease_indices.append(idx)
                      unique_indices.add(idx)
        max_model_disease_idx = pi.shape[1] - 1
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
             print(f"No valid matching disease indices found for {disease_group}.")
             results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0}
             continue

        # --- Sex Filtering (using pce_df only) ---
        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'

        # Get integer positions based on mask
        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]

        if target_sex:
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} individuals in cohort")
            if len(int_indices_pce) == 0:
                 print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                 results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0}
                 continue
        
        # --- Calculate AUC & Events/Rate (using integer positions) ---
        if len(int_indices_pce) == 0: 
            auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
        else:
            # Slice tensors and DataFrame using the integer positions
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce] 
            current_N_auc = len(int_indices_pce)

            risks_auc = torch.zeros(current_N_auc, device=pi.device) # 1-year risk score
            outcomes_auc = torch.zeros(current_N_auc, device=pi.device) # Outcome over follow_up_duration_years
            processed_indices_auc_final = [] 

            n_prevalent_excluded = 0
            for i in range(current_N_auc):
                age = current_pce_df_auc.iloc[i]['age']
                t_enroll = int(age - 30)
                if t_enroll < 0 or t_enroll >= current_pi_auc.shape[2]: continue

                # INCIDENT DISEASE FILTER: Only for single-disease outcomes
                if len(disease_indices) == 1:
                    prevalent = False
                    for d_idx in disease_indices:
                        if d_idx >= current_Y_100k_auc.shape[1]:
                            continue
                        if torch.any(current_Y_100k_auc[i, d_idx, :t_enroll] > 0):
                            prevalent = True
                            break
                    if prevalent:
                        n_prevalent_excluded += 1
                        continue  # Skip this patient for this disease group
                # ... rest of your code ...

                age_enroll = t_enroll + 30  # or however you define enrollment age
                end_time = min(t_enroll + follow_up_duration_years, current_Y_100k_auc.shape[2])

                # Default: censored at end_time
                age_at_event = end_time + 30 - 1  # or however you define age at end
                event = 0

                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]:
                        continue
                    # Find the first event time for this disease
                    event_times = torch.where(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0)[0]
                    if len(event_times) > 0:
                        this_event_age = t_enroll + event_times[0].item() + 30  # adjust for age
                        if this_event_age < age_at_event:
                            age_at_event = this_event_age
                            event = 1

                pi_diseases = current_pi_auc[i, disease_indices, t_enroll]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                risks_auc[i] = yearly_risk # Use 1-year risk as score

                # --- Outcome: Check event in next follow_up_duration_years ---
                # MODIFIED end_time calculation
                end_time = min(t_enroll + follow_up_duration_years, current_Y_100k_auc.shape[2]) 
                if end_time <= t_enroll: continue
                
                event_found_auc = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0): 
                        outcomes_auc[i] = 1
                        event_found_auc = True
                        break
                processed_indices_auc_final.append(i) 

            if not processed_indices_auc_final:
                 auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
            else:
                 # Calculate based on processed individuals
                 risks_np = risks_auc[processed_indices_auc_final].cpu().numpy()
                 outcomes_np = outcomes_auc[processed_indices_auc_final].cpu().numpy()
                 n_processed = len(outcomes_np)
                 if disease_group in ["Bipolar_Disorder", "Depression"]:
                    df = pd.DataFrame({
                        "risk": risks_np,
                        "outcome": outcomes_np
                    })
                    df.to_csv(f"debug_{disease_group}.csv", index=False)
                    
                    # Calculate AUC
                 if len(np.unique(outcomes_np)) > 1:
                      fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
                      auc_score = auc(fpr, tpr)
                 else:
                      auc_score = np.nan
                      print(f"Warning: Only one class present ({np.unique(outcomes_np)}) for AUC.")
                 
                 # Calculate Events and Rate based on this cohort
                 n_events = int(np.sum(outcomes_np))
                 event_rate = (n_events / n_processed * 100) if n_processed > 0 else 0.0
        
        # Store results
        results[disease_group] = {
            'auc': auc_score,
            'n_events': n_events,
            'event_rate': event_rate
        }
        
        # Updated printout
        print(f"AUC (Score: 1-Yr Risk, Outcome: {follow_up_duration_years}-Yr Event): {auc_score if not np.isnan(auc_score) else 'N/A'} (calculated on {n_processed} individuals)") 
        print(f"Events ({follow_up_duration_years}-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {n_processed} individuals)") 
        print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")

    # Updated printout
    print(f"\nSummary of Results (Prospective {follow_up_duration_years}-Year Outcome, 1-Year Score, Sex-Adjusted):") 
    print("-" * 60)
    print(f"{'Disease Group':<20} {'AUC':<8} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 60)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f}" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<8} {res['n_events']:<10d} {rate_str}")
    print("-" * 60)

    return results

def evaluate_major_diseases_wsex_with_bootstrap(model, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, follow_up_duration_years=10):
    """
    Same as evaluate_major_diseases_wsex but adds bootstrap CIs for AUC.
    Uses exact same time logic and event counting as original.
    Also prints sex-stratified AUCs (except for sex-specific diseases) and ASCVD AUCs for patients with pre-existing RA or breast cancer.
    """

    major_diseases = {
        'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                  'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': ['Colon cancer', 'Cancer of bronchus; lung', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm','Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver'],
        'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
        'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],# Sex-specific
        'Prostate_Cancer': ['Cancer of prostate'], # Sex-specific
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
    }

    # For ASCVD analysis with pre-existing conditions
    pre_existing_conditions = {
        'RA': ['Rheumatoid arthritis'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast']
    }

    results = {}
    
    if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")
    
    with torch.no_grad():
        pi, _, _ = model.forward()
        
    N_pi = pi.shape[0]
    N_pce = len(pce_df)
    N_y100k = Y_100k.shape[0]
    
    if not (N_pi == N_pce == N_y100k):
        print(f"Warning: Size mismatch for evaluation cohort. pi: {N_pi}, pce_df: {N_pce}, Y_100k: {N_y100k}. Using minimum size.")
        min_N = min(N_pi, N_pce, N_y100k)
        pi = pi[:min_N]
        pce_df = pce_df.iloc[:min_N]
        Y_100k = Y_100k[:min_N]
    pce_df = pce_df.reset_index(drop=True) 

    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group} ({follow_up_duration_years}-Year Outcome, 1-Year Score)...")
        
        disease_indices = []
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                 if idx not in unique_indices:
                      disease_indices.append(idx)
                      unique_indices.add(idx)
        max_model_disease_idx = pi.shape[1] - 1
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
             print(f"No valid matching disease indices found for {disease_group}.")
             results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 
                                     'ci_lower': np.nan, 'ci_upper': np.nan}
             continue

        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'

        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]

        if target_sex:
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} individuals in cohort")
            if len(int_indices_pce) == 0:
                 print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                 results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0,
                                         'ci_lower': np.nan, 'ci_upper': np.nan}
                 continue
        
        if len(int_indices_pce) == 0: 
            auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
            ci_lower = np.nan; ci_upper = np.nan
        else:
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce] 
            current_N_auc = len(int_indices_pce)

            # Use lists for all per-patient results
            risks_auc = []
            outcomes_auc = []
            age_enrolls = []
            age_at_events = []
            event_indicators = []
            n_prevalent_excluded = 0

            # (ASCVD pre-existing logic stays as before)

            for i in range(current_N_auc): 
                age = current_pce_df_auc.iloc[i]['age'] 
                t_enroll = int(age - 30)
                if t_enroll < 0 or t_enroll >= current_pi_auc.shape[2]: continue

                # INCIDENT DISEASE FILTER: Only for single-disease outcomes
                if len(disease_indices) == 1:
                    prevalent = False
                    for d_idx in disease_indices:
                        if d_idx >= current_Y_100k_auc.shape[1]:
                            continue
                        if torch.any(current_Y_100k_auc[i, d_idx, :t_enroll] > 0):
                            prevalent = True
                            break
                    if prevalent:
                        n_prevalent_excluded += 1
                        continue  # Skip this patient for this disease group

                pi_diseases = current_pi_auc[i, disease_indices, t_enroll]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                end_time = min(t_enroll + follow_up_duration_years, current_Y_100k_auc.shape[2]) 
                if end_time <= t_enroll: continue

                # --- C-index: Find time-to-event and event indicator ---
                age_enroll = t_enroll + 30
                age_at_event = end_time + 30 - 1
                event = 0
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    event_times = torch.where(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0)[0]
                    if len(event_times) > 0:
                        this_event_age = t_enroll + event_times[0].item() + 30
                        if this_event_age < age_at_event:
                            age_at_event = this_event_age
                            event = 1

                # --- Outcome: Check event in next follow_up_duration_years ---
                outcome = 0
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0): 
                        outcome = 1
                        break

                # Only add to lists if included
                risks_auc.append(yearly_risk.item() if hasattr(yearly_risk, 'item') else float(yearly_risk))
                outcomes_auc.append(outcome)
                age_enrolls.append(age_enroll)
                age_at_events.append(age_at_event)
                event_indicators.append(event)

            n_processed = len(risks_auc)
            if n_processed == 0:
                auc_score = np.nan; n_events = 0; event_rate = 0.0; ci_lower = np.nan; ci_upper = np.nan; c_index = np.nan
            else:
                risks_np = np.array(risks_auc)
                outcomes_np = np.array(outcomes_auc)
                age_enrolls_np = np.array(age_enrolls)
                age_at_events_np = np.array(age_at_events)
                event_indicators_np = np.array(event_indicators)
                durations = age_at_events_np - age_enrolls_np
                from lifelines.utils import concordance_index
                try:
                    c_index = concordance_index(durations, -risks_np, event_indicators_np)
                except Exception as e:
                    print(f"C-index calculation failed: {e}")
                    c_index = np.nan
                if len(np.unique(outcomes_np)) > 1:
                    fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
                    auc_score = auc(fpr, tpr)
                    aucs = []
                    for _ in range(n_bootstraps):
                        indices = np.random.choice(len(risks_np), size=len(risks_np), replace=True)
                        if len(np.unique(outcomes_np[indices])) > 1:
                            fpr_boot, tpr_boot, _ = roc_curve(outcomes_np[indices], risks_np[indices])
                            bootstrap_auc = auc(fpr_boot, tpr_boot)
                            aucs.append(bootstrap_auc)
                    if aucs:
                        ci_lower = np.percentile(aucs, 2.5)
                        ci_upper = np.percentile(aucs, 97.5)
                    else:
                        ci_lower = ci_upper = np.nan
                else:
                    auc_score = np.nan
                    ci_lower = ci_upper = np.nan
                    print(f"Warning: Only one class present ({np.unique(outcomes_np)}) for AUC.")
                n_events = int(np.sum(outcomes_np))
                event_rate = (n_events / n_processed * 100) if n_processed > 0 else 0.0
        results[disease_group] = {
            'auc': auc_score,
            'n_events': n_events,
            'event_rate': event_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'c_index': c_index
        }
        print(f"AUC: {auc_score:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) (calculated on {n_processed} individuals)")
        print(f"Events (10-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {n_processed} individuals)")
        print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
    print(f"\nSummary of Results (Prospective {follow_up_duration_years}-Year Outcome, 1-Year Score, Sex-Adjusted):")
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str}")
    print("-" * 80)
    return results

# --- Function to Fit Cox Models on Training Slice ---
def fit_cox_baseline_models(Y_full, FH_processed, train_indices, disease_mapping, major_diseases, disease_names, follow_up_duration_years=10):
    """Fits Cox proportional hazards models using age as time scale."""
    from lifelines import CoxPHFitter
    fitted_models = {}
    print(f"Fitting Cox models using training indices [{train_indices.min()}:{train_indices.max()+1}]...")
    
    try:
        Y_train = Y_full[train_indices]
        FH_train = FH_processed.iloc[train_indices].reset_index(drop=True)
    except IndexError as e:
        raise IndexError(f"Training indices out of bounds for Y_full/FH_processed. Error: {e}") from e
    
    for disease_group, disease_names_list in major_diseases.items():
        fh_cols = disease_mapping.get(disease_group, [])
        if not fh_cols: print(f" - {disease_group}: No FH columns, fitting Sex only.")
        print(f" - Fitting {disease_group}...")
        
        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1

        if target_sex_code is not None:
            mask_train = (FH_train['sex'] == target_sex_code)
        else:
            mask_train = pd.Series(True, index=FH_train.index)
        
        current_FH_train = FH_train[mask_train].copy()
        current_Y_train = Y_train[mask_train]
        
        if len(current_FH_train) == 0:
            print(f"   Warning: No individuals for target sex code {target_sex_code} in training slice.")
            fitted_models[disease_group] = None
            continue
        
        # Find disease indices
        disease_indices = []
        unique_indices = set()
        for disease in disease_names_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_full.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        
        if not disease_indices:
            fitted_models[disease_group] = None
            continue
        
        # Prepare data for Cox model
        cox_data = []
        n_prevalent_excluded = 0  # Counter for excluded prevalent cases
        for i in range(len(current_FH_train)):
            age_at_enrollment = current_FH_train.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)
            if t_enroll < 0 or t_enroll >= current_Y_train.shape[2]:
                continue
            
            # Exclude prevalent cases for single-disease groups only
            if len(disease_indices) == 1:
                d_idx = disease_indices[0]
                if torch.any(current_Y_train[i, d_idx, :t_enroll] > 0):
                    n_prevalent_excluded += 1
                    continue  # skip prevalent case
            
            end_time = min(t_enroll + follow_up_duration_years, current_Y_train.shape[2])
            if end_time <= t_enroll:
                continue
            
            # --- AGGREGATE ACROSS DISEASES: one row per person ---
            event_found = False
            earliest_event_time = None
            for d_idx in disease_indices:
                Y_slice = current_Y_train[i, d_idx, t_enroll:end_time]
                if (torch.is_tensor(Y_slice) and torch.any(Y_slice > 0)) or \
                (isinstance(Y_slice, np.ndarray) and np.any(Y_slice > 0)):
                    this_event_time = np.where(Y_slice > 0)[0][0] + t_enroll
                    if (earliest_event_time is None) or (this_event_time < earliest_event_time):
                        earliest_event_time = this_event_time
                    event_found = True
            if event_found:
                age_at_event = 30 + earliest_event_time  # Convert to actual age
                event = 1
            else:
                age_at_event = 30 + end_time - 1  # Convert to actual age
                event = 0

            row = {
                'age_enroll': age_at_enrollment,  # Age at enrollment (entry)
                'age': age_at_event,              # Age at event/censoring (exit)
                'event': event,
                'sex': current_FH_train.iloc[i]['sex']
            }
            # Add family history indicators
            if fh_cols:
                valid_fh_cols = [col for col in fh_cols if col in current_FH_train.columns]
                if valid_fh_cols:
                    row['fh'] = current_FH_train.iloc[i][valid_fh_cols].any()
            cox_data.append(row)

        
        if not cox_data:
            fitted_models[disease_group] = None
            continue
            
        cox_df = pd.DataFrame(cox_data)

        print(cox_df.head())
        print(cox_df.shape)
        problem_rows = cox_df[cox_df['age_enroll'] >= cox_df['age']]
        print(problem_rows)
        print(f"Number of rows with entry >= duration: {problem_rows.shape[0]}")
            # After building cox_df
        mask = cox_df['age_enroll'] >= cox_df['age']
        cox_df.loc[mask, 'age'] = cox_df.loc[mask, 'age_enroll'] + 1.0  # 
        problem_rows = cox_df[cox_df['age_enroll'] >= cox_df['age']]
        print(problem_rows)
        print(f"Number of rows with entry >= duration: {problem_rows.shape[0]}")

          # Check if we have enough events
        if cox_df['event'].sum() < 5:  # Require at least 5 events
            print(f"   Warning: Too few events ({cox_df['event'].sum()}) for {disease_group}")
            fitted_models[disease_group] = None
            continue
        
        try:
            # Fit Cox model using formula interface
            cph = CoxPHFitter()
            
            # Build formula based on available covariates
            formula = 'sex'
            if 'fh' in cox_df.columns:
                formula += ' + fh'

            if target_sex_code is not None:
                formula = 'fh' if 'fh' in cox_df.columns else '1'  # Use intercept-only model
            print(formula)
            
            cph.fit(cox_df, duration_col='age', event_col='event', entry_col='age_enroll', formula=formula)
            fitted_models[disease_group] = cph
            print(f"   Model fitted for {disease_group} using {len(cox_df)} samples.")
            if len(disease_indices) == 1:
                print(f"   Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
        except Exception as e:
            print(f"   Error fitting {disease_group}: {e}")
            fitted_models[disease_group] = None
    
    print("Finished fitting Cox models.")
    return fitted_models


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def fit_glm_baseline_models(Y_full, FH_processed, train_indices, disease_mapping, major_diseases, disease_names, follow_up_duration_years=10):
    """Fits Cox proportional hazards models using age as time scale."""
    from lifelines import CoxPHFitter
    fitted_models = {}
    print(f"Fitting Cox models using training indices [{train_indices.min()}:{train_indices.max()+1}]...")
    
    try:
        Y_train = Y_full[train_indices]
        FH_train = FH_processed.iloc[train_indices].reset_index(drop=True)
    except IndexError as e:
        raise IndexError(f"Training indices out of bounds for Y_full/FH_processed. Error: {e}") from e
    
    for disease_group, disease_names_list in major_diseases.items():
        fh_cols = disease_mapping.get(disease_group, [])
        if not fh_cols: print(f" - {disease_group}: No FH columns, fitting Sex only.")
        print(f" - Fitting {disease_group}...")
        
        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1

        if target_sex_code is not None:
            mask_train = (FH_train['sex'] == target_sex_code)
        else:
            mask_train = pd.Series(True, index=FH_train.index)
        
        current_FH_train = FH_train[mask_train].copy()
        current_Y_train = Y_train[mask_train]
        
        if len(current_FH_train) == 0:
            print(f"   Warning: No individuals for target sex code {target_sex_code} in training slice.")
            fitted_models[disease_group] = None
            continue
        
        # Find disease indices
        disease_indices = []
        unique_indices = set()
        for disease in disease_names_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_full.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        
        if not disease_indices:
            fitted_models[disease_group] = None
            continue
        
        # Prepare data for Cox model
        cox_data = []
        n_prevalent_excluded = 0  # Counter for excluded prevalent cases
        for i in range(len(current_FH_train)):
            age_at_enrollment = current_FH_train.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)
            if t_enroll < 0 or t_enroll >= current_Y_train.shape[2]:
                continue
            
            # Exclude prevalent cases for single-disease groups only
            if len(disease_indices) == 1:
                d_idx = disease_indices[0]
                if torch.any(current_Y_train[i, d_idx, :t_enroll] > 0):
                    n_prevalent_excluded += 1
                    continue  # skip prevalent case
            
            end_time = min(t_enroll + follow_up_duration_years, current_Y_train.shape[2])
            if end_time <= t_enroll:
                continue
            
            for d_idx in disease_indices:
                Y_slice = current_Y_train[i, d_idx, t_enroll:end_time]
                if (torch.is_tensor(Y_slice) and torch.any(Y_slice > 0)) or \
                   (isinstance(Y_slice, np.ndarray) and np.any(Y_slice > 0)):
                    event_time = np.where(Y_slice > 0)[0][0] + t_enroll
                    age_at_event = 30 + event_time  # Convert to actual age
                    event = 1
                else:
                    age_at_event = 30 + end_time - 1  # Convert to actual age
                    event = 0
                
                # Create row for Cox model - now using actual age
                row = {
                    'age': age_at_event,  # Age at event/censoring
                    'event': event,
                    'sex': current_FH_train.iloc[i]['sex']
                }
                
                # Add family history indicators
                if fh_cols:
                    valid_fh_cols = [col for col in fh_cols if col in current_FH_train.columns]
                    if valid_fh_cols:
                        row['fh'] = current_FH_train.iloc[i][valid_fh_cols].any()

                cox_data.append(row)
        
        if not cox_data:
            fitted_models[disease_group] = None
            continue
            
        cox_df = pd.DataFrame(cox_data)

          # Check if we have enough events
        if cox_df['event'].sum() < 5:  # Require at least 5 events
            print(f"   Warning: Too few events ({cox_df['event'].sum()}) for {disease_group}")
            fitted_models[disease_group] = None
            continue
        
        try:
            # Fit Cox model using formula interface
            glm = LogisticRegression(max_iter=1000)
            
            # Build formula based on available covariates
            formula = 'sex'
            if 'fh' in cox_df.columns:
                formula += ' + fh'

            if target_sex_code is not None:
                formula = 'fh' if 'fh' in cox_df.columns else '1'  # Use intercept-only model
           
            
            glm.fit(cox_df, duration_col='age', event_col='event', formula=formula)
            fitted_models[disease_group] = cph
            print(f"   Model fitted for {disease_group} using {len(cox_df)} samples.")
            if len(disease_indices) == 1:
                print(f"   Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
        except Exception as e:
            print(f"   Error fitting {disease_group}: {e}")
            fitted_models[disease_group] = None
    
    print("Finished fitting glm models.")
    return fitted_models




# --- Function to Evaluate Cox Models on Test Set ---
def evaluate_cox_baseline_models(fitted_models, Y_test, FH_test, disease_mapping, major_diseases, disease_names, follow_up_duration_years=10, pce_df=None):
    """Evaluates pre-fitted Cox models on the test set. Optionally takes pce_df for PCE/PREVENT AUC reporting."""
    from lifelines.utils import concordance_index
    test_results = {}
    print("\nEvaluating Cox models on test data...")
    
    if not (len(Y_test) == len(FH_test)):
        raise ValueError(f"Test data size mismatch: Y_test ({len(Y_test)}), FH_test ({len(FH_test)})")
    
    FH_test = FH_test.reset_index(drop=True)
    if pce_df is not None:
        pce_df = pce_df.reset_index(drop=True)
    
    # For ASCVD analysis with pre-existing conditions
    pre_existing_conditions = {
        'RA': ['Rheumatoid arthritis'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast']
    }
    
    for disease_group, model in fitted_models.items():
        if model is None:
            test_results[disease_group] = {'c_index': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
            
        print(f" - Evaluating {disease_group}...")
        fh_cols = disease_mapping.get(disease_group, [])
        
        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1
        
        if target_sex_code is not None:
            mask_test = (FH_test['sex'] == target_sex_code)
        else:
            mask_test = pd.Series(True, index=FH_test.index)
        
        current_FH_test = FH_test[mask_test].copy()
        current_Y_test = Y_test[mask_test]
        if pce_df is not None:
            current_pce_df = pce_df[mask_test].copy()
        else:
            current_pce_df = None
        
        if len(current_FH_test) == 0:
            print(f"   Warning: No individuals for target sex code {target_sex_code}.")
            test_results[disease_group] = {'c_index': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        
        # Find disease indices
        disease_indices = []
        unique_indices = set()
        for disease in major_diseases.get(disease_group, []):
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_test.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        
        if not disease_indices:
            test_results[disease_group] = {'c_index': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        
        eval_data = []
        processed_indices = []
        n_prevalent_excluded = 0

        for i in range(len(current_FH_test)):
            age_at_enrollment = current_FH_test.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)
            if t_enroll < 0 or t_enroll >= current_Y_test.shape[2]:
                continue

            # Exclude prevalent cases for single-disease groups only
            if len(disease_indices) == 1:
                d_idx = disease_indices[0]
                if torch.any(current_Y_test[i, d_idx, :t_enroll] > 0):
                    n_prevalent_excluded += 1
                    continue  # skip prevalent case

            end_time = min(t_enroll + follow_up_duration_years, current_Y_test.shape[2])
            if end_time <= t_enroll:
                continue

            # --- AGGREGATE ACROSS DISEASES: one row per person ---
            event_found = False
            earliest_event_time = None
            for d_idx in disease_indices:
                Y_slice = current_Y_test[i, d_idx, t_enroll:end_time]
                if (torch.is_tensor(Y_slice) and torch.any(Y_slice > 0)) or \
                (isinstance(Y_slice, np.ndarray) and np.any(Y_slice > 0)):
                    this_event_time = np.where(Y_slice > 0)[0][0] + t_enroll
                    if (earliest_event_time is None) or (this_event_time < earliest_event_time):
                        earliest_event_time = this_event_time
                    event_found = True
            if event_found:
                age_at_event = 30 + earliest_event_time
                event = 1
            else:
                age_at_event = 30 + end_time - 1
                event = 0

            row = {
                'age_enroll': age_at_enrollment,
                'age': age_at_event,
                'event': event,
                'sex': current_FH_test.iloc[i]['sex']
            }
            if fh_cols:
                valid_fh_cols = [col for col in fh_cols if col in current_FH_test.columns]
                if valid_fh_cols:
                    row['fh'] = current_FH_test.iloc[i][valid_fh_cols].any()
            eval_data.append(row)
            processed_indices.append(current_FH_test.index[i])
                
        if not eval_data:
            print("   Warning: No individuals processed for evaluation.")
            test_results[disease_group] = {'c_index': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        
        eval_df = pd.DataFrame(eval_data)
        eval_df['orig_index'] = processed_indices  # Add for alignment
        
        try:
            # Get predicted risk scores using formula
            formula = 'sex'
            if 'fh' in eval_df.columns:
                formula += ' + fh'
            
            risk_scores = model.predict_partial_hazard(eval_df)
            
            # Calculate overall concordance index
            c_index = concordance_index(
                eval_df['age'],
                -risk_scores,  # Negative because higher risk should have shorter survival
                eval_df['event']
            )
            
            # Calculate confidence interval using bootstrap
            n_bootstraps = 100
            c_indices = []
            for _ in range(n_bootstraps):
                bootstrap_idx = np.random.choice(len(eval_df), len(eval_df), replace=True)
                bootstrap_df = eval_df.iloc[bootstrap_idx]
                bootstrap_risk = model.predict_partial_hazard(bootstrap_df)
                c_idx = concordance_index(
                    bootstrap_df['age'],
                    -bootstrap_risk,
                    bootstrap_df['event']
                )
                c_indices.append(c_idx)
            
            ci_lower = np.percentile(c_indices, 2.5)
            ci_upper = np.percentile(c_indices, 97.5)
            
            n_events = eval_df['event'].sum()
            n_total = len(eval_df)
            
            test_results[disease_group] = {
                'c_index': c_index,
                'ci': (ci_lower, ci_upper),
                'n_events': n_events,
                'n_total': n_total
            }
            
            print(f"   Overall C-index: {c_index:.3f} ({ci_lower:.3f}-{ci_upper:.3f})")
            print(f"   Events: {n_events}/{n_total}")
            if len(disease_indices) == 1:
                print(f"   Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
            
            # Sex-stratified analysis (except for sex-specific diseases)
            if disease_group not in ['Breast_Cancer', 'Prostate_Cancer']:
                print("\n   Sex-stratified analysis:")
                for sex in [0, 1]:  # 0 for female, 1 for male
                    sex_mask = eval_df['sex'] == sex
                    sex_df = eval_df[sex_mask]
                    if len(sex_df) > 0:
                        sex_risk = model.predict_partial_hazard(sex_df)
                        sex_c_index = concordance_index(
                            sex_df['age'],
                            -sex_risk,
                            sex_df['event']
                        )
                        sex_events = sex_df['event'].sum()
                        print(f"   {'Female' if sex == 0 else 'Male'}: C-index = {sex_c_index:.3f}, Events = {sex_events}/{len(sex_df)}")
            
            # Pre-existing condition analysis for ASCVD
            if disease_group == 'ASCVD':
                print("\n   ASCVD risk in patients with pre-existing conditions:")
                for condition in pre_existing_conditions.keys():
                    condition_mask = eval_df[f'has_{condition}'] == True
                    condition_df = eval_df[condition_mask]
                    if len(condition_df) > 0:
                        condition_risk = model.predict_partial_hazard(condition_df)
                        condition_c_index = concordance_index(
                            condition_df['age'],
                            -condition_risk,
                            condition_df['event']
                        )
                        condition_events = condition_df['event'].sum()
                        print(f"   {condition}: C-index = {condition_c_index:.3f}, Events = {condition_events}/{len(condition_df)}")
                # PCE and PREVENT AUC for ASCVD if available in pce_df
                from sklearn.metrics import roc_auc_score
                if current_pce_df is not None:
                    # Use processed_indices to align with eval_df
                    aligned_pce_df = current_pce_df.iloc[eval_df['orig_index']].reset_index(drop=True)
                    # PCE
                    if 'pce_goff_fuull' in aligned_pce_df.columns:
                        pce_vals = aligned_pce_df['pce_goff_fuull'].values
                        events = eval_df['event'].values
                        mask = ~np.isnan(pce_vals)
                        if np.any(mask):
                            try:
                                auc_pce = roc_auc_score(events[mask], pce_vals[mask])
                                print(f"   PCE (pce_goff_fuull) AUC: {auc_pce:.3f}")
                            except Exception as e:
                                print(f"   PCE AUC error: {e}")
                        else:
                            print("   PCE AUC error: all values are NaN")
                    # PREVENT
                    if 'prevent_impute' in aligned_pce_df.columns:
                        prevent_vals = aligned_pce_df['prevent_impute'].values
                        mask = ~np.isnan(prevent_vals)
                        if np.any(mask):
                            try:
                                auc_prevent = roc_auc_score(events[mask], prevent_vals[mask])
                                print(f"   PREVENT (prevent_impute) AUC: {auc_prevent:.3f}")
                            except Exception as e:
                                print(f"   PREVENT AUC error: {e}")
                        else:
                            print("   PREVENT AUC error: all values are NaN")

        
        except Exception as e:
            print(f"   Error evaluating {disease_group}: {e}")
            test_results[disease_group] = {'c_index': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
    
    print("Finished evaluating Cox models.")
    return test_results



major_diseases = { # Example, ADD ALL YOURS
    'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis'], 'Diabetes': ['Type 2 diabetes'],
    'Atrial_Fib': ['Atrial fibrillation and flutter'], 'CKD': ['Chronic renal failure [CKD]'],
    'All_Cancers': ['Colon cancer', 'Breast cancer [female]', 'Cancer of prostate'], 
    'Stroke': ['Cerebral artery occlusion, with cerebral infarction'], 'Heart_Failure': ['Congestive heart failure (CHF) NOS'],
    'Pneumonia': ['Pneumonia'], 'COPD': ['Chronic airway obstruction'], 'Osteoporosis': ['Osteoporosis NOS'],
    'Anemia': ['Iron deficiency anemias'], 'Colorectal_Cancer': ['Colon cancer'],
    'Breast_Cancer': ['Breast cancer [female]'], 'Prostate_Cancer': ['Cancer of prostate'], 
    'Lung_Cancer': ['Cancer of bronchus; lung'], 'Bladder_Cancer': ['Malignant neoplasm of bladder'],
    'Secondary_Cancer': ['Secondary malignant neoplasm'], 'Depression': ['Major depressive disorder'],
    'Anxiety': ['Anxiety disorder'], 'Bipolar_Disorder': ['Bipolar'], 'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
    'Psoriasis': ['Psoriasis vulgaris'], 'Ulcerative_Colitis': ['Ulcerative colitis'], 'Crohns_Disease': ['Regional enteritis'],
    'Asthma': ['Asthma'], 'Parkinsons': ["Parkinson's disease"], 'Multiple_Sclerosis': ['Multiple sclerosis'],
    'Thyroid_Disorders': ['Hypothyroidism NOS']
}
disease_mapping = { # Example, ADD ALL YOURS
     'ASCVD': ['heart_disease', 'heart_disease.1'], 'Stroke': ['stroke', 'stroke.1'],
     'Diabetes': ['diabetes', 'diabetes.1'], 'Breast_Cancer': ['breast_cancer', 'breast_cancer.1'],
     'Prostate_Cancer': ['prostate_cancer', 'prostate_cancer.1'], 'Lung_Cancer': ['lung_cancer', 'lung_cancer.1'],
     'Colorectal_Cancer': ['bowel_cancer', 'bowel_cancer.1'], 'Depression': [], 'Osteoporosis': [], 
     'Parkinsons': ['parkinsons', 'parkinsons.1'], 'COPD': [], 'Anemia': [], 'CKD': [], 
     'Heart_Failure': ['heart_disease', 'heart_disease.1'], 'Pneumonia': [], 'Atrial_Fib': [], 
     'Bladder_Cancer': [], 'Secondary_Cancer': [], 'Anxiety': [], 'Bipolar_Disorder': [], 
     'Rheumatoid_Arthritis': [], 'Psoriasis': [], 'Ulcerative_Colitis': [], 'Crohns_Disease': [], 
     'Asthma': [], 'Multiple_Sclerosis': [], 'Thyroid_Disorders': []
 }

import torch
import pandas as pd
import numpy as np # Using numpy for standard math functions

def calculate_enrollment_event_rates(Y_full, enrollment_df, disease_names, major_diseases):
    """
    Calculates 1-year and 10-year event rates from enrollment for major diseases.

    Args:
        Y_full (torch.Tensor): The full event tensor (N_full, D, T).
        enrollment_df (pd.DataFrame): DataFrame with enrollment info (N_full rows),
                                     must contain 'age' and 'sex' columns.
        disease_names (list): List of disease names corresponding to dim D of Y_full.
        major_diseases (dict): Dictionary mapping disease groups to lists of disease names.

    Returns:
        dict: A dictionary where keys are disease groups and values are dicts
              containing 'rate_1y', 'count_1y', 'processed_1y',
              'rate_10y', 'count_10y', 'processed_10y'.
    """
    results = {}
    N_full = Y_full.shape[0]
    T_max = Y_full.shape[2]
    D_max = Y_full.shape[1]

    # --- Input Validation ---
    if len(enrollment_df) != N_full:
        raise ValueError(f"Mismatch between Y_full ({N_full}) and enrollment_df ({len(enrollment_df)}) rows.")
    if 'age' not in enrollment_df.columns:
        raise ValueError("'age' column missing from enrollment_df.")
    if 'sex' not in enrollment_df.columns:
        raise ValueError("'sex' column missing from enrollment_df.")

    enrollment_df = enrollment_df.reset_index(drop=True) # Ensure 0-based index

    for disease_group, disease_list in major_diseases.items():
        print(f"\nCalculating rates for {disease_group}...")

        # --- Get Disease Indices ---
        disease_indices = []
        unique_indices = set()
        for disease in disease_list:
            # Find exact or partial matches (lowercase)
            indices = [i for i, name in enumerate(disease_names)
                       if disease.lower() in name.lower()]
            for idx in indices:
                 if idx not in unique_indices and idx < D_max: # Check bounds
                      disease_indices.append(idx)
                      unique_indices.add(idx)

        if not disease_indices:
             print(f"  Warning: No valid matching disease indices found for {disease_group}.")
             results[disease_group] = {'rate_1y': 0.0, 'count_1y': 0, 'processed_1y': 0,
                                       'rate_10y': 0.0, 'count_10y': 0, 'processed_10y': 0}
             continue
        print(f"  Found {len(disease_indices)} indices: {disease_indices[:5]}...") # Print first few

        # --- Sex Filtering ---
        target_sex_code = None # Assuming 0 for female, 1 for male based on prior code
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1

        if target_sex_code is not None:
            mask_enroll = (enrollment_df['sex'] == target_sex_code)
            # Get integer positions (iloc indices) where mask is True
            process_indices = np.where(mask_enroll)[0]
            print(f"  Filtering for sex code {target_sex_code}: Found {len(process_indices)} individuals.")
            if len(process_indices) == 0:
                 print(f"  Warning: No individuals found for target sex code '{target_sex_code}'. Skipping.")
                 results[disease_group] = {'rate_1y': 0.0, 'count_1y': 0, 'processed_1y': 0,
                                           'rate_10y': 0.0, 'count_10y': 0, 'processed_10y': 0}
                 continue
        else:
            # Use all integer indices if not sex-specific
            process_indices = np.arange(N_full)

        # --- Calculate Event Counts and Rates ---
        event_count_1y = 0
        event_count_10y = 0
        processed_count_1y = 0
        processed_count_10y = 0

        # Convert relevant part of Y_full to boolean once for efficiency
        Y_bool = Y_full[:, disease_indices, :] > 0

        for idx in process_indices: # Iterate through filtered indices
            # Access DataFrame row using iloc with the absolute index
            age_at_enrollment = enrollment_df.iloc[idx]['age']
            # Calculate time index corresponding to enrollment age
            # Assuming age 30 corresponds to t=0
            t_enroll = int(age_at_enrollment - 30)

            # Check if enrollment time is valid within the tensor's time dimension
            if t_enroll < 0 or t_enroll >= T_max:
                continue # Skip this individual if enrollment is out of bounds

            # --- 1-Year Window ---
            t_end_1y = min(t_enroll + 1, T_max)
            if t_end_1y > t_enroll: # Ensure window has non-zero length
                processed_count_1y += 1
                # Check if any event occurred in any relevant disease index within the window
                # Use torch.any along the disease dimension (dim=1) and time dimension (dim=2)
                if torch.any(Y_bool[idx, :, t_enroll:t_end_1y]):
                    event_count_1y += 1

            # --- 10-Year Window ---
            t_end_10y = min(t_enroll + 10, T_max)
            if t_end_10y > t_enroll: # Ensure window has non-zero length
                processed_count_10y += 1
                # Check if any event occurred in any relevant disease index within the window
                if torch.any(Y_bool[idx, :, t_enroll:t_end_10y]):
                     event_count_10y += 1

        # Calculate rates based on the number actually processed for each window
        rate_1y = (event_count_1y / processed_count_1y) if processed_count_1y > 0 else 0.0
        rate_10y = (event_count_10y / processed_count_10y) if processed_count_10y > 0 else 0.0

        results[disease_group] = {
            'rate_1y': rate_1y,
            'count_1y': event_count_1y,
            'processed_1y': processed_count_1y,
            'rate_10y': rate_10y,
            'count_10y': event_count_10y,
            'processed_10y': processed_count_10y
        }
        print(f"  1-Year: Count={event_count_1y}, Processed={processed_count_1y}, Rate={rate_1y:.4f}")
        print(f"  10-Year: Count={event_count_10y}, Processed={processed_count_10y}, Rate={rate_10y:.4f}")

    # --- Print Summary ---
    print("\n--- Event Rate Summary ---")
    print(f"{'Disease Group':<22} | {'1Y Count':<10} {'1Y Processed':<12} {'1Y Rate':<10} | {'10Y Count':<10} {'10Y Processed':<12} {'10Y Rate':<10}")
    print("-" * 100)
    for group, res in results.items():
        print(f"{group:<22} | {res['count_1y']:<10d} {res['processed_1y']:<12d} {res['rate_1y']:<10.4f} | {res['count_10y']:<10d} {res['processed_10y']:<12d} {res['rate_10y']:<10.4f}")
    print("-" * 100)

    return results


def plot_model_comparison_bars(diseases, aladynoulli_scores, cox_scores, cox_ci, additional_models=None, figsize=(15, 10)):
    """
    Create a bar plot comparing model performances.
    
    Args:
        diseases (list): List of disease names
        aladynoulli_scores (list): AUC scores for Aladynoulli model
        cox_scores (list): AUC scores for Cox model
        cox_ci (list): List of tuples containing (lower, upper) CI bounds for Cox
        additional_models (dict): Dictionary mapping disease to (score, name) for additional models
        figsize (tuple): Figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set style
    plt.style.use('seaborn')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Number of diseases and bar width
    n_diseases = len(diseases)
    bar_width = 0.35
    
    # Positions for bars
    indices = np.arange(n_diseases)
    
    # Create bars
    aladynoulli_bars = ax.bar(indices - bar_width/2, aladynoulli_scores, bar_width, 
                             label='Aladynoulli', color='#2ecc71', alpha=0.8)
    cox_bars = ax.bar(indices + bar_width/2, cox_scores, bar_width,
                     label='Cox', color='#3498db', alpha=0.8)
    
    # Add error bars for Cox model
    cox_errors_lower = [score - ci[0] for score, ci in zip(cox_scores, cox_ci)]
    cox_errors_upper = [ci[1] - score for score, ci in zip(cox_scores, cox_ci)]
    ax.errorbar(indices + bar_width/2, cox_scores, 
                yerr=[cox_errors_lower, cox_errors_upper],
                fmt='none', color='#2980b9', capsize=5)
    
    # Add additional models if provided
    if additional_models:
        for disease, models in additional_models.items():
            if disease in diseases:
                idx = diseases.index(disease)
                for i, (score, name) in enumerate(models):
                    offset = bar_width * (i + 1)
                    ax.bar(idx + offset, score, bar_width,
                          label=name if idx == 0 else "", 
                          color=plt.cm.Set2(i+2))
    
    # Customize plot
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Model Performance Comparison by Disease', fontsize=14, pad=20)
    ax.set_xticks(indices)
    ax.set_xticklabels(diseases, rotation=45, ha='right')
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set y-axis limits with some padding
    ax.set_ylim(0.4, 0.8)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Example usage:
# additional_models = {
#     'ASCVD': [(0.678, 'PCE'), (0.66, 'PREVENT')],
#     'Breast_Cancer': [(0.541, 'Gail')]
# }
# fig = plot_model_comparison_bars(diseases, aladynoulli_scores, cox_scores, cox_ci, additional_models)

def plot_ten_year_comparison_bars(aladyn_results, cox_results, event_rate_results, major_diseases, figsize=(15, 8), save_path=None):
    """
    Create a bar plot comparing 10-year model performances with event rate-based standard errors.
    
    Args:
        aladyn_results (dict): Results from Aladynoulli model
        cox_results (dict): Results from Cox model with disease-specific c-indices
        event_rate_results (dict): Results from calculate_enrollment_event_rates
        major_diseases (dict): Dictionary of major diseases
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure. If provided, saves as PDF
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set figure DPI for high quality
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert dictionary keys to list for plotting
    disease_list = list(major_diseases.keys())
    
    # Number of diseases and bar width
    n_diseases = len(disease_list)
    bar_width = 0.35
    
    # Positions for bars
    indices = np.arange(n_diseases)
    
    # Extract metrics and calculate standard errors
    aladyn_scores = []
    cox_scores = []
    standard_errors = []
    
    for disease in disease_list:
        try:
            # Get Aladynoulli score
            aladyn_scores.append(aladyn_results[disease]['auc'])
            
            # Get Cox score for this disease
            cox_scores.append(cox_results[disease]['c_index'])
            
            # Calculate standard error from event rate
            res = event_rate_results[disease]
            rate = res['rate_10y']
            n = res['processed_10y']
            # Multiply standard error by 1.96 for 95% confidence interval
            se = 1.96 * np.sqrt(rate * (1-rate) / n)
            standard_errors.append(se)
        except KeyError as e:
            print(f"Warning: Missing data for {disease}: {e}")
            aladyn_scores.append(np.nan)
            cox_scores.append(np.nan)
            standard_errors.append(np.nan)
    
    # Convert to numpy arrays for easier manipulation
    aladyn_scores = np.array(aladyn_scores)
    cox_scores = np.array(cox_scores)
    standard_errors = np.array(standard_errors)
    
    # Create bars with darker, more saturated colors
    ax.bar(indices - bar_width/2, aladyn_scores, bar_width, 
           label='Aladynoulli', color='#27ae60', alpha=0.9)
    ax.bar(indices + bar_width/2, cox_scores, bar_width,
           label='Cox', color='#2980b9', alpha=0.9)
    
    # Add error bars using event rate standard errors with enhanced visibility
    for i, (score, se) in enumerate(zip(aladyn_scores, standard_errors)):
        x = indices[i] - bar_width/2
        ax.vlines(x, score - se, score + se, color='#27ae60', 
                 linewidth=3, capstyle='round')
        ax.hlines([score - se, score + se], x - 0.05, x + 0.05, 
                 color='#27ae60', linewidth=3)

    for i, (score, se) in enumerate(zip(cox_scores, standard_errors)):
        x = indices[i] + bar_width/2
        ax.vlines(x, score - se, score + se, color='#2980b9', 
                 linewidth=3, capstyle='round')
        ax.hlines([score - se, score + se], x - 0.05, x + 0.05, 
                 color='#2980b9', linewidth=3)
    
    # Add PCE/PREVENT for ASCVD with thicker error bars
    if 'ASCVD' in disease_list:
        ascvd_idx = disease_list.index('ASCVD')
        # Get ASCVD-specific standard error
        ascvd_se = standard_errors[ascvd_idx]
        
        # Add PCE with error bar
        x = indices[ascvd_idx] + 1.5*bar_width
        ax.bar(x, 0.678, bar_width, label='PCE', color='#c0392b', alpha=0.9)
        ax.vlines(x, 0.678 - ascvd_se, 0.678 + ascvd_se, color='#c0392b',
                 linewidth=3, capstyle='round')
        ax.hlines([0.678 - ascvd_se, 0.678 + ascvd_se], x - 0.05, x + 0.05,
                 color='#c0392b', linewidth=3)
        
        # Add PREVENT with error bar
        x = indices[ascvd_idx] + 2.5*bar_width
        ax.bar(x, 0.66, bar_width, label='PREVENT', color='#8e44ad', alpha=0.9)
        ax.vlines(x, 0.66 - ascvd_se, 0.66 + ascvd_se, color='#8e44ad',
                 linewidth=3, capstyle='round')
        ax.hlines([0.66 - ascvd_se, 0.66 + ascvd_se], x - 0.05, x + 0.05,
                 color='#8e44ad', linewidth=3)
    
    # Add Gail for Breast Cancer with thicker error bars
    if 'Breast_Cancer' in disease_list:
        bc_idx = disease_list.index('Breast_Cancer')
        # Get breast cancer-specific standard error
        bc_se = standard_errors[bc_idx]
        
        # Add Gail with error bar
        x = indices[bc_idx] + 1.5*bar_width
        ax.bar(x, 0.541, bar_width, label='Gail', color='#f39c12', alpha=0.9)
        ax.vlines(x, 0.541 - bc_se, 0.541 + bc_se, color='#f39c12',
                 linewidth=3, capstyle='round')
        ax.hlines([0.541 - bc_se, 0.541 + bc_se], x - 0.05, x + 0.05,
                 color='#f39c12', linewidth=3)
    
    # Customize plot
    ax.set_ylabel('10-year AUC', fontsize=12, fontweight='bold')
    ax.set_title('10-year Performance Comparison by Disease', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(indices)
    ax.set_xticklabels(disease_list, rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Set y-axis limits
    ax.set_ylim(0.4, 0.8)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set background colors
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path is not None:
        # Ensure the figure is saved with high quality
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    
    return fig

def create_ten_year_comparison_plot(aladyn_results, cox_results, event_rate_results, major_diseases, save_path=None):
    """
    Creates a 10-year comparison plot for major diseases including ASCVD and breast cancer.
    
    Args:
        aladyn_results (dict): Results from Aladynoulli model
        cox_results (dict): Results from Cox model
        event_rate_results (dict): Results from calculate_enrollment_event_rates
        major_diseases (dict): Dictionary of major diseases
        save_path (str, optional): Path to save the plot as PDF. If None, plot won't be saved.
    """
    fig = plot_ten_year_comparison_bars(
        aladyn_results=aladyn_results,
        cox_results=cox_results,
        event_rate_results=event_rate_results,
        major_diseases=major_diseases,
        figsize=(15, 8)
    )
    
    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    
    return fig



def plot_disease_lambda_alignment_for_test(model, Y_test, censored_indices, original_event_times, 
                                          disease_idx=112, sig_idx=None, n_samples=5):
    """
    Plot lambda values aligned with disease occurrences for censored patients
    
    Args:
        model: Trained model with censored event times
        Y_test: Original uncensored data
        censored_indices: Indices of patients who were censored
        original_event_times: Original event times for censored patients
        disease_idx: Index of the disease to analyze (default: 112 for MI)
        sig_idx: Specific signature index to plot (if None, will use most associated)
        n_samples: Number of patients to sample for plotting
    """
    # Filter patients who actually had the disease
    patients_with_disease = []
    diagnosis_times = []
    patient_indices = []
    
    for i, patient_idx in enumerate(censored_indices):
        # Check if this patient had the disease in the test data
        diag_time = torch.where(Y_test[patient_idx, disease_idx])[0]
        if len(diag_time) > 0:
            patients_with_disease.append(patient_idx)
            diagnosis_times.append(diag_time[0].item())
            patient_indices.append(i)  # Store position in censored_indices list
    
    # If no patients found, try a different disease
    if len(patients_with_disease) == 0:
        print(f"No patients found with disease {disease_idx}. Try a different disease index.")
        return
    
    # Sample patients if we have more than requested
    if len(patients_with_disease) > n_samples:
        sample_idx = np.random.choice(len(patients_with_disease), n_samples, replace=False)
        patients_with_disease = [patients_with_disease[i] for i in sample_idx]
        diagnosis_times = [diagnosis_times[i] for i in sample_idx]
        patient_indices = [patient_indices[i] for i in sample_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    time_points = np.arange(model.T)
    
    # Find signature that most strongly associates with this disease if not specified
    if sig_idx is None:
        psi_disease = model.psi[:, disease_idx].detach()
        sig_idx = torch.argmax(psi_disease).item()
    
    # Get disease name if available
    disease_name = model.disease_names[disease_idx] if hasattr(model, 'disease_names') and model.disease_names is not None else f"Disease {disease_idx}"
    
    # Plot for each sampled patient
    for i, patient in enumerate(patients_with_disease):
        diag_time = diagnosis_times[i]
        censor_time = original_event_times[patient_indices[i]] - 2  # 2 years before event
        
        # Plot lambda (detached)
        lambda_values = torch.softmax(model.lambda_[patient].detach(), dim=0)[sig_idx]
        ax.plot(time_points, lambda_values.numpy(), alpha=0.7, label=f'Patient {patient}')
        
        # Mark diagnosis time with solid line
        ax.axvline(x=diag_time, linestyle='-', color='blue', alpha=0.3)
        
        # Mark censoring time with dashed line
        ax.axvline(x=censor_time, linestyle='--', color='red', alpha=0.3)
    
    # Add legend for vertical lines
    ax.axvline(x=0, linestyle='-', color='blue', alpha=0.3, label='Diagnosis Time')
    ax.axvline(x=0, linestyle='--', color='red', alpha=0.3, label='Censoring Time')
    
    # Set title and labels
    ax.set_title(f'Lambda Values for Signature {sig_idx} (Most Associated with {disease_name})\n'
                f'Blue Lines: Diagnosis Times, Red Lines: Censoring Times')
    ax.set_xlabel('Time')
    ax.set_ylabel('Lambda (proportion)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return patient info for further analysis
    return {
        'patients': patients_with_disease,
        'diagnosis_times': diagnosis_times,
        'censoring_times': [original_event_times[i] - 2 for i in patient_indices]
    }



def create_censored_event_times(Y, E, ascvd_indices=[111, 112, 113, 114, 115, 116], censoring_window=2):
    """
    Create censored event times for ASCVD predictive evaluation
    
    Args:
        Y: Disease occurrence tensor [N, D, T]
        E: Original event times
        ascvd_indices: Indices of ASCVD diseases to censor
        censoring_window: Number of years before event to censor
        
    Returns:
        E_censored: Modified event times with ASCVD events censored
        censored_indices: Indices of patients who were censored
        original_event_times: Original event times for censored patients
    """
    # Create a copy of event times to modify
    E_censored = E.clone()
    
    # Track which patients were censored and their original event times
    censored_indices = []
    original_event_times = []
    
    # For each patient
    for i in range(Y.shape[0]):
        # Only look at ASCVD diseases
        ascvd_data = Y[i, ascvd_indices, :]
        
        # Find the first occurrence of any ASCVD disease for this patient
        any_ascvd = torch.sum(ascvd_data, dim=0) > 0
        
        # Find the first time index where any ASCVD disease occurred
        ascvd_times = torch.where(any_ascvd)[0]
        
        if len(ascvd_times) > 0:
            # Get the earliest ASCVD event time
            event_time = ascvd_times[0].item()
            
            # If they had an event and it's not at the very beginning
            if event_time > censoring_window:
                # Store original event time
                original_event_times.append(event_time)
                
                # Censor by setting event time to be earlier
                censor_time = max(0, event_time - censoring_window)
                
                # Update event time in E_censored
                # This depends on the structure of E
                if len(E.shape) == 1:
                    # If E is a 1D tensor [N]
                    E_censored[i] = censor_time
                elif len(E.shape) == 2:
                    # If E is a 2D tensor [N, D] or [N, T]
                    # We need to determine which dimension to update
                    if E.shape[1] == Y.shape[1]:  # [N, D]
                        # Update event times for ASCVD diseases
                        for d_idx in ascvd_indices:
                            if d_idx < E.shape[1]:  # Make sure index is valid
                                E_censored[i, d_idx] = censor_time
                    else:  # Assume [N, T] or similar
                        # Set all times after censor_time to 0 or another indicator
                        E_censored[i, censor_time:] = 0
                
                # Track this patient
                censored_indices.append(i)
    
    return E_censored, censored_indices, original_event_times
def analyze_specific_patients(model, patient_indices, disease_idx=112, sig_idx=None, save_pdf=None):
    """
    Analyze lambda values for specific patients.
    
    Args:
        model: Trained model
        patient_indices: List of patient indices to analyze
        disease_idx: Index of the disease to analyze (default: 112 for MI)
        sig_idx: Specific signature index to plot (if None, will use most associated)
        save_pdf: Optional path to save the plot as PDF
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    time_points = np.arange(model.T)
    
    # Find signature that most strongly associates with this disease if not specified
    if sig_idx is None:
        psi_disease = model.psi[:, disease_idx].detach()
        sig_idx = torch.argmax(psi_disease).item()
    
    # Get disease name if available
    disease_name = model.disease_names[disease_idx] if hasattr(model, 'disease_names') and model.disease_names is not None else f"Disease {disease_idx}"
    
    # Get color cycle from the current style
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot for each patient
    for i, patient in enumerate(patient_indices):
        if patient >= model.lambda_.shape[0]:
            print(f"Patient {patient} is out of range. Skipping.")
            continue
            
        # Get color for this patient
        color = colors[i % len(colors)]
            
        # Plot lambda (detached)
        lambda_values = torch.softmax(model.lambda_[patient].detach(), dim=0)[sig_idx]
        ax.plot(time_points, lambda_values.numpy(), alpha=0.7, label=f'Patient {patient}', color=color)
        
        # If we have Y data, mark disease occurrence with matching color
        if hasattr(model, 'Y'):
            diag_times = torch.where(model.Y[patient, disease_idx])[0]
            for t in diag_times:
                ax.axvline(x=t.item(), linestyle='-', color=color, alpha=0.3)
    
    # Set title and labels
    ax.set_title(f'Lambda Values for Signature {sig_idx}\n(Most Associated with {disease_name})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Lambda (proportion)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to PDF if path is provided
    if save_pdf:
        plt.savefig(save_pdf, bbox_inches='tight', dpi=300, format='pdf')
    
    plt.show()
    return fig



import random
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt

def compare_case_control_trajectories(model, Y_test, disease_idx=112, n_controls=100, sig_idx=None):
    """
    Compare lambda trajectories between disease cases and controls
    
    Args:
        model: Trained ALADYNOULLI model
        Y_test: Test data tensor
        disease_idx: Index of the disease to analyze
        n_controls: Number of control patients to sample
        sig_idx: Signature index (if None, will use most associated signature)
    """
    # Find signature most associated with this disease
    if sig_idx is None:
        psi_disease = model.psi[:, disease_idx].detach()
        sig_idx = torch.argmax(psi_disease).item()
    
    # Find patients who develop the disease
    cases = []
    case_diag_times = []
    for patient_idx in range(Y_test.shape[0]):
        diag_times = torch.where(Y_test[patient_idx, disease_idx])[0]
        if len(diag_times) > 0:
            cases.append(patient_idx)
            case_diag_times.append(diag_times[0].item())
    
    # Find control patients (who never develop the disease)
    controls = []
    for patient_idx in range(Y_test.shape[0]):
        if not torch.any(Y_test[patient_idx, disease_idx]).item():
            controls.append(patient_idx)
    
    # Sample controls if there are too many
    if len(controls) > n_controls:
        controls = random.sample(controls, n_controls)
    
    print(f"Found {len(cases)} cases and {len(controls)} controls")
    
    # Get lambda values
    case_lambdas = []
    for patient_idx in cases:
        lambda_values = torch.softmax(model.lambda_[patient_idx].detach(), dim=0)[sig_idx].numpy()
        case_lambdas.append(lambda_values)
    
    control_lambdas = []
    for patient_idx in controls:
        lambda_values = torch.softmax(model.lambda_[patient_idx].detach(), dim=0)[sig_idx].numpy()
        control_lambdas.append(lambda_values)
    
    # Calculate means and confidence intervals
    case_mean = np.mean(case_lambdas, axis=0)
    case_std = np.std(case_lambdas, axis=0) / np.sqrt(len(cases))  # Standard error
    control_mean = np.mean(control_lambdas, axis=0)
    control_std = np.std(control_lambdas, axis=0) / np.sqrt(len(controls))  # Standard error
    
    # Plot
    time_points = np.arange(len(case_mean))
    plt.figure(figsize=(12, 6))
    
    plt.plot(time_points, case_mean, 'r-', label=f'{get_disease_name(disease_idx)} Cases')
    plt.fill_between(time_points, case_mean - 1.96*case_std, case_mean + 1.96*case_std, color='r', alpha=0.2)
    
    plt.plot(time_points, control_mean, 'b-', label='Controls (No Disease)')
    plt.fill_between(time_points, control_mean - 1.96*control_std, control_mean + 1.96*control_std, color='b', alpha=0.2)
    
    plt.title(f'Lambda Trajectories for Signature {sig_idx}')
    plt.xlabel('Time')
    plt.ylabel('Lambda Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate when trajectories significantly diverge
    t_stats = []
    p_values = []
    for t in range(len(case_mean)):
        case_vals = [c[t] for c in case_lambdas]
        control_vals = [c[t] for c in control_lambdas]
        t_stat, p_val = ttest_ind(case_vals, control_vals, equal_var=False)
        t_stats.append(t_stat)
        p_values.append(p_val)
    
    # Find earliest significant divergence (with Bonferroni correction)
    divergence_time = None
    for t, p in enumerate(p_values):
        if p < 0.05 / len(p_values):
            divergence_time = t
            break
    
    if divergence_time is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(time_points, -np.log10(p_values), 'k-')
        plt.axhline(y=-np.log10(0.05 / len(p_values)), color='r', linestyle='--', 
                   label='Significance Threshold (Bonferroni-corrected)')
        plt.axvline(x=divergence_time, color='g', linestyle='--',
                   label=f'First Significant Divergence: Time {divergence_time}')
        plt.title('Statistical Significance of Case-Control Difference')
        plt.xlabel('Time')
        plt.ylabel('-log10(p-value)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Trajectories significantly diverge at time {divergence_time}")
        
        # Calculate average diagnosis time for context
        avg_diag_time = np.mean(case_diag_times)
        print(f"Average diagnosis time: {avg_diag_time:.1f}")
        print(f"Average lead time: {avg_diag_time - divergence_time:.1f} time units")
    else:
        print("No significant divergence found")
    
    return {
        'case_mean': case_mean,
        'control_mean': control_mean,
        'p_values': p_values,
        'divergence_time': divergence_time,
        'avg_diagnosis_time': np.mean(case_diag_times) if case_diag_times else None
    }

def get_disease_name(disease_idx):
    """
    Get the name of a disease based on its index
    """
    disease_names = {
        111: "Unstable Angina",
        112: "Myocardial Infarction",
        113: "Coronary Atherosclerosis",
        114: "Other Chronic Ischemic Heart Disease",
        115: "Heart Failure",
        116: "Atrial Fibrillation"
    }
    return disease_names.get(disease_idx, f"Disease {disease_idx}")

def create_combined_mi_plots(model, Y_test, censored_indices, original_event_times, disease_idx=112, sig_idx=None):
    """
    Create a combined figure showing both population-level trajectories and individual examples
    with censoring times for Myocardial Infarction.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Population-level plot (top)
    # Find signature most associated with this disease
    if sig_idx is None:
        psi_disease = model.psi[:, disease_idx].detach()
        sig_idx = torch.argmax(psi_disease).item()
    
    # Get cases and controls
    cases = []
    case_diag_times = []
    for patient_idx in range(Y_test.shape[0]):
        diag_times = torch.where(Y_test[patient_idx, disease_idx])[0]
        if len(diag_times) > 0:
            cases.append(patient_idx)
            case_diag_times.append(diag_times[0].item())
    
    controls = []
    for patient_idx in range(Y_test.shape[0]):
        if not torch.any(Y_test[patient_idx, disease_idx]).item():
            controls.append(patient_idx)
    
    # Sample 100 controls
    if len(controls) > 100:
        controls = random.sample(controls, 100)
    
    # Calculate trajectories
    case_lambdas = []
    for patient_idx in cases:
        lambda_values = torch.softmax(model.lambda_[patient_idx].detach(), dim=0)[sig_idx].numpy()
        case_lambdas.append(lambda_values)
    
    control_lambdas = []
    for patient_idx in controls:
        lambda_values = torch.softmax(model.lambda_[patient_idx].detach(), dim=0)[sig_idx].numpy()
        control_lambdas.append(lambda_values)
    
    # Calculate means and confidence intervals
    case_mean = np.mean(case_lambdas, axis=0)
    case_std = np.std(case_lambdas, axis=0) / np.sqrt(len(cases))
    control_mean = np.mean(control_lambdas, axis=0)
    control_std = np.std(control_lambdas, axis=0) / np.sqrt(len(controls))
    
    time_points = np.arange(len(case_mean))
    
    # Plot population trajectories
    ax1.plot(time_points, case_mean, 'r-', label='Myocardial Infarction Cases')
    ax1.fill_between(time_points, case_mean - 1.96*case_std, case_mean + 1.96*case_std, color='r', alpha=0.2)
    ax1.plot(time_points, control_mean, 'b-', label='Controls (No Disease)')
    ax1.fill_between(time_points, control_mean - 1.96*control_std, control_mean + 1.96*control_std, color='b', alpha=0.2)
    ax1.set_title(f'Population Lambda Trajectories for Signature {sig_idx}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Lambda Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Individual trajectories plot (bottom)
    # Find patients with disease in censored set
    patients_with_disease = []
    diagnosis_times = []
    patient_indices = []
    
    for i, patient_idx in enumerate(censored_indices):
        diag_time = torch.where(Y_test[patient_idx, disease_idx])[0]
        if len(diag_time) > 0:
            patients_with_disease.append(patient_idx)
            diagnosis_times.append(diag_time[0].item())
            patient_indices.append(i)
    
    # Sample 5 patients if we have more
    if len(patients_with_disease) > 5:
        sample_idx = np.random.choice(len(patients_with_disease), 5, replace=False)
        patients_with_disease = [patients_with_disease[i] for i in sample_idx]
        diagnosis_times = [diagnosis_times[i] for i in sample_idx]
        patient_indices = [patient_indices[i] for i in sample_idx]
    
    # Plot individual trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(patients_with_disease)))
    for i, (patient, color) in enumerate(zip(patients_with_disease, colors)):
        diag_time = diagnosis_times[i]
        censor_time = original_event_times[patient_indices[i]] - 2
        
        lambda_values = torch.softmax(model.lambda_[patient].detach(), dim=0)[sig_idx]
        ax2.plot(time_points, lambda_values.numpy(), color=color, alpha=0.7, label=f'Patient {patient}')
        
        # Mark diagnosis and censoring times
        ax2.axvline(x=diag_time, color=color, linestyle='-', alpha=0.3)
        ax2.axvline(x=censor_time, color=color, linestyle='--', alpha=0.3)
    
    # Add legend lines for diagnosis and censoring
    ax2.axvline(x=0, linestyle='-', color='gray', alpha=0.3, label='Diagnosis Time')
    ax2.axvline(x=0, linestyle='--', color='gray', alpha=0.3, label='Censoring Time')
    
    ax2.set_title('Individual Patient Trajectories with Diagnosis and Censoring Times')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Lambda Value')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig




def create_proper_calibration_plots(checkpoint_path, cov_df, n_bins=10, use_log_scale=True, min_bin_count=1000, save_path=None):
    """Create calibration plots comparing predicted vs observed event rates for at-risk individuals.
    
    Args:
        checkpoint_path: Path to model checkpoint
        cov_df: DataFrame containing enrollment ages
        n_bins: Number of bins for calibration
        use_log_scale: Whether to use log-scale binning (recommended for rare events)
        min_bin_count: Minimum number of samples per bin
        save_path: Path to save plot
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    
    # Get parameters from state dict
    lambda_ = state_dict['lambda_']  # Shape: (N, K, T)
    phi = state_dict['phi']  # Shape: (K, D, T)
    kappa = state_dict['kappa']  # Shape: scalar
    Y = checkpoint['Y']  # Shape: (N, D, T)
    
    # Calculate theta (normalized lambda)
    theta = torch.softmax(lambda_, dim=1)
    
    # Calculate phi probabilities (sigmoid)
    phi_prob = torch.sigmoid(phi)
    
    # Calculate pi (disease probabilities)
    pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * kappa
    
    # Convert to numpy
    pi_np = pi.detach().numpy()
    Y_np = Y.detach().numpy()
    
    N, D, T = Y_np.shape
    
    # Create at_risk mask
    at_risk = np.ones_like(Y_np, dtype=bool)
    for n in range(N):
        for d in range(D):
            event_times = np.where(Y_np[n,d,:])[0]
            if len(event_times) > 0:
                at_risk[n,d,(event_times[0]+1):] = False
    
    # Create two sets of predictions/observations
    
    # 1. Enrollment only
    enroll_pred = []
    enroll_obs = []
    
    for d in range(D):
        for i, row in enumerate(cov_df.itertuples()):
            enroll_age = row.age
            enroll_time = int(enroll_age - 30)  # Convert age to time index
            
            if enroll_time < 0 or enroll_time >= T:
                continue
                
            if at_risk[i,d,enroll_time]:
                enroll_pred.append(pi_np[i,d,enroll_time])
                enroll_obs.append(Y_np[i,d,enroll_time])
    
    # 2. All follow-up
    all_pred = []
    all_obs = []
    
    for t in range(T):
        mask_t = at_risk[:,:,t]
        if mask_t.sum() > 0:
            all_pred.extend(pi_np[:,:,t][mask_t])
            all_obs.extend(Y_np[:,:,t][mask_t])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    def plot_calibration(pred, obs, ax, title):
        # Create bins in log or linear space
        if use_log_scale:
            bin_edges = np.logspace(np.log10(max(1e-7, min(pred))), 
                                  np.log10(max(pred)), 
                                  n_bins + 1)
        else:
            bin_edges = np.linspace(min(pred), max(pred), n_bins + 1)
        
        # Calculate statistics for each bin
        bin_means = []
        obs_means = []
        counts = []
        
        for i in range(n_bins):
            mask = (pred >= bin_edges[i]) & (pred < bin_edges[i + 1])
            if np.sum(mask) >= min_bin_count:
                bin_means.append(np.mean(pred[mask]))
                obs_means.append(np.mean(obs[mask]))
                counts.append(np.sum(mask))
        
        # Plot
        if use_log_scale:
            ax.plot([1e-7, 1], [1e-7, 1], '--', color='gray', alpha=0.5, label='Perfect calibration')
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.plot([0, max(pred)], [0, max(pred)], '--', color='gray', alpha=0.5, label='Perfect calibration')
        
        ax.plot(bin_means, obs_means, 'o-', color='#1f77b4', 
                markersize=8, linewidth=2, label='Observed rates')
        
        # Add counts as annotations
        for i, (x, y, c) in enumerate(zip(bin_means, obs_means, counts)):
            ax.annotate(f'n={c:,}', (x, y), xytext=(0, 10), 
                       textcoords='offset points', ha='center', fontsize=8)
        
        # Add summary statistics
        mse = np.mean((np.array(bin_means) - np.array(obs_means))**2)
        mean_pred = np.mean(pred)
        mean_obs = np.mean(obs)
        
        stats_text = f'MSE: {mse:.2e}\n'
        stats_text += f'Mean Predicted: {mean_pred:.2e}\n'
        stats_text += f'Mean Observed: {mean_obs:.2e}\n'
        stats_text += f'N total: {sum(counts):,}'
        
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.set_xlabel('Predicted Event Rate', fontsize=12)
        ax.set_ylabel('Observed Event Rate', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(loc='lower right')
    
    # Create both plots
    plot_calibration(np.array(enroll_pred), np.array(enroll_obs), 
                    ax1, 'Calibration at Enrollment\n(At-Risk Only)')
    plot_calibration(np.array(all_pred), np.array(all_obs), 
                    ax2, 'Calibration Across All Follow-up\n(At-Risk Only)')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    
    return fig

# --- Function to Evaluate Cox Models on Test Set with 10-Year AUC ---
def evaluate_cox_baseline_models_auc(fitted_models, Y_test, FH_test, disease_mapping, major_diseases, disease_names, follow_up_duration_years=10, pce_df=None, n_bootstraps=100):
    """
    Evaluates pre-fitted Cox models on the test set using 10-year risk AUC (C-statistic).
    For each subject, computes the predicted 10-year risk and the binary outcome (event within 10 years).
    Returns AUC and bootstrap CIs for each disease group.
    """
    from lifelines.utils import concordance_index
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import pandas as pd
    test_results = {}
    print("\nEvaluating Cox models on test data (10-year AUC)...")

    if not (len(Y_test) == len(FH_test)):
        raise ValueError(f"Test data size mismatch: Y_test ({len(Y_test)}), FH_test ({len(FH_test)})")

    FH_test = FH_test.reset_index(drop=True)
    if pce_df is not None:
        pce_df = pce_df.reset_index(drop=True)

    for disease_group, model in fitted_models.items():
        if model is None:
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue

        print(f" - Evaluating {disease_group}...")
        fh_cols = disease_mapping.get(disease_group, [])

        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1

        if target_sex_code is not None:
            mask_test = (FH_test['sex'] == target_sex_code)
        else:
            mask_test = pd.Series(True, index=FH_test.index)

        current_FH_test = FH_test[mask_test].copy()
        current_Y_test = Y_test[mask_test]
        if len(current_FH_test) == 0:
            print(f"   Warning: No individuals for target sex code {target_sex_code}.")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue

        # Find disease indices
        disease_indices = []
        unique_indices = set()
        for disease in major_diseases.get(disease_group, []):
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_test.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)

        if not disease_indices:
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue

        # Prepare data for evaluation
        eval_data = []
        for i in range(len(current_FH_test)):
            age_at_enrollment = current_FH_test.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)
            if t_enroll < 0 or t_enroll >= current_Y_test.shape[2]:
                continue
            end_time = min(t_enroll + follow_up_duration_years, current_Y_test.shape[2])
            if end_time <= t_enroll:
                continue
            # For each disease in group, check for event in window
            had_event = 0
            event_time = None
            for d_idx in disease_indices:
                Y_slice = current_Y_test[i, d_idx, t_enroll:end_time]
                if (torch.is_tensor(Y_slice) and torch.any(Y_slice > 0)) or \
                   (isinstance(Y_slice, np.ndarray) and np.any(Y_slice > 0)):
                    had_event = 1
                    # For time-to-event, could use: event_time = np.where(Y_slice > 0)[0][0] + t_enroll
                    break
            # Prepare row for Cox prediction
            row = {
                'age': age_at_enrollment,
                'sex': current_FH_test.iloc[i]['sex']
            }
            if fh_cols:
                valid_fh_cols = [col for col in fh_cols if col in current_FH_test.columns]
                if valid_fh_cols:
                    row['fh'] = current_FH_test.iloc[i][valid_fh_cols].any()
            eval_data.append((row, had_event))

        if not eval_data:
            print("   Warning: No individuals processed for evaluation.")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue

        # Build DataFrame for prediction
        pred_df = pd.DataFrame([row for row, _ in eval_data])
        outcomes = np.array([had_event for _, had_event in eval_data])
        n_total = len(outcomes)
        n_events = int(np.sum(outcomes))

        # Predict 10-year risk for each subject
        # Use model.predict_survival_function to get S(t) at 10 years after enrollment
        # If model uses 'age' as time scale, predict at age+10
        try:
            surv_funcs = model.predict_survival_function(pred_df)
            # Each column is a subject, index is time (age)
            # For each subject, find S(age+10)
            ten_year_risks = []
            for i, row in pred_df.iterrows():
                age0 = row['age']
                target_age = age0 + follow_up_duration_years
                # Find closest time in survival function index
                surv = surv_funcs[i]
                # If index is not integer ages, interpolate
                if target_age in surv.index:
                    s10 = surv.loc[target_age]
                else:
                    s10 = np.interp(target_age, surv.index.values, surv.values)
                risk10 = 1 - s10
                ten_year_risks.append(risk10)
            ten_year_risks = np.array(ten_year_risks)
        except Exception as e:
            print(f"   Error predicting 10-year risk: {e}")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': n_events, 'n_total': n_total}
            continue

        # Compute AUC
        try:
            if len(np.unique(outcomes)) > 1:
                auc = roc_auc_score(outcomes, ten_year_risks)
                # Bootstrap CIs
                aucs = []
                for _ in range(n_bootstraps):
                    idx = np.random.choice(n_total, n_total, replace=True)
                    if len(np.unique(outcomes[idx])) > 1:
                        aucs.append(roc_auc_score(outcomes[idx], ten_year_risks[idx]))
                if aucs:
                    ci_lower = np.percentile(aucs, 2.5)
                    ci_upper = np.percentile(aucs, 97.5)
                else:
                    ci_lower = ci_upper = np.nan
            else:
                auc = np.nan
                ci_lower = ci_upper = np.nan
                print(f"   Warning: Only one class present for AUC.")
        except Exception as e:
            print(f"   Error computing AUC: {e}")
            auc = np.nan
            ci_lower = ci_upper = np.nan

        test_results[disease_group] = {
            'auc': auc,
            'ci': (ci_lower, ci_upper),
            'n_events': n_events,
            'n_total': n_total
        }
        print(f"   10-year AUC: {auc:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) | Events: {n_events}/{n_total}")
    print("Finished evaluating Cox models (10-year AUC).")
    return test_results



## add the aladynoulli results to the cox results
## do the PCE updated each year

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def fit_and_eval_glm_baseline_models(Y_full, FH_processed, train_indices, test_indices, disease_mapping, major_diseases, disease_names, follow_up_duration_years=10):
    """
    Fit and evaluate GLM (logistic regression) for 10-year risk prediction for each disease group.
    Uses enrollment covariates (age, sex, FH if available) and binary outcome (event in 10 years).
    Returns dict of AUCs and predictions for each group.
    """
    results = {}
    Y_train = Y_full[train_indices]
    Y_test = Y_full[test_indices]
    FH_train = FH_processed.iloc[train_indices].reset_index(drop=True)
    FH_test = FH_processed.iloc[test_indices].reset_index(drop=True)
    
    for disease_group, disease_names_list in major_diseases.items():
        fh_cols = disease_mapping.get(disease_group, [])
        print(f"- Fitting GLM for {disease_group}...")
        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1
        
        # --- Prepare training data ---
        if target_sex_code is not None:
            mask_train = (FH_train['sex'] == target_sex_code)
        else:
            mask_train = pd.Series(True, index=FH_train.index)
        current_FH_train = FH_train[mask_train].copy()
        current_Y_train = Y_train[mask_train]
        
        # --- Prepare test data ---
        if target_sex_code is not None:
            mask_test = (FH_test['sex'] == target_sex_code)
        else:
            mask_test = pd.Series(True, index=FH_test.index)
        current_FH_test = FH_test[mask_test].copy()
        current_Y_test = Y_test[mask_test]
        
        # --- Find disease indices ---
        disease_indices = []
        unique_indices = set()
        for disease in disease_names_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_full.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        if not disease_indices:
            print(f"  No valid indices for {disease_group}.")
            results[disease_group] = {'auc': np.nan, 'n_events': 0, 'n_total': 0}
            continue
        
        # --- Build X and y for train/test ---
        def build_Xy(FH, Y, disease_indices):
            X = []
            y = []
            for i in range(len(FH)):
                age = FH.iloc[i]['age']
                sex = FH.iloc[i]['sex']
                t_enroll = int(age - 30)
                if t_enroll < 0 or t_enroll >= Y.shape[2]:
                    continue
                end_time = min(t_enroll + follow_up_duration_years, Y.shape[2])
                # Binary label: 1 if any event in group in 10 years, else 0
                had_event = 0
                for d_idx in disease_indices:
                    if torch.any(Y[i, d_idx, t_enroll:end_time] > 0):
                        had_event = 1
                        break
                # Covariates: age, sex, FH columns if available
                row = [age, sex]
                if fh_cols:
                    valid_fh_cols = [col for col in fh_cols if col in FH.columns]
                    if valid_fh_cols:
                        row.append(FH.iloc[i][valid_fh_cols].any())
                X.append(row)
                y.append(had_event)
            return np.array(X), np.array(y)
        
        X_train, y_train = build_Xy(current_FH_train, current_Y_train, disease_indices)
        X_test, y_test = build_Xy(current_FH_test, current_Y_test, disease_indices)
        
        if np.sum(y_train) < 5 or np.sum(y_test) < 5:
            print(f"  Too few events for {disease_group} (train: {np.sum(y_train)}, test: {np.sum(y_test)})")
            results[disease_group] = {'auc': np.nan, 'n_events': int(np.sum(y_test)), 'n_total': len(y_test)}
            continue
        
        # --- Fit GLM ---
        glm = LogisticRegression(max_iter=1000)
        glm.fit(X_train, y_train)
        y_pred = glm.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        results[disease_group] = {
            'auc': auc,
            'n_events': int(np.sum(y_test)),
            'n_total': len(y_test),
            'y_pred': y_pred,
            'y_true': y_test
        }
        print(f"  GLM AUC for {disease_group}: {auc:.3f} (Events: {np.sum(y_test)}/{len(y_test)})")
    return results




def evaluate_major_diseases_wsex_with_bootstrap_dynamic(model, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, follow_up_duration_years=10, patient_indices=None):
    """
    Evaluate dynamic 10-year risk for each patient using Aladynoulli model, with bootstrap CIs for AUC.
    For each patient, at each year after enrollment, use the model to get the risk for that year.
    The cumulative 10-year risk is 1 - prod(1 - yearly_risks).
    If patient_indices is provided, subset all data to those indices.
    """
    import numpy as np
    import torch
    import pandas as pd
    from sklearn.metrics import roc_curve, auc

    major_diseases = {
        'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                  'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': ['Colon cancer', 'Cancer of bronchus; lung', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm','Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver'],
        'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
        'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],
        'Prostate_Cancer': ['Cancer of prostate'],
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
    }

    results = {}
    if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")

    # Subset all data if patient_indices is provided
    if patient_indices is not None:
        Y_100k = Y_100k[patient_indices]
        E_100k = E_100k[patient_indices]
        pce_df = pce_df.iloc[patient_indices].reset_index(drop=True)

    with torch.no_grad():
        pi, _, _ = model.forward()
    if patient_indices is not None:
        pi = pi[patient_indices]
    N_pi = pi.shape[0]
    N_pce = len(pce_df)
    N_y100k = Y_100k.shape[0]
    if not (N_pi == N_pce == N_y100k):
        print(f"Warning: Size mismatch for evaluation cohort. pi: {N_pi}, pce_df: {N_pce}, Y_100k: {N_y100k}. Using minimum size.")
        min_N = min(N_pi, N_pce, N_y100k)
        pi = pi[:min_N]
        pce_df = pce_df.iloc[:min_N]
        Y_100k = Y_100k[:min_N]
    pce_df = pce_df.reset_index(drop=True)

    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group} (Dynamic 10-Year Risk)...")
        disease_indices = []
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        max_model_disease_idx = pi.shape[1] - 1
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
            print(f"No valid matching disease indices found for {disease_group}.")
            results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan}
            continue

        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'
        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]
        if target_sex:
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} individuals in cohort")
            if len(int_indices_pce) == 0:
                print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan}
                continue
        if len(int_indices_pce) == 0:
            auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
            ci_lower = np.nan; ci_upper = np.nan
        else:
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce]
            current_N_auc = len(int_indices_pce)
            risks_auc = np.zeros(current_N_auc)
            outcomes_auc = np.zeros(current_N_auc)
            processed_indices_auc_final = []
            n_prevalent_excluded = 0
            for i in range(current_N_auc):
                age = current_pce_df_auc.iloc[i]['age']
                t_enroll = int(age - 30)
                if t_enroll < 0 or t_enroll + follow_up_duration_years >= current_pi_auc.shape[2]:
                    continue
                # INCIDENT DISEASE FILTER: Only for single-disease outcomes
                if len(disease_indices) == 1:
                    prevalent = False
                    for d_idx in disease_indices:
                        if d_idx >= current_Y_100k_auc.shape[1]:
                            continue
                        if torch.any(current_Y_100k_auc[i, d_idx, :t_enroll] > 0):
                            prevalent = True
                            break
                    if prevalent:
                        n_prevalent_excluded += 1
                        continue
                # Collect yearly risks for years 1 to 10 after enrollment
                yearly_risks = []
                for t in range(1, follow_up_duration_years + 1):
                    pi_diseases = current_pi_auc[i, disease_indices, t_enroll + t]
                    yearly_risk = 1 - torch.prod(1 - pi_diseases)
                    yearly_risks.append(yearly_risk.item())
                # Compute cumulative 10-year risk, but might be a problem if diseaes occurs, so we should really do td cox
                survival_prob = np.prod([1 - r for r in yearly_risks])
                ten_year_risk = 1 - survival_prob
                risks_auc[i] = ten_year_risk
                # Outcome: did any event occur in the 10 years after enrollment? is this ok because not everyone will have 10 years of follow-up?
                end_time = min(t_enroll + follow_up_duration_years, current_Y_100k_auc.shape[2])
                event_found = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0):
                        outcomes_auc[i] = 1
                        event_found = True
                        break
                processed_indices_auc_final.append(i)
            if not processed_indices_auc_final:
                auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
                ci_lower = np.nan; ci_upper = np.nan
            else:
                risks_np = risks_auc[processed_indices_auc_final]
                outcomes_np = outcomes_auc[processed_indices_auc_final]
                n_processed = len(outcomes_np)
                if len(np.unique(outcomes_np)) > 1:
                    fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
                    auc_score = auc(fpr, tpr)
                    aucs = []
                    for _ in range(n_bootstraps):
                        indices = np.random.choice(len(risks_np), size=len(risks_np), replace=True)
                        if len(np.unique(outcomes_np[indices])) > 1:
                            fpr_boot, tpr_boot, _ = roc_curve(outcomes_np[indices], risks_np[indices])
                            bootstrap_auc = auc(fpr_boot, tpr_boot)
                            aucs.append(bootstrap_auc)
                    if aucs:
                        ci_lower = np.percentile(aucs, 2.5)
                        ci_upper = np.percentile(aucs, 97.5)
                    else:
                        ci_lower = ci_upper = np.nan
                else:
                    auc_score = np.nan
                    ci_lower = ci_upper = np.nan
                    print(f"Warning: Only one class present ({np.unique(outcomes_np)}) for AUC.")
                n_events = int(np.sum(outcomes_np))
                event_rate = (n_events / n_processed * 100) if n_processed > 0 else 0.0
        results[disease_group] = {
            'auc': auc_score,
            'n_events': n_events,
            'event_rate': event_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        print(f"AUC: {auc_score:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) (calculated on {n_processed} individuals)")
        print(f"Events (10-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {n_processed} individuals)")
        print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
    print(f"\nSummary of Results (Dynamic 10-Year Risk, Sex-Adjusted):")
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str}")
    print("-" * 80)
    return results




def evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year(model, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, follow_up_duration_years=1, patient_indices=None):
    """
    Evaluate 1-year risk for each patient using Aladynoulli model, with bootstrap CIs for AUC.
    For each patient, use the model to get the risk for the first year after enrollment.
    If patient_indices is provided, subset all data to those indices.
    """
    import numpy as np
    import torch
    import pandas as pd
    from sklearn.metrics import roc_curve, auc

    major_diseases = {
        'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                  'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': ['Colon cancer', 'Cancer of bronchus; lung', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm','Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver'],
        'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
        'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],
        'Prostate_Cancer': ['Cancer of prostate'],
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
    }

    results = {}
    if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")

    # Subset all data if patient_indices is provided
    if patient_indices is not None:
        Y_100k = Y_100k[patient_indices]
        E_100k = E_100k[patient_indices]
        pce_df = pce_df.iloc[patient_indices].reset_index(drop=True)

    with torch.no_grad():
        pi, _, _ = model.forward()
    if patient_indices is not None:
        pi = pi[patient_indices]
    N_pi = pi.shape[0]
    N_pce = len(pce_df)
    N_y100k = Y_100k.shape[0]
    if not (N_pi == N_pce == N_y100k):
        print(f"Warning: Size mismatch for evaluation cohort. pi: {N_pi}, pce_df: {N_pce}, Y_100k: {N_y100k}. Using minimum size.")
        min_N = min(N_pi, N_pce, N_y100k)
        pi = pi[:min_N]
        pce_df = pce_df.iloc[:min_N]
        Y_100k = Y_100k[:min_N]
    pce_df = pce_df.reset_index(drop=True)

    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group} (1-Year Risk)...")
        disease_indices = []
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        max_model_disease_idx = pi.shape[1] - 1
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
            print(f"No valid matching disease indices found for {disease_group}.")
            results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan, 'c_index': np.nan}
            continue

        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'
        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]
        if target_sex:
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} individuals in cohort")
            if len(int_indices_pce) == 0:
                print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan, 'c_index': np.nan}
                continue
        if len(int_indices_pce) == 0:
            auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
            ci_lower = np.nan; ci_upper = np.nan
            c_index = np.nan
        else:
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce]
            current_N_auc = len(int_indices_pce)
            risks_auc = torch.zeros(current_N_auc, device=pi.device)
            outcomes_auc = torch.zeros(current_N_auc, device=pi.device)
            processed_indices_auc_final = [] 
            # For C-index
            age_enrolls = []
            age_at_events = []
            event_indicators = []
            n_prevalent_excluded = 0
            for i in range(current_N_auc):
                age = current_pce_df_auc.iloc[i]['age']
                t_enroll = int(age - 30)
                if t_enroll < 0 or t_enroll >= current_pi_auc.shape[2]:
                    continue
                # INCIDENT DISEASE FILTER: Only for single-disease outcomes
                if len(disease_indices) == 1:
                    prevalent = False
                    for d_idx in disease_indices:
                        if d_idx >= current_Y_100k_auc.shape[1]:
                            continue
                        if torch.any(current_Y_100k_auc[i, d_idx, :t_enroll] > 0):
                            prevalent = True
                            break
                    if prevalent:
                        n_prevalent_excluded += 1
                        continue
                # Store risk for ALL valid enrollment times
                pi_diseases = current_pi_auc[i, disease_indices, t_enroll]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                risks_auc[i] = yearly_risk
                end_time = min(t_enroll + follow_up_duration_years, current_Y_100k_auc.shape[2]) 
                if end_time <= t_enroll: continue
                # --- C-index: Find time-to-event and event indicator ---
                age_enroll = t_enroll + 30
                age_at_event = end_time + 30 - 1
                event = 0
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    event_times = torch.where(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0)[0]
                    if len(event_times) > 0:
                        this_event_age = t_enroll + event_times[0].item() + 30
                        if this_event_age < age_at_event:
                            age_at_event = this_event_age
                            event = 1
                age_enrolls.append(age_enroll)
                age_at_events.append(age_at_event)
                event_indicators.append(event)
                # --- Outcome: Check event in next follow_up_duration_years ---
                event_found_auc = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0): 
                        outcomes_auc[i] = 1
                        event_found_auc = True
                        break
                processed_indices_auc_final.append(i) 
            if not processed_indices_auc_final:
                 auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
                 ci_lower = np.nan; ci_upper = np.nan
                 c_index = np.nan
            else:
                 # Get risks/outcomes only for AUC calculation
                 risks_np = risks_auc[processed_indices_auc_final].cpu().numpy()
                 outcomes_np = outcomes_auc[processed_indices_auc_final].cpu().numpy()
                 n_processed = len(outcomes_np)
                 # For C-index, filter to processed indices
                 age_enrolls_np = np.array(age_enrolls)[processed_indices_auc_final]
                 age_at_events_np = np.array(age_at_events)[processed_indices_auc_final]
                 event_indicators_np = np.array(event_indicators)[processed_indices_auc_final]
                 durations = age_at_events_np - age_enrolls_np
                 # Calculate C-index
                 from lifelines.utils import concordance_index
                 try:
                     c_index = concordance_index(durations, risks_np, event_indicators_np)
                 except Exception as e:
                     print(f"C-index calculation failed: {e}")
                     c_index = np.nan
                 if disease_group in ["Bipolar_Disorder", "Depression"]:
                    df = pd.DataFrame({
                        "risk": risks_np,
                        "outcome": outcomes_np
                    })
                    df.to_csv(f"debug_{disease_group}.csv", index=False)
                 # Calculate AUC using roc_curve + auc consistently
                 if len(np.unique(outcomes_np)) > 1:
                      fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
                      auc_score = auc(fpr, tpr)
                      # Bootstrap CI calculation using same method
                      aucs = []
                      for _ in range(n_bootstraps):
                          indices = np.random.choice(len(risks_np), size=len(risks_np), replace=True)
                          if len(np.unique(outcomes_np[indices])) > 1:
                              fpr_boot, tpr_boot, _ = roc_curve(outcomes_np[indices], risks_np[indices])
                              bootstrap_auc = auc(fpr_boot, tpr_boot)
                              aucs.append(bootstrap_auc)
                      if aucs:
                          ci_lower = np.percentile(aucs, 2.5)
                          ci_upper = np.percentile(aucs, 97.5)
                      else:
                          ci_lower = ci_upper = np.nan
                 else:
                      auc_score = np.nan
                      ci_lower = ci_upper = np.nan
                      print(f"Warning: Only one class present ({np.unique(outcomes_np)}) for AUC.")
                 # Calculate events using ALL outcomes
                 n_events = int(torch.sum(outcomes_auc).item())
                 event_rate = (n_events / current_N_auc * 100)
        results[disease_group] = {
            'auc': auc_score,
            'n_events': n_events,
            'event_rate': event_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'c_index': c_index
        }
        print(f"AUC: {auc_score:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) (calculated on {n_processed} individuals)") 
        print(f"C-index: {c_index:.3f} (calculated on {n_processed} individuals)")
        print(f"Events ({follow_up_duration_years}-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {current_N_auc} individuals)") 
        print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
    print(f"\nSummary of Results (1-Year Risk, Sex-Adjusted):")
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10} {'C-index':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        c_index_str = f"{res['c_index']:.3f}" if not np.isnan(res['c_index']) else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str:<10} {c_index_str}")
    print("-" * 80)
    return results


def evaluate_cox_baseline_models_1yrscore_10youtcome_auc(fitted_models, Y_test, FH_test, disease_mapping, major_diseases, disease_names, follow_up_duration_years=10, n_bootstraps=100):
    """
    Evaluates Cox models using 1-year risk at enrollment as the score, and 10-year event as the outcome.
    Returns AUC and bootstrap CIs for each disease group.
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import pandas as pd
    print("\nEvaluating Cox models (1-year risk at enrollment, 10-year outcome AUC)...")
    test_results = {}
    if not (len(Y_test) == len(FH_test)):
        raise ValueError(f"Test data size mismatch: Y_test ({len(Y_test)}), FH_test ({len(FH_test)})")
    FH_test = FH_test.reset_index(drop=True)
    for disease_group, model in fitted_models.items():
        if model is None:
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        print(f" - Evaluating {disease_group}...")
        fh_cols = disease_mapping.get(disease_group, [])
        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1
        if target_sex_code is not None:
            mask_test = (FH_test['sex'] == target_sex_code)
        else:
            mask_test = pd.Series(True, index=FH_test.index)
        current_FH_test = FH_test[mask_test].copy()
        current_Y_test = Y_test[mask_test]
        if len(current_FH_test) == 0:
            print(f"   Warning: No individuals for target sex code {target_sex_code}.")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        # Find disease indices
        disease_indices = []
        unique_indices = set()
        for disease in major_diseases.get(disease_group, []):
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_test.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        if not disease_indices:
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        eval_data = []
        for i in range(len(current_FH_test)):
            age_at_enrollment = current_FH_test.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)
            if t_enroll < 0 or t_enroll >= current_Y_test.shape[2]:
                continue
            end_time = min(t_enroll + follow_up_duration_years, current_Y_test.shape[2])
            if end_time <= t_enroll:
                continue
            # For each disease in group, check for event in window
            had_event = 0
            for d_idx in disease_indices:
                Y_slice = current_Y_test[i, d_idx, t_enroll:end_time]
                if (torch.is_tensor(Y_slice) and torch.any(Y_slice > 0)) or \
                   (isinstance(Y_slice, np.ndarray) and np.any(Y_slice > 0)):
                    had_event = 1
                    break
            # Prepare row for Cox prediction
            row = {
                'age': age_at_enrollment,
                'sex': current_FH_test.iloc[i]['sex']
            }
            if fh_cols:
                valid_fh_cols = [col for col in fh_cols if col in current_FH_test.columns]
                if valid_fh_cols:
                    row['fh'] = current_FH_test.iloc[i][valid_fh_cols].any()
            eval_data.append((row, had_event))
        if not eval_data:
            print("   Warning: No individuals processed for evaluation.")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        pred_df = pd.DataFrame([row for row, _ in eval_data])
        outcomes = np.array([had_event for _, had_event in eval_data])
        n_total = len(outcomes)
        n_events = int(np.sum(outcomes))
        # Predict 1-year risk at enrollment
        try:
            surv_funcs = model.predict_survival_function(pred_df)
            one_year_risks = []
            for i, row in pred_df.iterrows():
                age0 = row['age']
                s0 = surv_funcs[i].loc[age0] if age0 in surv_funcs[i].index else np.interp(age0, surv_funcs[i].index.values, surv_funcs[i].values)
                s1 = surv_funcs[i].loc[age0+1] if (age0+1) in surv_funcs[i].index else np.interp(age0+1, surv_funcs[i].index.values, surv_funcs[i].values)
                risk1 = 1 - s1 / s0 if s0 > 0 else 0.0
                one_year_risks.append(risk1)
            one_year_risks = np.array(one_year_risks)
        except Exception as e:
            print(f"   Error predicting 1-year risk: {e}")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': n_events, 'n_total': n_total}
            continue
        # Compute AUC
        try:
            if len(np.unique(outcomes)) > 1:
                auc_val = roc_auc_score(outcomes, one_year_risks)
                aucs = []
                for _ in range(n_bootstraps):
                    idx = np.random.choice(n_total, n_total, replace=True)
                    if len(np.unique(outcomes[idx])) > 1:
                        aucs.append(roc_auc_score(outcomes[idx], one_year_risks[idx]))
                if aucs:
                    ci_lower = np.percentile(aucs, 2.5)
                    ci_upper = np.percentile(aucs, 97.5)
                else:
                    ci_lower = ci_upper = np.nan
            else:
                auc_val = np.nan
                ci_lower = ci_upper = np.nan
                print(f"   Warning: Only one class present for AUC.")
        except Exception as e:
            print(f"   Error computing AUC: {e}")
            auc_val = np.nan
            ci_lower = ci_upper = np.nan
        test_results[disease_group] = {
            'auc': auc_val,
            'ci': (ci_lower, ci_upper),
            'n_events': n_events,
            'n_total': n_total
        }
        print(f"   1-year risk AUC (10-year outcome): {auc_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) | Events: {n_events}/{n_total}")
    print("Finished evaluating Cox models (1-year risk, 10-year outcome AUC).")
    return test_results

def evaluate_cox_baseline_models_auc_with_aladynoulli(fitted_models, Y_test, FH_test, disease_mapping, major_diseases, disease_names, aladynoulli_1yr_risk, follow_up_duration_years=10, n_bootstraps=100):
    """
    Evaluates Cox models with Aladynoulli 1-year risk at enrollment as an additional covariate.
    For each disease group, fits a new Cox model with standard covariates plus 'aladynoulli_1yr',
    predicts 10-year risk, and computes AUC (with bootstrap CIs).
    """
    from lifelines import CoxPHFitter
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import pandas as pd
    print("\nEvaluating Cox models (with Aladynoulli 1-year risk as covariate, 10-year AUC)...")
    test_results = {}
    if not (len(Y_test) == len(FH_test) == len(aladynoulli_1yr_risk)):
        raise ValueError(f"Test data size mismatch: Y_test ({len(Y_test)}), FH_test ({len(FH_test)}), aladynoulli_1yr_risk ({len(aladynoulli_1yr_risk)})")
    FH_test = FH_test.reset_index(drop=True)
    aladynoulli_1yr_risk = np.array(aladynoulli_1yr_risk)
    for disease_group, model in fitted_models.items():
        print(f" - Evaluating {disease_group}...")
        fh_cols = disease_mapping.get(disease_group, [])
        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1
        if target_sex_code is not None:
            mask_test = (FH_test['sex'] == target_sex_code)
        else:
            mask_test = pd.Series(True, index=FH_test.index)
        current_FH_test = FH_test[mask_test].copy()
        current_Y_test = Y_test[mask_test]
        current_aladyn_1yr = aladynoulli_1yr_risk[mask_test.values]
        if len(current_FH_test) == 0:
            print(f"   Warning: No individuals for target sex code {target_sex_code}.")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        # Find disease indices
        disease_indices = []
        unique_indices = set()
        for disease in major_diseases.get(disease_group, []):
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_test.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        if not disease_indices:
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        eval_data = []
        for i in range(len(current_FH_test)):
            age_at_enrollment = current_FH_test.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)
            if t_enroll < 0 or t_enroll >= current_Y_test.shape[2]:
                continue
            end_time = min(t_enroll + follow_up_duration_years, current_Y_test.shape[2])
            if end_time <= t_enroll:
                continue
            # For each disease in group, check for event in window
            had_event = 0
            for d_idx in disease_indices:
                Y_slice = current_Y_test[i, d_idx, t_enroll:end_time]
                if (torch.is_tensor(Y_slice) and torch.any(Y_slice > 0)) or \
                   (isinstance(Y_slice, np.ndarray) and np.any(Y_slice > 0)):
                    had_event = 1
                    break
            row = {
                'age': age_at_enrollment,
                'sex': current_FH_test.iloc[i]['sex'],
                'aladynoulli_1yr': current_aladyn_1yr[i]
            }
            if fh_cols:
                valid_fh_cols = [col for col in fh_cols if col in current_FH_test.columns]
                if valid_fh_cols:
                    row['fh'] = current_FH_test.iloc[i][valid_fh_cols].any()
            eval_data.append((row, had_event))
        if not eval_data:
            print("   Warning: No individuals processed for evaluation.")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        pred_df = pd.DataFrame([row for row, _ in eval_data])
        outcomes = np.array([had_event for _, had_event in eval_data])
        n_total = len(outcomes)
        n_events = int(np.sum(outcomes))
        # Fit new Cox model with aladynoulli_1yr as covariate
        cph = CoxPHFitter()
        # Build formula
        formula = 'sex + aladynoulli_1yr'
        if 'fh' in pred_df.columns:
            formula += ' + fh'
        # For sex-specific, drop sex from formula
        if disease_group in ['Breast_Cancer', 'Prostate_Cancer']:
            formula = 'aladynoulli_1yr'
            if 'fh' in pred_df.columns:
                formula += ' + fh'
        # Prepare DataFrame for CoxPHFitter
        cox_df = pred_df.copy()
        cox_df['event'] = outcomes
        cox_df['duration'] = follow_up_duration_years  # All have same follow-up for risk prediction
        try:
            cph.fit(cox_df, duration_col='duration', event_col='event', formula=formula)
            # Predict survival at 10 years
            surv_funcs = cph.predict_survival_function(pred_df)
            ten_year_risks = []
            for i in range(surv_funcs.shape[1]):
                surv = surv_funcs.iloc[:, i]
                timeline = surv_funcs.index.values
                # Interpolate at 10 years
                s10 = np.interp(follow_up_duration_years, timeline, surv.values)
                ten_year_risks.append(1 - s10)
            ten_year_risks = np.array(ten_year_risks)
        except Exception as e:
            print(f"   Error fitting or predicting: {e}")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': n_events, 'n_total': n_total}
            continue
        # Compute AUC
        try:
            if len(np.unique(outcomes)) > 1:
                auc_val = roc_auc_score(outcomes, ten_year_risks)
                aucs = []
                for _ in range(n_bootstraps):
                    idx = np.random.choice(n_total, n_total, replace=True)
                    if len(np.unique(outcomes[idx])) > 1:
                        aucs.append(roc_auc_score(outcomes[idx], ten_year_risks[idx]))
                if aucs:
                    ci_lower = np.percentile(aucs, 2.5)
                    ci_upper = np.percentile(aucs, 97.5)
                else:
                    ci_lower = ci_upper = np.nan
            else:
                auc_val = np.nan
                ci_lower = ci_upper = np.nan
                print(f"   Warning: Only one class present for AUC.")
        except Exception as e:
            print(f"   Error computing AUC: {e}")
            auc_val = np.nan
            ci_lower = ci_upper = np.nan
        test_results[disease_group] = {
            'auc': auc_val,
            'ci': (ci_lower, ci_upper),
            'n_events': n_events,
            'n_total': n_total
        }
        print(f"   10-year AUC (Cox+Aladynoulli 1yr): {auc_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) | Events: {n_events}/{n_total}")
    print("Finished evaluating Cox models (with Aladynoulli 1-year risk, 10-year AUC).")
    return test_results

def fit_cox_baseline_models_with_aladynoulli(Y_full, FH_processed, aladynoulli_1yr_risk, train_indices, disease_mapping, major_diseases, disease_names, follow_up_duration_years=10):
    """
    Fits Cox models on the training set, including Aladynoulli 1-year risk as an additional covariate.
    """
    from lifelines import CoxPHFitter
    import pandas as pd
    fitted_models = {}
    print(f"Fitting Cox models (with Aladynoulli 1-year risk) using training indices [{train_indices.min()}:{train_indices.max()+1}]...")
    try:
        Y_train = Y_full[train_indices]
        FH_train = FH_processed.iloc[train_indices].reset_index(drop=True)
        aladyn_train = aladynoulli_1yr_risk[train_indices]
    except IndexError as e:
        raise IndexError(f"Training indices out of bounds. Error: {e}") from e
    for disease_group, disease_names_list in major_diseases.items():
        fh_cols = disease_mapping.get(disease_group, [])
        print(f" - Fitting {disease_group}...")
        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1
        if target_sex_code is not None:
            mask_train = (FH_train['sex'] == target_sex_code)
        else:
            mask_train = pd.Series(True, index=FH_train.index)
        current_FH_train = FH_train[mask_train].copy()
        current_Y_train = Y_train[mask_train]
        current_aladyn_train = aladyn_train[mask_train.values]
        if len(current_FH_train) == 0:
            print(f"   Warning: No individuals for target sex code {target_sex_code} in training slice.")
            fitted_models[disease_group] = None
            continue
        # Find disease indices
        disease_indices = []
        unique_indices = set()
        for disease in disease_names_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_full.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        if not disease_indices:
            fitted_models[disease_group] = None
            continue
        cox_data = []
        n_prevalent_excluded = 0
        for i in range(len(current_FH_train)):
            age_at_enrollment = current_FH_train.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)
            if t_enroll < 0 or t_enroll >= current_Y_train.shape[2]:
                continue
            if len(disease_indices) == 1:
                d_idx = disease_indices[0]
                if torch.any(current_Y_train[i, d_idx, :t_enroll] > 0):
                    n_prevalent_excluded += 1
                    continue
            end_time = min(t_enroll + follow_up_duration_years, current_Y_train.shape[2])
            if end_time <= t_enroll:
                continue
            for d_idx in disease_indices:
                Y_slice = current_Y_train[i, d_idx, t_enroll:end_time]
                if (torch.is_tensor(Y_slice) and torch.any(Y_slice > 0)) or \
                   (isinstance(Y_slice, np.ndarray) and np.any(Y_slice > 0)):
                    event_time = np.where(Y_slice > 0)[0][0] + t_enroll
                    age_at_event = 30 + event_time
                    event = 1
                else:
                    age_at_event = 30 + end_time - 1
                    event = 0
                row = {
                    'age': age_at_event,
                    'event': event,
                    'sex': current_FH_train.iloc[i]['sex'],
                    'aladynoulli_1yr': current_aladyn_train[i]
                }
                if fh_cols:
                    valid_fh_cols = [col for col in fh_cols if col in current_FH_train.columns]
                    if valid_fh_cols:
                        row['fh'] = current_FH_train.iloc[i][valid_fh_cols].any()
                cox_data.append(row)
        if not cox_data:
            fitted_models[disease_group] = None
            continue
        cox_df = pd.DataFrame(cox_data)
        if cox_df['event'].sum() < 5:
            print(f"   Warning: Too few events ({cox_df['event'].sum()}) for {disease_group}")
            fitted_models[disease_group] = None
            continue
        try:
            cph = CoxPHFitter()
            formula = 'sex + aladynoulli_1yr'
            if 'fh' in cox_df.columns:
                formula += ' + fh'
            if target_sex_code is not None:
                formula = 'aladynoulli_1yr'
                if 'fh' in cox_df.columns:
                    formula += ' + fh'
            cph.fit(cox_df, duration_col='age', event_col='event', formula=formula)
            fitted_models[disease_group] = cph
            print(f"   Model fitted for {disease_group} using {len(cox_df)} samples.")
            if len(disease_indices) == 1:
                print(f"   Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
        except Exception as e:
            print(f"   Error fitting {disease_group}: {e}")
            fitted_models[disease_group] = None
    print("Finished fitting Cox models (with Aladynoulli 1-year risk).")
    return fitted_models

def evaluate_cox_baseline_models_with_aladynoulli(fitted_models, Y_test, FH_test, aladynoulli_1yr_risk, disease_mapping, major_diseases, disease_names, follow_up_duration_years=10, n_bootstraps=100):
    """
    Evaluates fitted Cox models (with Aladynoulli 1-year risk) on the test set.
    For each disease group, predicts 10-year risk, computes AUC (with bootstrap CIs), and prints/returns results.
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import pandas as pd
    print("\nEvaluating Cox models (with Aladynoulli 1-year risk, 10-year AUC)...")
    test_results = {}
    if not (len(Y_test) == len(FH_test) == len(aladynoulli_1yr_risk)):
        raise ValueError(f"Test data size mismatch: Y_test ({len(Y_test)}), FH_test ({len(FH_test)}), aladynoulli_1yr_risk ({len(aladynoulli_1yr_risk)})")
    FH_test = FH_test.reset_index(drop=True)
    aladynoulli_1yr_risk = np.array(aladynoulli_1yr_risk)
    for disease_group, model in fitted_models.items():
        print(f" - Evaluating {disease_group}...")
        fh_cols = disease_mapping.get(disease_group, [])
        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1
        if target_sex_code is not None:
            mask_test = (FH_test['sex'] == target_sex_code)
        else:
            mask_test = pd.Series(True, index=FH_test.index)
        current_FH_test = FH_test[mask_test].copy()
        current_Y_test = Y_test[mask_test]
        current_aladyn_1yr = aladynoulli_1yr_risk[mask_test.values]
        if len(current_FH_test) == 0:
            print(f"   Warning: No individuals for target sex code {target_sex_code}.")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        # Find disease indices
        disease_indices = []
        unique_indices = set()
        for disease in major_diseases.get(disease_group, []):
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_test.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        if not disease_indices:
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        eval_data = []
        for i in range(len(current_FH_test)):
            age_at_enrollment = current_FH_test.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)
            if t_enroll < 0 or t_enroll >= current_Y_test.shape[2]:
                continue
            end_time = min(t_enroll + follow_up_duration_years, current_Y_test.shape[2])
            if end_time <= t_enroll:
                continue
            had_event = 0
            for d_idx in disease_indices:
                Y_slice = current_Y_test[i, d_idx, t_enroll:end_time]
                if (torch.is_tensor(Y_slice) and torch.any(Y_slice > 0)) or \
                   (isinstance(Y_slice, np.ndarray) and np.any(Y_slice > 0)):
                    had_event = 1
                    break
            row = {
                'age': age_at_enrollment,
                'sex': current_FH_test.iloc[i]['sex'],
                'aladynoulli_1yr': current_aladyn_1yr[i]
            }
            if fh_cols:
                valid_fh_cols = [col for col in fh_cols if col in current_FH_test.columns]
                if valid_fh_cols:
                    row['fh'] = current_FH_test.iloc[i][valid_fh_cols].any()
            eval_data.append((row, had_event))
        if not eval_data:
            print("   Warning: No individuals processed for evaluation.")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': 0, 'n_total': 0}
            continue
        pred_df = pd.DataFrame([row for row, _ in eval_data])
        outcomes = np.array([had_event for _, had_event in eval_data])
        n_total = len(outcomes)
        n_events = int(np.sum(outcomes))
        try:
            surv_funcs = model.predict_survival_function(pred_df)
            ten_year_risks = 1 - surv_funcs.loc[follow_up_duration_years].values
        except Exception as e:
            print(f"   Error predicting 10-year risk: {e}")
            test_results[disease_group] = {'auc': np.nan, 'ci': (np.nan, np.nan), 'n_events': n_events, 'n_total': n_total}
            continue
        try:
            if len(np.unique(outcomes)) > 1:
                auc_val = roc_auc_score(outcomes, ten_year_risks)
                aucs = []
                for _ in range(n_bootstraps):
                    idx = np.random.choice(n_total, n_total, replace=True)
                    if len(np.unique(outcomes[idx])) > 1:
                        aucs.append(roc_auc_score(outcomes[idx], ten_year_risks[idx]))
                if aucs:
                    ci_lower = np.percentile(aucs, 2.5)
                    ci_upper = np.percentile(aucs, 97.5)
                else:
                    ci_lower = ci_upper = np.nan
            else:
                auc_val = np.nan
                ci_lower = ci_upper = np.nan
                print(f"   Warning: Only one class present for AUC.")
        except Exception as e:
            print(f"   Error computing AUC: {e}")
            auc_val = np.nan
            ci_lower = ci_upper = np.nan
        test_results[disease_group] = {
            'auc': auc_val,
            'ci': (ci_lower, ci_upper),
            'n_events': n_events,
            'n_total': n_total
        }
        print(f"   10-year AUC (Cox+Aladynoulli 1yr): {auc_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) | Events: {n_events}/{n_total}")
    print("Finished evaluating Cox models (with Aladynoulli 1-year risk, 10-year AUC).")
    return test_results

def dynamic_aladynoulli_auc_for_preexisting(model, Y_100k, E_100k, disease_names, pce_df, preexisting_group, n_bootstraps=100, follow_up_duration_years=10):
    """
    Compute dynamic 10-year ASCVD AUC for patients with pre-existing RA or breast cancer.
    preexisting_group: 'Rheumatoid_Arthritis' or 'Breast_Cancer'
    """
    # Get indices for pre-existing condition
    preexisting_diseases = {
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast']
    }
    ascvd_diseases = [
        'Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease',
        'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'
    ]
    # Find indices for pre-existing
    preexisting_indices = []
    for disease in preexisting_diseases[preexisting_group]:
        preexisting_indices += [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
    # Find indices for ASCVD
    ascvd_indices = []
    for disease in ascvd_diseases:
        ascvd_indices += [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
    # Find patients with pre-existing at enrollment
    preexisting_patients = []
    for i, row in enumerate(pce_df.itertuples()):
        age = row.age
        t_enroll = int(age - 30)
        if t_enroll < 0 or t_enroll >= Y_100k.shape[2]:
            continue
        for d_idx in preexisting_indices:
            if torch.any(Y_100k[i, d_idx, :t_enroll] > 0):
                preexisting_patients.append(i)
                break
    if not preexisting_patients:
        print(f"No patients with {preexisting_group} at enrollment.")
        return ModuleNotFoundError
    # Use dynamic logic for ASCVD only
    results = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
        model, Y_100k, E_100k, disease_names, pce_df,
        n_bootstraps=n_bootstraps, follow_up_duration_years=follow_up_duration_years,
        patient_indices=preexisting_patients
    )
    return results['ASCVD']

from sklearn.metrics import roc_auc_score

def pce_prevent_auc_for_preexisting(Y_100k, pce_df, preexisting_patients, disease_names, follow_up_duration_years=10):
    """
    Compute AUC for PCE and PREVENT for ASCVD in the same patients.
    """
    ascvd_diseases = [
        'Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease',
        'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'
    ]
    ascvd_indices = []
    for disease in ascvd_diseases:
        ascvd_indices += [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
    y_true = []
    for i, row in enumerate(pce_df.itertuples()):
        age = row.age
        t_enroll = int(age - 30)
        if t_enroll < 0 or t_enroll >= Y_100k.shape[2]:
            y_true.append(0)
            continue
        end_time = min(t_enroll + follow_up_duration_years, Y_100k.shape[2])
        had_event = 0
        for d_idx in ascvd_indices:
            if torch.any(Y_100k[i, d_idx, t_enroll:end_time] > 0):
                had_event = 1
                break
        y_true.append(had_event)
    y_true = np.array(y_true)
    mask = np.array([i in preexisting_patients for i in range(len(pce_df))])
    y_true = y_true[mask]
    pce_scores = pce_df.loc[mask, 'pce_goff_fuull'].values
    prevent_scores = pce_df.loc[mask, 'prevent_impute'].values
    auc_pce = roc_auc_score(y_true, pce_scores)
    auc_prevent = roc_auc_score(y_true, prevent_scores)
    print(f"PCE AUC: {auc_pce:.3f}, PREVENT AUC: {auc_prevent:.3f}")
    return auc_pce, auc_prevent



def get_preexisting_patient_indices(Y_100k, pce_df, disease_names, preexisting_group):
    """
    Returns indices of patients with pre-existing RA or breast cancer at enrollment.
    preexisting_group: 'Rheumatoid_Arthritis' or 'Breast_Cancer'
    """
    preexisting_diseases = {
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast']
    }
    preexisting_indices = []
    for disease in preexisting_diseases[preexisting_group]:
        preexisting_indices += [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
    preexisting_patients = []
    for i, row in enumerate(pce_df.itertuples()):
        age = row.age
        t_enroll = int(age - 30)
        if t_enroll < 0 or t_enroll >= Y_100k.shape[2]:
            continue
        for d_idx in preexisting_indices:
            if torch.any(Y_100k[i, d_idx, :t_enroll] > 0):
                preexisting_patients.append(i)
                break
    return preexisting_patients

def fit_and_eval_glm_baseline_models_with_noulli(
    Y_full, FH_processed, train_indices, test_indices, disease_mapping, major_diseases, disease_names,
    aladynoulli_1yr_risk_train, aladynoulli_1yr_risk_test, follow_up_duration_years=10):
    """
    Fit and evaluate GLM (logistic regression) for 10-year risk prediction for each disease group.
    Uses enrollment covariates (age, sex, FH if available, and aladynoulli_1yr_risk) and binary outcome (event in 10 years).
    Returns dict of AUCs and predictions for each group.
    """
    results = {}
    Y_train = Y_full[train_indices]
    Y_test = Y_full[test_indices]
    FH_train = FH_processed.iloc[train_indices].reset_index(drop=True)
    FH_test = FH_processed.iloc[test_indices].reset_index(drop=True)
    aladyn_train = aladynoulli_1yr_risk_train
    aladyn_test = aladynoulli_1yr_risk_test

    for disease_group, disease_names_list in major_diseases.items():
        fh_cols = disease_mapping.get(disease_group, [])
        print(f"- Fitting GLM+Noulli for {disease_group}...")
        target_sex_code = None
        if disease_group == 'Breast_Cancer': target_sex_code = 0
        elif disease_group == 'Prostate_Cancer': target_sex_code = 1

        # --- Prepare training data ---
        if target_sex_code is not None:
            mask_train = (FH_train['sex'] == target_sex_code)
        else:
            mask_train = pd.Series(True, index=FH_train.index)
        current_FH_train = FH_train[mask_train].copy()
        current_Y_train = Y_train[mask_train]
        current_aladyn_train = aladyn_train[mask_train.values]

        # --- Prepare test data ---
        if target_sex_code is not None:
            mask_test = (FH_test['sex'] == target_sex_code)
        else:
            mask_test = pd.Series(True, index=FH_test.index)
        current_FH_test = FH_test[mask_test].copy()
        current_Y_test = Y_test[mask_test]
        current_aladyn_test = aladyn_test[mask_test.values]

        # --- Find disease indices ---
        disease_indices = []
        unique_indices = set()
        for disease in disease_names_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices and idx < Y_full.shape[1]:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        if not disease_indices:
            print(f"  No valid indices for {disease_group}.")
            results[disease_group] = {'auc': np.nan, 'n_events': 0, 'n_total': 0}
            continue

        # --- Build X and y for train/test ---
        def build_Xy(FH, Y, aladyn, disease_indices):
            X = []
            y = []
            for i in range(len(FH)):
                age = FH.iloc[i]['age']
                sex = FH.iloc[i]['sex']
                t_enroll = int(age - 30)
                if t_enroll < 0 or t_enroll >= Y.shape[2]:
                    continue
                end_time = min(t_enroll + follow_up_duration_years, Y.shape[2])
                # Binary label: 1 if any event in group in 10 years, else 0
                had_event = 0
                for d_idx in disease_indices:
                    if torch.any(Y[i, d_idx, t_enroll:end_time] > 0):
                        had_event = 1
                        break
                # Covariates: age, sex, FH columns if available, plus aladynoulli_1yr
                row = [age, sex]
                if fh_cols:
                    valid_fh_cols = [col for col in fh_cols if col in FH.columns]
                    if valid_fh_cols:
                        row.append(FH.iloc[i][valid_fh_cols].any())
                row.append(aladyn[i])
                X.append(row)
                y.append(had_event)
            return np.array(X), np.array(y)

        X_train, y_train = build_Xy(current_FH_train, current_Y_train, current_aladyn_train, disease_indices)
        X_test, y_test = build_Xy(current_FH_test, current_Y_test, current_aladyn_test, disease_indices)

        if np.sum(y_train) < 5 or np.sum(y_test) < 5:
            print(f"  Too few events for {disease_group} (train: {np.sum(y_train)}, test: {np.sum(y_test)})")
            results[disease_group] = {'auc': np.nan, 'n_events': int(np.sum(y_test)), 'n_total': len(y_test)}
            continue

        # --- Fit GLM ---
        glm = LogisticRegression(max_iter=1000)
        glm.fit(X_train, y_train)
        y_pred = glm.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        results[disease_group] = {
            'auc': auc,
            'n_events': int(np.sum(y_test)),
            'n_total': len(y_test),
            'y_pred': y_pred,
            'y_true': y_test
        }
        print(f"  GLM+Noulli AUC for {disease_group}: {auc:.3f} (Events: {np.sum(y_test)}/{len(y_test)})")
    return results


def get_major_disease_1_10year_auc(Y_full, FH_processed, train_indices, test_indices, disease_mapping, major_diseases, disease_names, aladynoulli_1yr_risk_train, aladynoulli_1yr_risk_test, follow_up_duration_years=10):
    """
    Get 10-year AUC for major diseases.
    """
    