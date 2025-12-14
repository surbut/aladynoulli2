"""
Generate 10-year risk calibration curve by age.
Compares mean predicted 10-year risk vs observed 10-year event rate by age group.
Uses the full pi tensor for predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")

# Load full pi tensor
pi_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt")
pi_full = torch.load(pi_path, map_location='cpu')
if torch.is_tensor(pi_full):
    pi_full = pi_full.numpy()
print(f"Loaded pi tensor: shape {pi_full.shape}")

# Load Y tensor for outcomes
Y_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt")
Y_full = torch.load(Y_path, map_location='cpu')
if torch.is_tensor(Y_full):
    Y_full = Y_full.numpy()
print(f"Loaded Y tensor: shape {Y_full.shape}")

# Load corrected E matrix (for at-risk calculations)
E_corrected_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_matrix_corrected.pt")
E_corrected_full = torch.load(E_corrected_path, map_location='cpu')
if torch.is_tensor(E_corrected_full):
    E_corrected_full = E_corrected_full.numpy()
print(f"Loaded E_corrected tensor: shape {E_corrected_full.shape}")

# Load enrollment ages
# Try to find age in baseline_df or pce_df
baseline_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv")
pce_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv")

if baseline_path.exists():
    baseline_df = pd.read_csv(baseline_path, nrows=len(pi_full))
    age_cols = [col for col in baseline_df.columns if 'age' in col.lower() and 'enroll' not in col.lower()]
    if age_cols:
        enrollment_ages = baseline_df[age_cols[0]].values[:len(pi_full)]
        print(f"Using age column from baseline_df: {age_cols[0]}")
    else:
        pce_df = pd.read_csv(pce_path, nrows=len(pi_full))
        enrollment_ages = pce_df['age'].values[:len(pi_full)]
        print("Using age from pce_df")
else:
    pce_df = pd.read_csv(pce_path, nrows=len(pi_full))
    enrollment_ages = pce_df['age'].values[:len(pi_full)]
    print("Using age from pce_df")

# Ensure alignment
min_len = min(len(pi_full), len(Y_full), len(E_corrected_full), len(enrollment_ages))
pi_full = pi_full[:min_len]
Y_full = Y_full[:min_len]
E_corrected_full = E_corrected_full[:min_len]
enrollment_ages = enrollment_ages[:min_len]

print(f"Aligned data: {min_len} patients")

# ============================================================================
# DISEASE DEFINITIONS
# ============================================================================

# ASCVD indices (adjust based on your disease_names)
# For now, we'll use a placeholder - you may need to adjust these
# Based on the image, this is for ASCVD
disease_group = 'ASCVD'
ascvd_disease_names = [
    'Myocardial infarction', 
    'Coronary atherosclerosis', 
    'Other acute and subacute forms of ischemic heart disease',
    'Unstable angina (intermediate coronary syndrome)', 
    'Angina pectoris', 
    'Other chronic ischemic heart disease, unspecified'
]

# Load disease names to find indices
essentials_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt")
essentials = torch.load(essentials_path, map_location='cpu', weights_only=False)
disease_names = essentials.get('disease_names', [])

# Find ASCVD indices
ascvd_indices = []
for disease in ascvd_disease_names:
    indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
    ascvd_indices.extend(indices)
ascvd_indices = list(set([idx for idx in ascvd_indices if idx < pi_full.shape[1]]))

print(f"Found {len(ascvd_indices)} ASCVD disease indices: {ascvd_indices}")

# ============================================================================
# CALCULATE 10-YEAR RISKS AND OUTCOMES BY AGE
# ============================================================================

print("\nCalculating 10-year risks and outcomes by age...")

# Define age bins (e.g., 5-year bins from 40-70)
age_bins = list(range(40, 71, 5))  # [40, 45, 50, 55, 60, 65, 70]
age_bin_labels = [f"{age}-{age+4}" for age in age_bins[:-1]] + [f"{age_bins[-1]}+"]

# Storage for results
age_group_data = []

for age_min in age_bins:
    age_max = age_min + 5 if age_min < 70 else 100
    
    # Find patients in this age group
    age_mask = (enrollment_ages >= age_min) & (enrollment_ages < age_max)
    patient_indices = np.where(age_mask)[0]
    
    if len(patient_indices) == 0:
        continue
    
    # Calculate 10-year risks and outcomes for this age group
    predicted_10yr_risks = []
    observed_10yr_events = []
    predicted_10yr_risks_percentiles = []
    
    n_prevalent_excluded = 0
    n_not_at_risk_excluded = 0
    
    for i in patient_indices:
        age = enrollment_ages[i]
        t_enroll = int(age - 30)
        
        if t_enroll < 0 or t_enroll + 10 >= pi_full.shape[2]:
            continue
        
        # EXCLUDE PREVALENT CASES: Check if patient has ASCVD before enrollment
        prevalent = False
        for d_idx in ascvd_indices:
            if d_idx >= Y_full.shape[1]:
                continue
            # Check if patient had any ASCVD event before enrollment
            if np.any(Y_full[i, d_idx, :t_enroll] > 0):
                prevalent = True
                break
        
        if prevalent:
            n_prevalent_excluded += 1
            continue  # Skip prevalent cases
        
        # CHECK AT-RISK STATUS: Patient must be at risk at enrollment
        # For ASCVD group, patient is at risk if at least one ASCVD disease has E >= t_enroll
        # This is simple: if they're at risk at enrollment, we can make a prediction
        at_risk_at_enroll = False
        for d_idx in ascvd_indices:
            if d_idx >= E_corrected_full.shape[1]:
                continue
            # Patient is at risk at time t if E >= t
            if E_corrected_full[i, d_idx] >= t_enroll:
                at_risk_at_enroll = True
                break
        
        if not at_risk_at_enroll:
            n_not_at_risk_excluded += 1
            continue  # Skip patients not at risk at enrollment
        
        # Calculate 10-year predicted risk (static: use enrollment prediction)
        # Get ASCVD predictions at enrollment
        pi_ascvd_enroll = pi_full[i, ascvd_indices, t_enroll]
        
        # Calculate 1-year risk at enrollment
        yearly_risk = 1 - np.prod(1 - pi_ascvd_enroll)
        
        # Convert to 10-year risk (assuming constant yearly risk)
        # This is the "static" approach: 1-year score for 10-year outcome
        ten_year_risk = 1 - (1 - yearly_risk) ** 10
        predicted_10yr_risks.append(ten_year_risk)
        predicted_10yr_risks_percentiles.append(ten_year_risk)
        
        # Calculate observed 10-year outcome
        # Simple approach: check if patient had any ASCVD event in the 10-year window
        # Only count events when patient is at risk (E >= t)
        # This matches the evaluation logic: we only count events for uncensored people
        t_end = min(t_enroll + 10, Y_full.shape[2])
        had_event = False
        
        for d_idx in ascvd_indices:
            if d_idx >= Y_full.shape[1]:
                continue
            # Check events in the 10-year window
            # Only count if patient is at risk at that timepoint (E >= t)
            for t in range(t_enroll, t_end):
                if E_corrected_full[i, d_idx] >= t:  # Patient is at risk at time t
                    if Y_full[i, d_idx, t] > 0:
                        had_event = True
                        break
            if had_event:
                break
        
        observed_10yr_events.append(int(had_event))
    
    if len(predicted_10yr_risks) == 0:
        continue
    
    # Calculate statistics for this age group
    mean_predicted = np.mean(predicted_10yr_risks) * 100  # Convert to percentage
    observed_rate = np.mean(observed_10yr_events) * 100  # Convert to percentage
    
    # Calculate percentiles
    p25 = np.percentile(predicted_10yr_risks_percentiles, 25) * 100
    p50 = np.percentile(predicted_10yr_risks_percentiles, 50) * 100
    p75 = np.percentile(predicted_10yr_risks_percentiles, 75) * 100
    
    # Confidence intervals for observed rate (Wilson score)
    n = len(observed_10yr_events)
    z = 1.96
    p = observed_rate / 100
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    halfwidth = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
    obs_lower = max(0, (center - halfwidth) * 100)
    obs_upper = min(100, (center + halfwidth) * 100)
    
    age_group_data.append({
        'age_center': (age_min + min(age_max, 70)) / 2,
        'age_min': age_min,
        'age_max': age_max,
        'mean_predicted': mean_predicted,
        'observed_rate': observed_rate,
        'obs_lower': obs_lower,
        'obs_upper': obs_upper,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'n_patients': n
    })
    
    print(f"Age {age_min}-{age_max}: Predicted={mean_predicted:.2f}%, Observed={observed_rate:.2f}% (n={n}, {n_prevalent_excluded} prevalent excluded, {n_not_at_risk_excluded} not at risk excluded)")

# Convert to DataFrame
calibration_df = pd.DataFrame(age_group_data)

# ============================================================================
# CREATE CALIBRATION PLOT
# ============================================================================

print("\nCreating calibration plot...")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot mean predicted risk
ax.plot(calibration_df['age_center'], calibration_df['mean_predicted'], 
        'o-', color='#2E86AB', linewidth=2.5, markersize=10, 
        label='Model predictions (mean)', zorder=5)

# Plot percentiles
ax.plot(calibration_df['age_center'], calibration_df['p25'], 
        '--', color='#A23B72', linewidth=1.5, alpha=0.7, 
        label='Model (25th percentile)', zorder=3)
ax.plot(calibration_df['age_center'], calibration_df['p50'], 
        '--', color='#F18F01', linewidth=1.5, alpha=0.7, 
        label='Model (50th percentile)', zorder=3)

# Plot observed risk with error bars
ax.plot(calibration_df['age_center'], calibration_df['observed_rate'], 
        'o-', color='#27AE60', linewidth=2.5, markersize=10, 
        label='Prevalence-based risk', zorder=5)

# Add error bars for observed rates
for _, row in calibration_df.iterrows():
    ax.errorbar(row['age_center'], row['observed_rate'],
               yerr=[[row['observed_rate'] - row['obs_lower']], 
                     [row['obs_upper'] - row['observed_rate']]],
               color='#27AE60', alpha=0.6, capsize=5, capthick=2, zorder=4)

# Calculate R²
r2 = r2_score(calibration_df['observed_rate'], calibration_df['mean_predicted'])
print(f"R² between model predictions and observed rates: {r2:.4f}")

# Calculate calibration metric (mean absolute error)
mae = np.mean(np.abs(calibration_df['mean_predicted'] - calibration_df['observed_rate']))
print(f"Mean Absolute Error: {mae:.2f}%")

# Formatting
ax.set_xlabel('Age (yr)', fontsize=12, fontweight='bold')
ax.set_ylabel('10-year risk (%)', fontsize=12, fontweight='bold')
ax.set_title(f'10-Year Risk Calibration Curve: {disease_group}', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

# Add metrics text
metrics_text = f"{disease_group} calibration = {r2:.4f}\nR² = {r2:.4f}"
ax.text(0.02, 0.98, metrics_text,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Set limits
ax.set_xlim(calibration_df['age_center'].min() - 2, calibration_df['age_center'].max() + 2)
ax.set_ylim(0, max(calibration_df[['mean_predicted', 'observed_rate', 'obs_upper']].max()) * 1.1)

plt.tight_layout()

# Save figure as PDF
output_dir = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig5")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "10yr_calibration_curve.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n✓ Saved calibration curve to: {output_file}")

# Save data
calibration_df.to_csv(output_dir / "10yr_calibration_data.csv", index=False)
print(f"✓ Saved calibration data to: {output_dir / '10yr_calibration_data.csv'}")

plt.show()

print("\n" + "="*80)
print("CALIBRATION SUMMARY")
print("="*80)
print(calibration_df[['age_min', 'age_max', 'mean_predicted', 'observed_rate', 'n_patients']].to_string(index=False))
print("="*80)
