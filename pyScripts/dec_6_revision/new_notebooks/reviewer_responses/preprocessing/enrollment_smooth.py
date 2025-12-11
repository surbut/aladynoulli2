
censor_df = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/censor_info.csv')
T = 52

# Convert to timepoints
max_timepoints = torch.tensor(
    (censor_df['max_censor'].values - 30).clip(0, T-1).astype(int)
)

# Only update censored cases (where E == T-1)
censored_mask = (E == T - 1)  # Shape: (N, D)

# For each patient, cap censored diseases to their max_timepoint
# Expand max_timepoints to match E shape
max_timepoints_expanded = max_timepoints.unsqueeze(1).expand_as(E)

# Update only censored positions
E_corrected = torch.where(
    censored_mask,
    torch.minimum(E, max_timepoints_expanded),
    E  # Keep event times as-is
)


# Compute new prevalence with at-risk filtering using corrected E
print("="*80)
print("COMPUTING NEW PREVALENCE WITH AT-RISK FILTERING")
print("="*80)

def compute_smoothed_prevalence_at_risk(Y, E_corrected, window_size=5, smooth_on_logit=True):
    """
    Compute smoothed prevalence with proper at-risk filtering.
    
    Parameters:
    -----------
    Y : torch.Tensor (N × D × T)
    E_corrected : torch.Tensor (N × D) - corrected event/censor times
    window_size : int - Gaussian smoothing window size
    smooth_on_logit : bool - Smooth on logit scale
    """
    if torch.is_tensor(Y):
        Y = Y.numpy()
    if torch.is_tensor(E_corrected):
        E_corrected = E_corrected.numpy()
    
    N, D, T = Y.shape
    prevalence_t = np.zeros((D, T))
    
    print(f"Computing prevalence for {D} diseases, {T} timepoints...")
    
    # Convert E_corrected to numpy if needed
    if torch.is_tensor(E_corrected):
        E_corrected_np = E_corrected.numpy()
    else:
        E_corrected_np = E_corrected
    
    for d in range(D):
        if d % 50 == 0:
            print(f"  Processing disease {d}/{D}...")
        
        for t in range(T):
            # Only include people who are still at risk at timepoint t
            at_risk_mask = (E_corrected_np[:, d] >= t) 
            
            # Alternative: Include people enrolled at exactly age_t only if they have minimum follow-up
            # For now, let's try excluding them entirely to see if that fixes the U-shape
            
            if at_risk_mask.sum() > 0:
                if torch.is_tensor(Y):
                    prevalence_t[d, t] = Y[at_risk_mask, d, t].numpy().mean()
                else:
                    prevalence_t[d, t] = Y[at_risk_mask, d, t].mean()
            else:
                prevalence_t[d, t] = np.nan
        
        # Smooth as before
        if smooth_on_logit:
            epsilon = 1e-8
            # Handle NaN values
            valid_mask = ~np.isnan(prevalence_t[d, :])
            if valid_mask.sum() > 0:
                logit_prev = np.full(T, np.nan)
                logit_prev[valid_mask] = np.log(
                    (prevalence_t[d, valid_mask] + epsilon) / 
                    (1 - prevalence_t[d, valid_mask] + epsilon)
                )
                # Smooth only valid values
                smoothed_logit = gaussian_filter1d(
                    np.nan_to_num(logit_prev, nan=0), 
                    sigma=window_size
                )
                # Restore NaN where original was NaN
                smoothed_logit[~valid_mask] = np.nan
                prevalence_t[d, :] = 1 / (1 + np.exp(-smoothed_logit))
        else:
            prevalence_t[d, :] = gaussian_filter1d(
                np.nan_to_num(prevalence_t[d, :], nan=0), 
                sigma=window_size
            )
    
    return prevalence_t

# Compute new prevalence
print("\nComputing new prevalence with at-risk filtering...")
new_prevalence_t = compute_smoothed_prevalence_at_risk(
    Y=Y, 
    E_corrected=E_corrected, 
    window_size=5,
    smooth_on_logit=True
)

print(f"\nNew prevalence shape: {new_prevalence_t.shape}")
print("Done!")



# Left: Old E
# Compare for disease 112
d_idx = 47
ages = np.arange(30, 30 + T)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(ages, oldp[d_idx, :], 'b-', linewidth=2, label='Original (no filtering)', alpha=0.7)
axes[0].plot(ages, test_prev_old_E[d_idx, :], 'r--', linewidth=2, label='At-risk with OLD E', alpha=0.7)
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Prevalence', fontsize=12)
axes[0].set_title('At-Risk Filtering with OLD E\n(Should show U-shape?)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: New E
axes[1].plot(ages, oldp[d_idx, :], 'b-', linewidth=2, label='Original (no filtering)', alpha=0.7)
axes[1].plot(ages, new_prevalence_t[d_idx, :], 'g--', linewidth=2, label='At-risk with NEW E', alpha=0.7)
axes[1].set_xlabel('Age', fontsize=12)
axes[1].set_ylabel('Prevalence', fontsize=12)
axes[1].set_title('At-Risk Filtering with NEW E\n(Should be similar?)', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("If BOTH show U-shape, the problem is with the at-risk filtering logic,")
print("not with the E correction. The filtering logic needs to be fixed.")
