def plot_signature_deviations_over_time(thetas, processed_ids, matched_results, 
                                       reference_thetas_path="reference_thetas.csv",
                                       sig_indices=None, years_before_event=5, 
                                       save_plots=True):
    """
    Plot signature deviations from baseline over time for treated vs controls
    
    Parameters:
    - thetas: Signature loadings (N x K x T)
    - processed_ids: Array of all processed patient IDs
    - matched_results: Results from treatment analysis containing matched patients
    - reference_thetas_path: Path to reference_thetas.csv file
    - sig_indices: List of signature indices to plot (if None, use first 5)
    - years_before_event: How many years before event to plot
    - save_plots: Whether to save plots to files
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Load reference thetas
    try:
        reference_theta = pd.read_csv(reference_thetas_path, header=0).values
        print(f"Loaded reference_theta from {reference_thetas_path}: {reference_theta.shape}")
    except Exception as e:
        print(f"Error loading reference thetas: {e}")
        return None
    
    # Extract matched patient information
    treated_eids = matched_results['matched_patients']['treated_eids']
    control_eids = matched_results['matched_patients']['control_eids']
    treated_indices = matched_results['matched_patients']['treated_indices']
    control_indices = matched_results['matched_patients']['control_indices']
    
    # Get the actual number of time points from theta data
    n_time_points = thetas.shape[2]
    print(f"Theta shape: {thetas.shape}, using last {n_time_points} time points")
    
    # Calculate time points (years before event) - match the actual data length
    time_points = np.linspace(-years_before_event, 0, n_time_points)
    
    # Define distinct colors for each signature
    signature_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # If no specific signatures provided, automatically select top 5 by correction effect
    if sig_indices is None:
        # We'll calculate this after we have the deviation data
        auto_select = True
        sig_indices = list(range(min(5, thetas.shape[1])))  # Placeholder
    else:
        auto_select = False
    
    # Create figure with subplots
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Get ALL treated and untreated patients (not just matched ones)
    all_treated_indices = []
    all_untreated_indices = []
    
    # Find all treated patients in the dataset
    for i, eid in enumerate(treated_eids):
        global_idx = treated_indices[i]
        if global_idx < thetas.shape[0]:
            all_treated_indices.append(global_idx)
    
    # Find all untreated patients (those not in treated group)
    all_patient_indices = set(range(thetas.shape[0]))
    treated_set = set(all_treated_indices)
    all_untreated_indices = list(all_patient_indices - treated_set)
    
    print(f"Before matching: {len(all_treated_indices)} treated, {len(all_untreated_indices)} untreated")
    
    # Calculate average deviations for ALL patients in each group
    if len(all_treated_indices) > 0:
        treated_trajs_all = [thetas[idx, :, :] for idx in all_treated_indices if thetas[idx, :, :].shape[1] == n_time_points]
        treated_avg_all = np.mean(treated_trajs_all, axis=0)
        
        if reference_theta.shape[1] >= n_time_points:
            ref_slice = reference_theta[:, -n_time_points:]
        else:
            ref_slice = reference_theta
        treated_dev_all = treated_avg_all - ref_slice
    
    if len(all_untreated_indices) > 0:
        untreated_trajs_all = [thetas[idx, :, :] for idx in all_untreated_indices if thetas[idx, :, :].shape[1] == n_time_points]
        untreated_avg_all = np.mean(untreated_trajs_all, axis=0)
        
        if reference_theta.shape[1] >= n_time_points:
            ref_slice = reference_theta[:, -n_time_points:]
        else:
            ref_slice = reference_theta
        untreated_dev_all = untreated_avg_all - ref_slice
    
    # Get trajectories for MATCHED patients only
    treated_trajs_matched = []
    control_trajs_matched = []
    
    # Get trajectories for matched treated patients
    for i, eid in enumerate(treated_eids):
        global_idx = treated_indices[i]
        if global_idx < thetas.shape[0]:
            traj = thetas[global_idx, :, :]
            if traj.shape[1] == n_time_points:
                treated_trajs_matched.append(traj)
    
    # Get trajectories for matched control patients
    for i, eid in enumerate(control_eids):
        global_idx = control_indices[i]
        if global_idx < thetas.shape[0]:
            traj = thetas[global_idx, :, :]
            if traj.shape[1] == n_time_points:
                control_trajs_matched.append(traj)
    
    # Calculate average deviations for matched groups
    if len(treated_trajs_matched) > 0:
        treated_avg_matched = np.mean(treated_trajs_matched, axis=0)
        if reference_theta.shape[1] >= n_time_points:
            ref_slice = reference_theta[:, -n_time_points:]
        else:
            ref_slice = reference_theta
        treated_dev_matched = treated_avg_matched - ref_slice
        print(f"Matched treated trajectories: {len(treated_trajs_matched)} patients")
    
    if len(control_trajs_matched) > 0:
        control_avg_matched = np.mean(control_trajs_matched, axis=0)
        if reference_theta.shape[1] >= n_time_points:
            ref_slice = reference_theta[:, -n_time_points:]
        else:
            ref_slice = reference_theta
        control_dev_matched = control_avg_matched - ref_slice
        print(f"Matched control trajectories: {len(control_trajs_matched)} patients")
    
    # AUTOMATICALLY SELECT SIGNATURES BY CORRECTION EFFECT (if not specified)
    if auto_select and len(all_treated_indices) > 0 and len(treated_trajs_matched) > 0:
        # Calculate correction effect: how much the difference between treated vs control changed
        # This measures how well matching improved the balance between groups
        
        # Before matching: difference between treated and untreated
        before_diff = np.abs(treated_dev_all - untreated_dev_all)
        before_diff_mean = np.mean(before_diff, axis=1)  # Average across time points
        
        # After matching: difference between treated and control
        after_diff = np.abs(treated_dev_matched - control_dev_matched)
        after_diff_mean = np.mean(after_diff, axis=1)  # Average across time points
        
        # Correction effect: how much the difference was reduced
        # Bigger reduction = better matching for that signature
        correction_effect = before_diff_mean - after_diff_mean
        
        # Select top 5 signatures by correction effect (biggest improvement)
        top_sig_idx = np.argsort(correction_effect)[-5:][::-1]
        sig_indices = top_sig_idx.tolist()
        
        print(f"Automatically selected signatures by correction effect:")
        for i, sig in enumerate(sig_indices):
            print(f"  Sig {sig}: improvement = {correction_effect[sig]:.6f}")
    
    # 1. TOP LEFT: Before Matching - Control/Untreated Group
    print("=== BEFORE MATCHING - CONTROL GROUP ===")
    ax = axes[0, 0]
    
    if len(all_untreated_indices) > 0:
        bottom_pos = np.zeros(n_time_points)
        bottom_neg = np.zeros(n_time_points)
        
        for i, sig in enumerate(sig_indices):
            values = untreated_dev_all[sig]
            pos_values = np.maximum(values, 0)
            neg_values = np.minimum(values, 0)
            
            ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                           label=f'Sig {sig}',
                           color=signature_colors[i % len(signature_colors)])
            ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                           color=signature_colors[i % len(signature_colors)], alpha=0.5)
            
            bottom_pos += pos_values
            bottom_neg += neg_values
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    ax.set_title(f'Before Matching: Control Group\n({len(all_untreated_indices)} patients)')
    ax.set_xlabel('Years Before Event')
    ax.set_ylabel('Δ Proportion (θ)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 2. BOTTOM LEFT: Before Matching - Treated Group
    print("=== BEFORE MATCHING - TREATED GROUP ===")
    ax = axes[1, 0]
    
    if len(all_treated_indices) > 0:
        bottom_pos = np.zeros(n_time_points)
        bottom_neg = np.zeros(n_time_points)
        
        for i, sig in enumerate(sig_indices):
            values = treated_dev_all[sig]
            pos_values = np.maximum(values, 0)
            neg_values = np.minimum(values, 0)
            
            ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                           label=f'Sig {sig}',
                           color=signature_colors[i % len(signature_colors)])
            ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                           color=signature_colors[i % len(signature_colors)], alpha=0.5)
            
            bottom_pos += pos_values
            bottom_neg += neg_values
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    ax.set_title(f'Before Matching: Treated Group\n({len(all_treated_indices)} patients)')
    ax.set_xlabel('Years Before Event')
    ax.set_ylabel('Δ Proportion (θ)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 3. TOP RIGHT: After Matching - Control Group
    print("=== AFTER MATCHING - CONTROL GROUP ===")
    ax = axes[0, 1]
    
    if len(control_trajs_matched) > 0:
        bottom_pos = np.zeros(n_time_points)
        bottom_neg = np.zeros(n_time_points)
        
        for i, sig in enumerate(sig_indices):
            values = control_dev_matched[sig]
            pos_values = np.maximum(values, 0)
            neg_values = np.minimum(values, 0)
            
            ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                           label=f'Sig {sig}',
                           color=signature_colors[i % len(signature_colors)])
            ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                           color=signature_colors[i % len(signature_colors)], alpha=0.5)
            
            bottom_pos += pos_values
            bottom_neg += neg_values
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    ax.set_title(f'After Matching: Control Group\n({len(control_trajs_matched)} patients)')
    ax.set_xlabel('Years Before Event')
    ax.set_ylabel('Δ Proportion (θ)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. BOTTOM RIGHT: After Matching - Treated Group
    print("=== AFTER MATCHING - TREATED GROUP ===")
    ax = axes[1, 1]
    
    if len(treated_trajs_matched) > 0:
        bottom_pos = np.zeros(n_time_points)
        bottom_neg = np.zeros(n_time_points)
        
        for i, sig in enumerate(sig_indices):
            values = treated_dev_matched[sig]
            pos_values = np.maximum(values, 0)
            neg_values = np.minimum(values, 0)
            
            ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                           label=f'Sig {sig}',
                           color=signature_colors[i % len(signature_colors)])
            ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                           color=signature_colors[i % len(signature_colors)], alpha=0.5)
            
            bottom_pos += pos_values
            bottom_neg += neg_values
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    ax.set_title(f'After Matching: Treated Group\n({len(treated_trajs_matched)} patients)')
    ax.set_xlabel('Years Before Event')
    ax.set_ylabel('Δ Proportion (θ)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # SET SAME Y-LIMITS FOR ALL PLOTS so we can compare
    if len(all_treated_indices) > 0 and len(treated_trajs_matched) > 0:
        # Calculate overall range from all plots
        all_treated_values = treated_dev_all[sig_indices].flatten()
        all_untreated_values = untreated_dev_all[sig_indices].flatten()
        matched_treated_values = treated_dev_matched[sig_indices].flatten()
        matched_control_values = control_dev_matched[sig_indices].flatten()
        
        all_values = np.concatenate([all_treated_values, all_untreated_values, 
                                   matched_treated_values, matched_control_values])
        max_dev = np.max(np.abs(all_values))
        
        # Set same y-limits for all plots
        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(-max_dev, max_dev)
    
    # Overall title
    fig.suptitle('Signature Deviations from Baseline Over Time: Before vs After Matching', 
                 fontsize=16, y=0.95)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('signature_deviations_over_time.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'signature_deviations_over_time.png'")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Before matching:")
    print(f"  Treated patients: {len(all_treated_indices)}")
    print(f"  Untreated patients: {len(all_untreated_indices)}")
    
    print(f"\nAfter matching:")
    print(f"  Treated patients: {len(treated_trajs_matched)}")
    print(f"  Control patients: {len(control_trajs_matched)}")
    
    return fig