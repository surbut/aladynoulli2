import random
import time
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st

random.seed(42)
np.random.seed(42)

@st.cache_data
def run_digital_twin_matching(
    treated_time_idx,
    untreated_eids,
    processed_ids,
    lambdas,
    Y,
    diabetes_idx=47,
    window=10,
    window_post=10,
    sig_idx=None,
    sample_size=1000,
    max_cases=None,
    age_at_enroll=None,
    age_tolerance=2
):
    """
    Match each treated individual to a control by signature trajectory in the years prior to drug start (multivariate: all signatures),
    then compare post-treatment Type 2 diabetes event rates and signature trajectories.
    For each treated, only a random sample of controls (sample_size) is considered.
    Prints progress and estimated time remaining.
    Supports both numpy arrays and torch tensors for Y.
    If max_cases is set, only process up to max_cases treated individuals.
    If age_at_enroll is provided, only match controls within age_tolerance years.
    
    Parameters:
        treated_time_idx: dict {eid: t0} for treated patients
        untreated_eids: list of EIDs who have not started the drug (or started much later)
        processed_ids: numpy array of patient IDs (strings or ints)
        lambdas: (N, n_signatures, n_timepoints) array (already softmaxed)
        Y: event tensor (N, n_diseases, n_timepoints) (numpy or torch)
        diabetes_idx: index of Type 2 diabetes in Y (default 47)
        window: years before t0 for matching (default 10)
        window_post: years after t0 for outcome (default 10)
        sig_idx: signature index to plot (default None, meaning all signatures)
        sample_size: number of controls to sample for each treated (default 1000)
        max_cases: maximum number of treated cases to process (default None, meaning all)
        age_at_enroll: dict {eid: age} for age at enrollment
        age_tolerance: age tolerance for matching (default 2 years)
    """
    matched_pairs = []
    treated_eids = list(treated_time_idx.keys())
    n_treated = len(treated_eids)
    start_time = time.time()

    try:
        import torch
        is_torch = isinstance(Y, torch.Tensor)
    except ImportError:
        is_torch = False

    if max_cases is not None:
        treated_eids = treated_eids[:max_cases]
        n_treated = len(treated_eids)

    for i, treated_eid in enumerate(treated_eids):
        t0 = treated_time_idx[treated_eid]
        try:
            treated_idx = np.where(processed_ids == int(treated_eid))[0][0]
        except Exception:
            continue
        if t0 < window:
            continue
        # Get the treated's signature trajectory in the pre-t0 window (all signatures)
        traj_treated = lambdas[treated_idx, :, t0-window:t0].flatten()

        # Age filtering for controls
        if age_at_enroll is not None:
            treated_age = age_at_enroll.get(int(treated_eid), None)
            eligible_controls = [eid for eid in untreated_eids if abs(age_at_enroll.get(int(eid), -999) - treated_age) <= age_tolerance]
        else:
            eligible_controls = untreated_eids

        # Randomly sample controls
        if len(eligible_controls) > sample_size:
            sampled_controls = random.sample(list(eligible_controls), sample_size)
        else:
            sampled_controls = eligible_controls

        control_trajs = []
        control_indices = []
        for eid in sampled_controls:
            try:
                idx = np.where(processed_ids == int(eid))[0][0]
            except Exception:
                continue
            if t0 < window:
                continue
            traj_control = lambdas[idx, :, t0-window:t0].flatten()
            control_trajs.append(traj_control)
            control_indices.append(idx)
        if not control_trajs:
            continue
        control_trajs = np.array(control_trajs)
        dists = np.linalg.norm(control_trajs - traj_treated, axis=1)
        best_match_idx = np.argmin(dists)
        matched_pairs.append((treated_idx, control_indices[best_match_idx], t0))

        if (i+1) % 500 == 0 or (i+1) == n_treated:
            elapsed = time.time() - start_time
            avg_per = elapsed / (i+1)
            remaining = avg_per * (n_treated - (i+1))
            print(f"Processed {i+1}/{n_treated} treated. Elapsed: {elapsed/60:.1f} min. Est. remaining: {remaining/60:.1f} min.")

    total_time = time.time() - start_time
    print(f"\nMatching complete. Total elapsed time: {total_time/60:.1f} min ({total_time:.1f} sec)")

    # For each matched pair, compare post-t0 signature and event rates
    trajectories_treated = []
    trajectories_control = []
    diabetes_events_treated = []
    diabetes_events_control = []

    for treated_idx, control_idx, t0 in matched_pairs:
        t_end = min(lambdas.shape[2], t0 + window_post)
        traj_treated = lambdas[treated_idx, :, t0:t_end]
        traj_control = lambdas[control_idx, :, t0:t_end]
        if traj_treated.shape[1] == window_post and traj_control.shape[1] == window_post:
            trajectories_treated.append(traj_treated)
            trajectories_control.append(traj_control)
            if is_torch:
                diabetes_event_treated = (Y[treated_idx, diabetes_idx, t0:t_end] > 0).any().item()
                diabetes_event_control = (Y[control_idx, diabetes_idx, t0:t_end] > 0).any().item()
            else:
                diabetes_event_treated = np.any(Y[treated_idx, diabetes_idx, t0:t_end] > 0)
                diabetes_event_control = np.any(Y[control_idx, diabetes_idx, t0:t_end] > 0)
            diabetes_events_treated.append(diabetes_event_treated)
            diabetes_events_control.append(diabetes_event_control)
        else:
            continue

    try:
        trajectories_treated = np.array(trajectories_treated)
        trajectories_control = np.array(trajectories_control)
    except Exception as e:
        print("Warning: Could not convert trajectories to numpy arrays due to inhomogeneous shapes.")
        print(f"Error: {e}")
    diabetes_events_treated = np.array(diabetes_events_treated)
    diabetes_events_control = np.array(diabetes_events_control)

    treated_event_rate = diabetes_events_treated.mean() if len(diabetes_events_treated) > 0 else float('nan')
    control_event_rate = diabetes_events_control.mean() if len(diabetes_events_control) > 0 else float('nan')
    print(f"Treated event rate: {treated_event_rate:.3f}")
    print(f"Control event rate: {control_event_rate:.3f}")

    return {
        'matched_pairs': matched_pairs,
        'trajectories_treated': trajectories_treated,
        'trajectories_control': trajectories_control,
        'diabetes_events_treated': diabetes_events_treated,
        'diabetes_events_control': diabetes_events_control,
        'treated_event_rate': treated_event_rate,
        'control_event_rate': control_event_rate
    }



def run_digital_twin_matching_single_sig(
    treated_time_idx,
    untreated_eids,
    processed_ids,
    lambdas,
    Y,
    diabetes_idx=47,
    window=10,
    window_post=10,
    sig_idx=15,
    sample_size=1000,
    max_cases=None,
    age_at_enroll=None,
    age_tolerance=2
):
    """
    Match each treated individual to a control by drug-specific signature (sig_idx) trajectory in the years prior to drug start,
    then compare post-treatment Type 2 diabetes event rates and signature trajectories.
    For each treated, only a random sample of controls (sample_size) is considered.
    Prints progress and estimated time remaining.
    Supports both numpy arrays and torch tensors for Y.
    If max_cases is set, only process up to max_cases treated individuals.
    
    Parameters:
        treated_time_idx: dict {eid: t0} for treated patients
        untreated_eids: list of EIDs who have not started the drug (or started much later)
        processed_ids: numpy array of patient IDs (strings or ints)
        lambdas: (N, n_signatures, n_timepoints) array (already softmaxed)
        Y: event tensor (N, n_diseases, n_timepoints) (numpy or torch)
        diabetes_idx: index of Type 2 diabetes in Y (default 47)
        window: years before t0 for matching (default 10)
        window_post: years after t0 for outcome (default 10)
        sig_idx: signature index to use for matching (default 15)
        sample_size: number of controls to sample for each treated (default 1000)
        max_cases: maximum number of treated cases to process (default None, meaning all)
        age_at_enroll: dict {eid: age} for age at enrollment
        age_tolerance: age tolerance for matching (default 2 years)
    """
    matched_pairs = []  # List to store matched (treated, control, t0) tuples
    treated_eids = list(treated_time_idx.keys())  # List of treated patient IDs
    n_treated = len(treated_eids)  # Number of treated patients
    start_time = time.time()  # Start timer for progress estimation

    # Detect if Y is torch tensor (for compatibility)
    try:
        import torch
        is_torch = isinstance(Y, torch.Tensor)
    except ImportError:
        is_torch = False

    # If max_cases is set, only use a subset of treated
    if max_cases is not None:
        treated_eids = treated_eids[:max_cases]
        n_treated = len(treated_eids)

    # Loop over each treated patient
    for i, treated_eid in enumerate(treated_eids):
        t0 = treated_time_idx[treated_eid]  # Drug start time for this patient
        # Find index in processed_ids
        try:
            treated_idx = np.where(processed_ids == int(treated_eid))[0][0]
        except Exception:
            if i < 5:
                print(f"[Check] Treated EID {treated_eid} not found in processed_ids.")
            continue  # Skip if not found
        if t0 < window:
            if i < 5:
                print(f"[Check] Treated EID {treated_eid} does not have enough history (t0={t0}, window={window}).")
            continue  # Not enough history
        # Get the treated's signature trajectory in the pre-t0 window
        traj_treated = lambdas[treated_idx, sig_idx, t0-window:t0]  # shape: (window,)
        if i < 5:
            print(f"[Check] Treated idx: {treated_idx}, EID: {treated_eid}, Age: {age_at_enroll.get(int(treated_eid), None)}")
            print(f"[Check] Traj_treated shape: {traj_treated.shape}")

        # Only consider controls within age_tolerance years of treated's age
        treated_age = age_at_enroll.get(int(treated_eid), None)
        if treated_age is None:
            if i < 5:
                print(f"[Check] No age info for treated EID {treated_eid}.")
            continue  # skip if no age info

        # Filter all untreated controls by age
        eligible_controls = [
            eid for eid in untreated_eids
            if abs(age_at_enroll.get(int(eid), -999) - treated_age) <= age_tolerance
        ]
        if i < 5:
            print(f"[Check] Number of eligible controls for treated EID {treated_eid}: {len(eligible_controls)}")
            if len(eligible_controls) > 0:
                print(f"[Check] Example eligible control EIDs: {eligible_controls[:5]}")

        if not eligible_controls:
            if i < 5:
                print(f"[Check] No eligible controls for treated EID {treated_eid}.")
            continue  # skip if no eligible controls

        # Sample controls from eligible controls
        if len(eligible_controls) > sample_size:
            sampled_controls = random.sample(eligible_controls, sample_size)
        else:
            sampled_controls = eligible_controls
        if i < 5:
            print(f"[Check] Number of sampled controls: {len(sampled_controls)}")
            if len(sampled_controls) > 0:
                print(f"[Check] Example sampled control EIDs: {sampled_controls[:5]}")

        control_trajs = []  # Store control trajectories
        control_indices = []  # Store control indices
        for eid in sampled_controls:
            try:
                idx = np.where(processed_ids == int(eid))[0][0]
            except Exception:
                if i < 5:
                    print(f"[Check] Sampled control EID {eid} not found in processed_ids.")
                continue  # Skip if not found
            if t0 < window:
                if i < 5:
                    print(f"[Check] Control EID {eid} does not have enough history (t0={t0}, window={window}).")
                continue  # Not enough history
            # Get control's signature trajectory in the same pre-t0 window
            traj_control = lambdas[idx, sig_idx, t0-window:t0]
            control_trajs.append(traj_control)
            control_indices.append(idx)
        if not control_trajs:
            if i < 5:
                print(f"[Check] No valid control trajectories for treated EID {treated_eid}.")
            continue  # skip if no valid controls
        control_trajs = np.array(control_trajs)
        # Compute Euclidean distance between treated and each control
        dists = np.linalg.norm(control_trajs - traj_treated, axis=1)
        if i < 5:
            print(f"[Check] Distances shape: {dists.shape}")
            print(f"[Check] Min distance: {dists.min()}, Max distance: {dists.max()}")
        # Find the closest control
        best_match_idx = np.argmin(dists)
        if i < 5:
            print(f"[Check] Best match index: {best_match_idx}, Control idx: {control_indices[best_match_idx]}, Distance: {dists[best_match_idx]}")
            print(f"[Check] Confirm min distance: {dists[best_match_idx] == dists.min()}")
        # Store the matched pair (treated_idx, control_idx, t0)
        matched_pairs.append((treated_idx, control_indices[best_match_idx], t0))

        # Print progress and estimated time remaining every 500 patients
        if (i+1) % 500 == 0 or (i+1) == n_treated:
            elapsed = time.time() - start_time
            avg_per = elapsed / (i+1)
            remaining = avg_per * (n_treated - (i+1))
            print(f"Processed {i+1}/{n_treated} treated. Elapsed: {elapsed/60:.1f} min. Est. remaining: {remaining/60:.1f} min.")

    total_time = time.time() - start_time  # Print total time taken for matching
    print(f"\nMatching complete. Total elapsed time: {total_time/60:.1f} min ({total_time:.1f} sec)")

    # Print ages of the first 5 matched pairs for verification
    if age_at_enroll is not None and len(matched_pairs) > 0:
        print("\n[Check] Ages of first 5 matched treated-control pairs:")
        for j, (treated_idx, control_idx, t0) in enumerate(matched_pairs[:5]):
            treated_eid = processed_ids[treated_idx]
            control_eid = processed_ids[control_idx]
            treated_age = age_at_enroll.get(int(treated_eid), None)
            control_age = age_at_enroll.get(int(control_eid), None)
            print(f"Pair {j+1}: Treated EID {treated_eid} (age {treated_age}) -- Control EID {control_eid} (age {control_age})")

    # For each matched pair, compare post-t0 signature and event rates
    trajectories_treated = []  # Store post-t0 signature trajectories for treated
    trajectories_control = []  # Store post-t0 signature trajectories for controls
    diabetes_events_treated = []  # Store diabetes event for treated
    diabetes_events_control = []  # Store diabetes event for controls

    for treated_idx, control_idx, t0 in matched_pairs:
        t_end = min(lambdas.shape[2], t0 + window_post)  # Define end of post-t0 window
        # Get post-t0 signature trajectories
        traj_treated = lambdas[treated_idx, sig_idx, t0:t_end]
        traj_control = lambdas[control_idx, sig_idx, t0:t_end]
        # Only keep pairs with full-length post-t0 window
        if traj_treated.shape[0] == window_post and traj_control.shape[0] == window_post:
            trajectories_treated.append(traj_treated)
            trajectories_control.append(traj_control)
            # Check if diabetes event occurred in post-t0 window
            if is_torch:
                diabetes_event_treated = (Y[treated_idx, diabetes_idx, t0:t_end] > 0).any().item()
                diabetes_event_control = (Y[control_idx, diabetes_idx, t0:t_end] > 0).any().item()
            else:
                diabetes_event_treated = np.any(Y[treated_idx, diabetes_idx, t0:t_end] > 0)
                diabetes_event_control = np.any(Y[control_idx, diabetes_idx, t0:t_end] > 0)
            diabetes_events_treated.append(diabetes_event_treated)
            diabetes_events_control.append(diabetes_event_control)
        else:
            continue  # skip pairs with short trajectories

    # Try to convert to arrays for analysis, but handle inhomogeneous shapes
    try:
        trajectories_treated = np.array(trajectories_treated)
        trajectories_control = np.array(trajectories_control)
    except Exception as e:
        print("Warning: Could not convert trajectories to numpy arrays due to inhomogeneous shapes.")
        print(f"Error: {e}")
    diabetes_events_treated = np.array(diabetes_events_treated)
    diabetes_events_control = np.array(diabetes_events_control)

    # Calculate event rates
    treated_event_rate = diabetes_events_treated.mean() if len(diabetes_events_treated) > 0 else float('nan')
    control_event_rate = diabetes_events_control.mean() if len(diabetes_events_control) > 0 else float('nan')
    print(f"Treated event rate: {treated_event_rate:.3f}")
    print(f"Control event rate: {control_event_rate:.3f}")

    return {
        'matched_pairs': matched_pairs,
        'trajectories_treated': trajectories_treated,
        'trajectories_control': trajectories_control,
        'diabetes_events_treated': diabetes_events_treated,
        'diabetes_events_control': diabetes_events_control,
        'treated_event_rate': treated_event_rate,
        'control_event_rate': control_event_rate
    }
