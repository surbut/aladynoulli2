import random
import time
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

random.seed(42)
np.random.seed(42)

disease_names_df = pd.read_csv("disease_names.csv")
disease_names_list = disease_names_df.iloc[:, 1].tolist()
disease_names = disease_names_list

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
    sample_size=1000,  # Ignored in new approach
    max_cases=None,
    age_at_enroll=None,
    age_tolerance=2,
    eid_to_sex=None,
    eid_to_pgs=None,
    pgs_weight=0.5,
    eid_to_dm2_prev=None,
    eid_to_antihtnbase=None,
    eid_to_dm1_prev=None,
    eid_to_ldl_prs=None,
    eid_to_cad_prs=None
):
    """
    Match each treated individual to a control by signature trajectory in the years prior to drug start (multivariate: all signatures),
    then compare post-treatment Type 2 diabetes event rates and signature trajectories.
    Uses NearestNeighbors for efficient matching over all eligible controls.
    """
    from sklearn.neighbors import NearestNeighbors
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

    # Prepare treated trajectories and indices
    treated_trajs = []
    valid_treated_indices = []
    treated_t0s = []
    for eid in treated_eids:
        t0 = treated_time_idx[eid]
        try:
            treated_idx = np.where(processed_ids == int(eid))[0][0]
        except Exception:
            continue
        if t0 < window:
            continue
        t0_int = int(t0)
        traj_treated = lambdas[treated_idx, :, t0_int-window:t0_int].flatten()
        treated_trajs.append(traj_treated)
        valid_treated_indices.append(treated_idx)
        treated_t0s.append(t0)

    # For each treated, determine eligible controls (by age/sex if needed)
    eligible_controls_list = []
    for i, eid in enumerate(treated_eids):
        t0 = treated_time_idx[eid]
        # Get treated's covariate values
        dm2_val = eid_to_dm2_prev.get(int(eid)) if eid_to_dm2_prev else None
        antihtn_val = eid_to_antihtnbase.get(int(eid)) if eid_to_antihtnbase else None
        dm1_val = eid_to_dm1_prev.get(int(eid)) if eid_to_dm1_prev else None
        ldl_prs = eid_to_ldl_prs.get(int(eid), 0) if eid_to_ldl_prs else 0
        cad_prs = eid_to_cad_prs.get(int(eid), 0) if eid_to_cad_prs else 0

        if age_at_enroll is not None and eid_to_sex is not None:
            treated_age = age_at_enroll.get(int(eid), None)
            treated_sex = eid_to_sex.get(int(eid), None)
            eligible_controls = [
                ceid for ceid in untreated_eids
                if (abs(age_at_enroll.get(int(ceid), -999) - treated_age) <= age_tolerance)
                and (eid_to_sex.get(int(ceid), None) == treated_sex)
            ]
        elif age_at_enroll is not None:
            treated_age = age_at_enroll.get(int(eid), None)
            eligible_controls = [
                ceid for ceid in untreated_eids
                if abs(age_at_enroll.get(int(ceid), -999) - treated_age) <= age_tolerance
            ]
        else:
            eligible_controls = untreated_eids

        # Now filter for exact match on comorbidities
        if eid_to_dm2_prev:
            eligible_controls = [ceid for ceid in eligible_controls if eid_to_dm2_prev.get(int(ceid)) == dm2_val]
        if eid_to_antihtnbase:
            eligible_controls = [ceid for ceid in eligible_controls if eid_to_antihtnbase.get(int(ceid)) == antihtn_val]
        if eid_to_dm1_prev:
            eligible_controls = [ceid for ceid in eligible_controls if eid_to_dm1_prev.get(int(ceid)) == dm1_val]

        eligible_controls_list.append(eligible_controls)

    # For each treated, match to closest eligible control using NearestNeighbors
    for i, (treated_idx, t0, eligible_controls) in enumerate(zip(valid_treated_indices, treated_t0s, eligible_controls_list)):
        if len(eligible_controls) == 0:
            continue
        # Build control trajectories for eligible controls
        control_indices = []
        control_trajs = []
        control_pgs_values = []
        for ceid in eligible_controls:
            try:
                cidx = np.where(processed_ids == int(ceid))[0][0]
            except Exception:
                continue
            if t0 < window:
                continue
            t0_int = int(t0)
            traj_control = lambdas[cidx, :, t0_int-window:t0_int].flatten()
            control_trajs.append(traj_control)
            control_indices.append(cidx)
            if eid_to_pgs:
                pgs_control = eid_to_pgs.get(int(ceid))
                control_pgs_values.append(pgs_control)
        if not control_trajs:
            continue
        control_trajs = np.array(control_trajs)
        t0_int = int(t0)
        traj_treated = lambdas[treated_idx, :, t0_int-window:t0_int].flatten().reshape(1, -1)
        # If PGS is used, combine trajectory and PGS distance
        if eid_to_pgs and pgs_weight > 0:
            pgs_treated = eid_to_pgs.get(int(processed_ids[treated_idx]))
            control_pgs_values = np.array(control_pgs_values)
            # Normalize trajectory and PGS distances
            nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(control_trajs)
            traj_dist, nn_idx = nn.kneighbors(traj_treated)
            traj_dist = traj_dist.flatten()
            # PGS distances
            pgs_dists = np.abs(control_pgs_values - pgs_treated)
            # Normalize
            std_traj = np.std(traj_dist)
            std_pgs = np.std(pgs_dists)
            norm_traj = traj_dist / std_traj if std_traj > 0 else traj_dist
            norm_pgs = pgs_dists / std_pgs if std_pgs > 0 else pgs_dists
            dists = (1 - pgs_weight) * norm_traj + pgs_weight * norm_pgs
            best_match_idx = np.argmin(dists)
        else:
            # Only trajectory
            nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(control_trajs)
            _, nn_idx = nn.kneighbors(traj_treated)
            best_match_idx = nn_idx[0][0]
        matched_pairs.append((treated_idx, control_indices[best_match_idx], t0))
        if (i+1) % 500 == 0 or (i+1) == len(valid_treated_indices):
            elapsed = time.time() - start_time
            avg_per = elapsed / (i+1)
            remaining = avg_per * (len(valid_treated_indices) - (i+1))
            print(f"Processed {i+1}/{len(valid_treated_indices)} treated. Elapsed: {elapsed/60:.1f} min. Est. remaining: {remaining/60:.1f} min.")

    total_time = time.time() - start_time
    print(f"\nMatching complete. Total elapsed time: {total_time/60:.1f} min ({total_time:.1f} sec)")

    # For each matched pair, compare post-t0 signature and event rates
    trajectories_treated = []
    trajectories_control = []
    diabetes_events_treated = []
    diabetes_events_control = []

    for treated_idx, control_idx, t0 in matched_pairs:
        t_end = min(lambdas.shape[2], t0 + window_post)
        t0_int = int(t0)
        traj_treated = lambdas[treated_idx, :, t0_int:t_end]
        traj_control = lambdas[control_idx, :, t0_int:t_end]
        if traj_treated.shape[1] == window_post and traj_control.shape[1] == window_post:
            trajectories_treated.append(traj_treated)
            trajectories_control.append(traj_control)
            if is_torch:
                diabetes_event_treated = (Y[treated_idx, diabetes_idx, t0_int:t_end] > 0).any().item()
                diabetes_event_control = (Y[control_idx, diabetes_idx, t0_int:t_end] > 0).any().item()
            else:
                diabetes_event_treated = np.any(Y[treated_idx, diabetes_idx, t0_int:t_end] > 0)
                diabetes_event_control = np.any(Y[control_idx, diabetes_idx, t0_int:t_end] > 0)
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



from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

def encode_smoking(status):
    # One-hot encoding: [Never, Previous, Current]
    if status == "Never":
        return [1, 0, 0]
    elif status == "Previous":
        return [0, 1, 0]
    elif status == "Current":
        return [0, 0, 1]
    else:
        return [0, 0, 0]  # For missing/unknown
    
    # Assuming your DataFrame is called ukb_pheno
# And the relevant columns are named as in your screenshot
def prev_condition(df, any_col, censor_age_col, enroll_age_col, new_col):
    df[new_col] = ((df[any_col] == 2) & (df[censor_age_col] < df[enroll_age_col])).astype(int)

def get_time_index(yob, presc_date, time_grid):
    if pd.isnull(yob) or pd.isnull(presc_date):
        return None
    if isinstance(presc_date, str):
        presc_date = pd.to_datetime(presc_date, errors='coerce')
    if pd.isnull(presc_date):
        return None
    age_at_presc = presc_date.year - yob
    return int(np.argmin(np.abs(time_grid + 30 - age_at_presc)))

def build_features(eids, t0s, processed_ids, thetas, covariate_dicts, sig_indices=None):
    features = []
    indices = []
    kept_eids = []
    window = 10
    n_signatures = thetas.shape[1]
    if sig_indices is None:
        sig_indices = list(range(n_signatures))
    expected_length = len(sig_indices) * window
    for eid, t0 in zip(eids, t0s):
        try:
            idx = np.where(processed_ids == int(eid))[0][0]
        except Exception:
            continue
        if t0 < window:
            continue  # Not enough history
        t0_int = int(t0)
        sig_traj = thetas[idx, sig_indices, t0_int-window:t0_int].flatten()
        if sig_traj.shape[0] != expected_length:
            continue  # Skip if not the right length
        age = covariate_dicts['age_at_enroll'].get(int(eid), 57)
        sex = int(covariate_dicts['sex'].get(int(eid), 0))
        dm2 = covariate_dicts['dm2_prev'].get(int(eid), 0)
        antihtn = covariate_dicts['antihtnbase'].get(int(eid), 0)
        dm1 = covariate_dicts['dm1_prev'].get(int(eid), 0)
        smoke = encode_smoking(covariate_dicts['smoke'].get(int(eid), None))
        ldl_prs = covariate_dicts.get('ldl_prs', {}).get(int(eid), 0)
        cad_prs = covariate_dicts.get('cad_prs', {}).get(int(eid), 0)
        tchol = covariate_dicts.get('tchol', {}).get(int(eid),0)
        hdl = covariate_dicts.get('hdl', {}).get(int(eid),0 )
        sbp = covariate_dicts.get('SBP', {}).get(int(eid), 0)
        pce_goff = covariate_dicts.get('pce_goff', {}).get(int(eid), 0.09)
        features.append(np.concatenate([
            sig_traj, [age, sex, dm2, antihtn, dm1, ldl_prs, cad_prs, tchol, hdl, sbp, pce_goff] + smoke
        ]))
        indices.append(idx)
        kept_eids.append(eid)
    return np.array(features), indices, kept_eids

def build_features_no_sigs(eids, t0s, processed_ids, covariate_dicts):
    features = []
    indices = []
    kept_eids = []
    window = 10  # Still needed for consistency, but not used for sigs
    for eid, t0 in zip(eids, t0s):
        try:
            idx = np.where(processed_ids == int(eid))[0][0]
        except Exception:
            continue
        if t0 < window:
            continue  # Not enough history
        age = covariate_dicts['age_at_enroll'].get(int(eid), 57)
        sex = int(covariate_dicts['sex'].get(int(eid), 0))
        dm2 = covariate_dicts['dm2_prev'].get(int(eid), 0)
        antihtn = covariate_dicts['antihtnbase'].get(int(eid), 0)
        dm1 = covariate_dicts['dm1_prev'].get(int(eid), 0)
        smoke = encode_smoking(covariate_dicts['smoke'].get(int(eid), None))
        ldl_prs = covariate_dicts.get('ldl_prs', {}).get(int(eid), 0)
        cad_prs = covariate_dicts.get('cad_prs', {}).get(int(eid), 0)
        tchol = covariate_dicts.get('tchol', {}).get(int(eid), 0)
        hdl = covariate_dicts.get('hdl', {}).get(int(eid), 0)
        sbp = covariate_dicts.get('SBP', {}).get(int(eid), 0)
        pce_goff = covariate_dicts.get('pce_goff', {}).get(int(eid), 0.09)
        features.append(np.array([
            age, sex, dm2, antihtn, dm1, ldl_prs, cad_prs, tchol, hdl, sbp, pce_goff, *smoke
        ]))
        indices.append(idx)
        kept_eids.append(eid)
    return np.array(features), indices, kept_eids

