import numpy as np


import numpy as np

def build_features_vectorized_strict(eids, t0s, processed_ids, thetas, covariate_dicts, sig_indices=None, 
                                     is_treated=False, treatment_dates=None, Y=None, event_indices=None, 
                                     exclude_controls_with_events=False, smoking_encoding='binary_current', cad_reference=30):
    """
    Semi-vectorized build_features:
    - Vectorized covariates, missingness, CAD exclusions, and clinical filters
    - Minimal loops for per-patient Y-slice checks and signature-traj extraction (variable windows)
    - Matches original filtering semantics (all the "continue" cases)
    """
    window = 10
    n_signatures = thetas.shape[1]
    if sig_indices is None:
        sig_indices = list(range(n_signatures))
    expected_length = len(sig_indices) * window

    # Map EIDs to indices in processed_ids
    pid_index = {int(pid): i for i, pid in enumerate(processed_ids)}

    # Preselect candidates: exist in processed_ids and have ≥window history
    candidates = []
    for eid, t0 in zip(eids, t0s):
        if eid in pid_index and t0 >= window:
            idx = pid_index[eid]
            if idx < thetas.shape[0]:
                candidates.append((eid, int(t0), idx))
    if len(candidates) == 0:
        return np.array([]), [], []

    cand_eids = np.array([int(e) for e, _, _ in candidates], dtype=int)
    cand_t0s  = np.array([t for _, t, _ in candidates], dtype=int)
    cand_idx  = np.array([i for _, _, i in candidates], dtype=int)

    # Vectorized covariate pulls
    def pull(name, default):
        d = covariate_dicts.get(name, {})
        return np.array([d.get(eid, default) for eid in cand_eids], dtype=float)
    # Use age_at_enroll from covariate_dicts (already calculated in make_covariate_dicts.py)
    age_at_enroll = pull('age_at_enroll', 57.0)
    sex           = pull('sex', 0.0)
    bmi           = pull('bmi', 27.5)
    dm2_prev      = pull('dm2_prev', np.nan)
    antihtnbase   = pull('antihtnbase', np.nan)
    dm1_prev      = pull('dm1_prev', np.nan)

    dm_any        = pull('Dm_Any', np.nan)
    ht_any        = pull('Ht_Any', np.nan)
    hyperlip_any  = pull('HyperLip_Any', np.nan)

    ldl_prs       = pull('ldl_prs', np.nan)
    hdl           = pull('hdl', np.nan)
    tchol         = pull('tchol', np.nan)
    sbp           = pull('SBP', np.nan)
    pce_goff      = pull('pce_goff', np.nan)  # Don't default to 0.09 - exclude missing values
    cad_prs       = pull('cad_prs', 0.0)

    # Smoking
    smoke_raw = [covariate_dicts.get('smoke', {}).get(int(eid), None) for eid in cand_eids]
    if smoking_encoding == 'binary_current':
        smoke = np.array([1 if s == "Current" else 0 for s in smoke_raw], dtype=float)
    else:
        smoke = np.array([[1,0,0] if s == "Never" else [0,1,0] if s == "Previous" else [0,0,1] if s == "Current" else [0,0,0] for s in smoke_raw], dtype=float)

    # Impute quantitative with means (match original defaults)
    def impute_mean(arr, fallback):
        mask = np.isnan(arr)
        if np.any(~mask):
            mean_val = np.nanmean(arr)
        else:
            mean_val = fallback
        arr[mask] = mean_val
        return arr

    ldl_prs = impute_mean(ldl_prs, 0.0)
    hdl     = impute_mean(hdl, 50.0)
    tchol   = impute_mean(tchol, 200.0)
    sbp     = impute_mean(sbp, 140.0)
    # pce_goff = np.where(np.isnan(pce_goff), 0.09, pce_goff)  # Don't impute - exclude missing values
    cad_prs  = np.where(np.isnan(cad_prs), 0.0, cad_prs)

    # Start with all candidates valid; knock out via mask like "continue"
    keep = np.ones(len(candidates), dtype=bool)

    # NEW: Exclude if missing PCE-Goff (key addition for statin trials)
    keep &= ~np.isnan(pce_goff)
    print(f"After PCE-Goff exclusion: {np.sum(keep)}/{len(candidates)} patients remaining")

    # Exclude if missing REQUIRED binary vars: dm2_prev, antihtnbase, dm1_prev
    keep &= ~np.isnan(dm2_prev)
    keep &= ~np.isnan(antihtnbase)
    keep &= ~np.isnan(dm1_prev)

    # Exclude if Dm_Any / Ht_Any / HyperLip_Any unknown
    keep &= ~np.isnan(dm_any)
    keep &= ~np.isnan(ht_any)
    keep &= ~np.isnan(hyperlip_any)

    # CAD exclusions:
    cad_any_arr      = np.array([covariate_dicts.get('Cad_Any', {}).get(int(eid), 0) for eid in cand_eids], dtype=float)
    cad_censor_age   = np.array([covariate_dicts.get('Cad_censor_age', {}).get(int(eid), np.nan) for eid in cand_eids], dtype=float)

    if is_treated and treatment_dates is not None:
        # Calculate treatment age using date-based approach (same as ObservationalTreatmentPatternLearner)
        # Map each eid to its treatment time (aligned to input lists)
        eid_to_ttime = {int(e): int(t) for e, t in zip(eids, treatment_dates) if int(e) in cand_eids}
        ttimes = np.array([eid_to_ttime.get(int(eid), np.nan) for eid in cand_eids], dtype=float)
        
        # Calculate treatment age: Cad_reference+ treatment_time (in years)
        # treatment_time is already in years since age 30, so add 30 to get actual age
        treatment_age = (30 + ttimes)
        reference_age = treatment_age
        
        # Missing ttimes => drop
        keep &= ~np.isnan(reference_age)
        # cad_any == 2 and cad_censor_age <= reference_age => exclude
        treat_mask = (cad_any_arr == 2) & ~np.isnan(cad_censor_age) & ~np.isnan(reference_age)
        keep &= ~(treat_mask & (cad_censor_age <= reference_age))
        # Age to use for matching (treatment age) else fall back
        age_for_match = np.where(~np.isnan(reference_age), reference_age, age_at_enroll)
    else:
        # controls: exclude if cad_censor_age <= age_at_enroll
        ctrl_mask = (cad_any_arr == 2) & ~np.isnan(cad_censor_age) & ~np.isnan(age_at_enroll)
        keep &= ~(ctrl_mask & (cad_censor_age <= age_at_enroll))
        age_for_match = age_at_enroll

    # Prepare Y-based event exclusions (loop; fast enough)
    if Y is not None and event_indices is not None:
        for i in range(len(candidates)):
            if not keep[i]:
                continue
            idx = cand_idx[i]
            t0  = cand_t0s[i]
            if is_treated and treatment_dates is not None:
                # Find patient-specific treatment time
                eid_i = cand_eids[i]
                # Use same mapping as above
                ttime = eid_to_ttime.get(int(eid_i), None)
                if ttime is None:
                    keep[i] = False
                    continue
                ttime = int(ttime)
                # pre-treatment events
                pre = Y[idx, event_indices, :ttime]
                if hasattr(pre, 'detach'):
                    pre = pre.detach().cpu().numpy()
                if np.any(pre > 0):
                    keep[i] = False
                    continue
                # events within 1 year post-treatment
                post1 = Y[idx, event_indices, ttime:min(ttime+1, Y.shape[2])]
                if hasattr(post1, 'detach'):
                    post1 = post1.detach().cpu().numpy()
                if np.any(post1 > 0):
                    keep[i] = False
                    continue
            else:
                # controls: optional pre-enrollment exclusion
                if exclude_controls_with_events:
                    pre = Y[idx, event_indices, :t0]
                    if hasattr(pre, 'detach'):
                        pre = pre.detach().cpu().numpy()
                    if np.any(pre > 0):
                        keep[i] = False
                        continue
                # events within 1 year post-enrollment
                post1 = Y[idx, event_indices, t0:min(t0+1, Y.shape[2])]
                if hasattr(post1, 'detach'):
                    post1 = post1.detach().cpu().numpy()
                if np.any(post1 > 0):
                    keep[i] = False
                    continue

    # Apply mask
    if not np.any(keep):
        return np.array([]), [], []

    kept_eids_arr = cand_eids[keep]
    kept_idx_arr  = cand_idx[keep]
    kept_t0s_arr  = cand_t0s[keep]

    # Build signatures (loop due to variable t0 windows)
    sig_rows = []
    for idx, t0 in zip(kept_idx_arr, kept_t0s_arr):
        seg = thetas[idx, sig_indices, t0-window:t0].flatten()
        # enforce same checks as original:
        if seg.shape[0] != expected_length or np.any(np.isnan(seg)):
            sig_rows.append(None)
        else:
            sig_rows.append(seg)

    sig_valid = np.array([r is not None for r in sig_rows], dtype=bool)
    kept_eids_arr = kept_eids_arr[sig_valid]
    kept_idx_arr  = kept_idx_arr[sig_valid]
    kept_t0s_arr  = kept_t0s_arr[sig_valid]
    sig_trajs = np.vstack([r for r in sig_rows if r is not None]) if np.any(sig_valid) else np.empty((0, expected_length))

    if sig_trajs.shape[0] == 0:
        return np.array([]), [], []

    # Prepare clinical features (same as your final vector)
    # Impute remaining simple numerics
    age_for_match = np.where(np.isnan(age_for_match), 57.0, age_for_match)
    sex = np.where(np.isnan(sex), 0.0, sex)
    bmi = np.where(np.isnan(bmi), 27.5, bmi)
    dm_any = np.where(np.isnan(dm_any), 1.0, dm_any)
    dm2_prev = np.where(np.isnan(dm2_prev), 0.0, dm2_prev)
    ht_any = np.where(np.isnan(ht_any), 1.0, ht_any)
    hyperlip_any = np.where(np.isnan(hyperlip_any), 1.0, hyperlip_any)
    antihtnbase = np.where(np.isnan(antihtnbase), 0.0, antihtnbase)

    # Align covariates to kept rows
    row_mask = keep.copy()
    # Build aligned arrays then subselect sig_valid rows
    def align(arr):
        return np.array(arr)[row_mask][sig_valid]

    age_feat   = align(age_for_match)
    sex_feat   = align(sex)
    bmi_feat   = align(bmi)
    dm_any_f   = align(dm_any)
    dm2_f      = align(dm2_prev)
    ht_any_f   = align(ht_any)
    hyperlip_f = align(hyperlip_any)
    antihtn_f  = align(antihtnbase)
    cad_prs_f  = align(cad_prs)
    pce_goff_f = align(pce_goff)

    if smoking_encoding == 'binary_current':
        smoke_f = align(smoke)
        clinical = np.column_stack([age_feat, sex_feat, bmi_feat, dm_any_f, dm2_f, ht_any_f, hyperlip_f, antihtn_f, cad_prs_f, pce_goff_f, smoke_f])
    else:
        smoke_keep = np.array(smoke)[row_mask][sig_valid]
        clinical = np.column_stack([age_feat, sex_feat, bmi_feat, dm_any_f, dm2_f, ht_any_f, hyperlip_f, antihtn_f, cad_prs_f, pce_goff_f,
                                    smoke_keep[:,0], smoke_keep[:,1], smoke_keep[:,2]])

    features = np.column_stack([sig_trajs, clinical])
    #features = np.column_stack([clinical])

    kept_eids_list = [int(x) for x in kept_eids_arr.tolist()]
    kept_idx_list  = kept_idx_arr.tolist()
    return features, kept_idx_list, kept_eids_list