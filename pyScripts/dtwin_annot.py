# ---------------------------------------------
# 1. Import necessary libraries
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# %load_ext autoreload  # <-- REMOVE: Jupyter magic, not valid in .py scripts
# %autoreload 2         # <-- REMOVE: Jupyter magic, not valid in .py scripts
from dt import *
import torch
import pandas as pd
# Set seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)

# ---------------------------------------------
# 2. Load data: PRS, patient IDs, PRS/disease names, G-matrix, covariates
# ---------------------------------------------
# Load polygenic risk scores (thetas) and processed patient IDs
# These are the main subject IDs and their PRS values
# thetas: (n_individuals, n_prs)
thetas = np.load("thetas.npy")
processed_ids = np.load("processed_patient_ids.npy").astype(int)

# Load PRS names and labels for plotting/interpretation
prs_names = pd.read_csv('prs_names.csv')
prs_labels = prs_names['Names'].tolist()

# Load disease names for reference
# (Assumes second column contains names)
disease_names_df = pd.read_csv("disease_names.csv")
disease_names = disease_names_df.iloc[:, 1].tolist()

# Load G-matrix (genotype/PRS matrix)
G = torch.load("/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/G_matrix.pt")
G = G.detach().cpu().numpy()

# Load covariate data (demographics, labs, etc.)
cov = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/matched_pce_df_400k.csv')
cov.columns = cov.columns.str.strip()
cov = cov.rename(columns={cov.columns[0]: 'eid'})
cov['eid'] = cov['eid'].astype(int)
# Parse enrollment date and calculate age at enrollment
cov['enrollment'] = pd.to_datetime(cov['Enrollment_Date'], errors='coerce')
cov['age_at_enroll'] = cov['enrollment'].dt.year - cov['birth_year']
age_at_enroll = dict(zip(cov['eid'], cov['age_at_enroll']))
eid_to_yob = dict(zip(cov['eid'], cov['birth_year']))

# ---------------------------------------------
# 3. Load and process prescription data
# ---------------------------------------------
prescription_path = 'prescriptions.csv'
df_treat = pd.read_csv(prescription_path)
df_treat['eid'] = df_treat['eid'].astype(int)
df_treat = df_treat.merge(cov[['eid', 'birth_year']], on='eid', how='left')
df_treat['from'] = pd.to_datetime(df_treat['from'], errors='coerce')

# ---------------------------------------------
# 4. Check overlap between covariate and processed IDs
# ---------------------------------------------
# Convert processed_ids to pandas Series for convenience
processed_ids = pd.Series(processed_ids)

cov_eids = set(cov['eid'])
processed_eids = set(processed_ids)

print("EIDs in cov but not in processed_ids:", cov_eids - processed_eids)
print("EIDs in processed_ids but not in cov:", processed_eids - cov_eids)
print("Number of EIDs in cov:", len(cov_eids))
print("Number of EIDs in processed_ids:", len(processed_eids))
print("Number of overlapping EIDs:", len(cov_eids & processed_eids))

# ---------------------------------------------
# 5. Filter prescription data for drug category (e.g., statins)
# ---------------------------------------------
drug_category = "statins"  # or "antidiabetic", etc.
df_drug = df_treat if drug_category == "All" else df_treat[df_treat['category'] == drug_category]
num_unique_eids = df_drug['eid'].nunique()
print(f"Number of unique individuals in {drug_category}: {num_unique_eids}")
# EIDs in df_treat and cov
treat_eids = set(df_drug['eid'])
overlap_treat_cov = treat_eids & cov_eids
print("Number of people in df_drug:", len(treat_eids))
print("Number of people in both df_drug and cov:", len(overlap_treat_cov))

# ---------------------------------------------
# 6. Identify incident treated individuals (first prescription after enrollment)
# ---------------------------------------------
# Find first prescription for each person
first_presc = df_drug.groupby('eid')['from'].min().reset_index()
first_presc = first_presc.merge(
    cov[['eid', 'Birthdate','Enrollment_Date']],
    left_on='eid', right_on='eid', how='left'
)
first_presc['Birthdate'] = pd.to_datetime(first_presc['Birthdate'])
first_presc['from'] = pd.to_datetime(first_presc['from'])
# Calculate age at first prescription
first_presc['age_at_first_script'] = (first_presc['from'] - first_presc['Birthdate']).dt.days / 365.25
# Only keep those whose first prescription is after enrollment (incident users)
incident_treated = first_presc[first_presc['from'] > first_presc['Enrollment_Date']].copy()
incident_treated['age_at_first_script'] = (incident_treated['from'] - incident_treated['Birthdate']).dt.days/365.25
incident_treated['years_since_30'] = (incident_treated['age_at_first_script'] - 30).round()
incident_treated.shape

# ---------------------------------------------
# 7. Add prior disease/condition flags to covariate table
# ---------------------------------------------
# These functions flag prior disease status at enrollment for each subject
prev_condition(cov, 'Dm_Any', 'Dm_censor_age', 'age_enrolled', 'prev_dm')
prev_condition(cov, 'DmT1_Any', 'DmT1_censor_age', 'age_enrolled', 'prev_dm1')
prev_condition(cov, 'Ht_Any', 'Ht_censor_age', 'age_enrolled', 'prev_ht')
prev_condition(cov, 'HyperLip_Any', 'HyperLip_censor_age', 'age_enrolled', 'prev_hl')

# ---------------------------------------------
# 8. Build mapping dictionaries for covariates and PRS
# ---------------------------------------------
ldl_idx = prs_labels.index('LDL_SF')   # LDL PRS index
cad_idx = prs_labels.index('CAD')      # CAD PRS index
eid_to_dm2_prev = dict(zip(cov['eid'], cov['prev_dm']))
eid_to_antihtnbase = dict(zip(cov['eid'], cov['prev_ht']))
eid_to_htn = dict(zip(cov['eid'], cov['prev_ht']))
eid_to_smoke = dict(zip(cov['eid'], cov['SmokingStatusv2']))
eid_to_dm1_prev = dict(zip(cov['eid'], cov['prev_dm1']))
eid_to_hl_prev = dict(zip(cov['eid'], cov['prev_hl']))
eid_to_sex = dict(zip(cov['eid'],cov['Sex']))
eid_to_age = dict(zip(cov['eid'],cov['age_enrolled']))
eid_to_ldl_prs = {eid: G[i, ldl_idx] for i, eid in enumerate(processed_ids)}
eid_to_cad_prs = {eid: G[i, cad_idx] for i, eid in enumerate(processed_ids)}
eid_to_race = dict(zip(cov['eid'],cov['race']))
eid_to_pce_goff = dict(zip(cov['eid'],cov['pce_goff']))
eid_to_tchol = dict(zip(cov['eid'],cov['tchol']))
eid_to_hdl = dict(zip(cov['eid'],cov['hdl']))
eid_to_sbp = dict(zip(cov['eid'],cov['SBP']))

# ---------------------------------------------
# 9. Define treated and control groups (EIDs and time zero)
# ---------------------------------------------
# Treated: incident users (first prescription after enrollment)
treated_eids = incident_treated['eid']
treated_t0s = incident_treated['years_since_30']
treated_t0_dict = dict(zip(treated_eids, treated_t0s))
len(treated_t0_dict)
len(treated_eids)

# Controls: those not in treated group
# (untreated_eids: EIDs for controls)
treated_eids_set = set(incident_treated['eid'])
untreated_eids = [eid for eid in processed_ids if eid not in treated_eids_set]

# When selecting from covariate DataFrame:
controls_df = cov[cov['eid'].isin(untreated_eids)]
controls = cov[cov['eid'].isin(untreated_eids)].copy()
controls['years_since_30'] = (controls['age_enrolled'] - 30).round()
control_eids = controls['eid']
control_t0s = controls['years_since_30']
print(len(untreated_eids))
print(len(treated_eids_set))
print(len(untreated_eids)+len(treated_eids_set))

# ---------------------------------------------
# 10. Build covariate dictionary for feature engineering
# ---------------------------------------------
covariate_dicts = {
    'age_at_enroll': eid_to_age,
    'sex': eid_to_sex,
    'dm2_prev': eid_to_dm2_prev,
    'antihtnbase': eid_to_antihtnbase,
    'dm1_prev': eid_to_hl_prev,  # Note: check if this is intentional
    'smoke': eid_to_smoke,
    'ldl_prs': eid_to_ldl_prs,
    'cad_prs': eid_to_cad_prs,
    'tchol': eid_to_tchol,
    'hdl': eid_to_hdl,
    'sbp': eid_to_sbp,
    'pce_goff': eid_to_pce_goff
}

# ---------------------------------------------
# 11. Prepare lists of EIDs and t0s for treated and controls
# ---------------------------------------------
from dt import *

treated_eids_list = list(treated_eids)
treated_t0s_list = list(treated_t0s)
control_eids_list = list(control_eids)
control_t0s_list = list(control_t0s)

# ---------------------------------------------
# 12. Build features for matching (using build_features)
# ---------------------------------------------
treated_features, treated_indices, treated_eids_matched = build_features(
    treated_eids_list, treated_t0s_list, processed_ids, thetas, covariate_dicts
)
control_features, control_indices, control_eids_matched = build_features(
    control_eids_list, control_t0s_list, processed_ids, thetas, covariate_dicts
)

# ---------------------------------------------
# 13. Impute missing values and standardize features
# ---------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
# Impute missing values (mean imputation)
imputer = SimpleImputer(strategy='mean')
all_features = np.vstack([treated_features, control_features])
imputer.fit(all_features)
treated_features_imputed = imputer.transform(treated_features)
control_features_imputed = imputer.transform(control_features)
# Standardize features (zero mean, unit variance)
scaler = StandardScaler().fit(np.vstack([treated_features_imputed, control_features_imputed]))
treated_features_std = scaler.transform(treated_features_imputed)
control_features_std = scaler.transform(control_features_imputed)
print(np.isnan(control_features_std).sum())
print(np.isnan(treated_features_std).sum())

# ---------------------------------------------
# 14. Perform nearest neighbor matching (1:1 matching)
# ---------------------------------------------
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(control_features_std)
distances, indices = nn.kneighbors(treated_features_std)
matched_control_indices = [control_indices[i[0]] for i in indices]
matched_treated_indices = treated_indices  # these are the treateds you matched
matched_control_eids = [control_eids_matched[i] for i in indices.flatten()]

# ---------------------------------------------
# 15. Calculate mean PRS for matched/unmatched cases and controls
# ---------------------------------------------
treated_indices = [np.where(processed_ids == int(eid))[0][0] for eid in treated_eids]
untreated_indices = [np.where(processed_ids == int(eid))[0][0] for eid in untreated_eids]

means_matched_cases = G[matched_treated_indices].mean(axis=0)
means_matched_controls = G[matched_control_indices].mean(axis=0)
means_unmatched_cases = G[treated_indices].mean(axis=0)
means_unmatched_controls = G[untreated_indices].mean(axis=0)

# ---------------------------------------------
# 16. Plot mean PRS for cases/controls (matched and unmatched)
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(len(prs_labels))
width = 0.35
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - width/2, means_unmatched_cases, width, label='Cases (all)')
ax.bar(x + width/2, means_unmatched_controls, width, label='Controls (all)')
ax.set_ylabel('Mean PGS Value')
ax.set_title('Mean PGS for Cases and Controls (Unmatched)')
ax.set_xticks(x)
ax.set_xticklabels(prs_labels, rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - width/2, means_matched_cases, width, label='Cases (matched)')
ax.bar(x + width/2, means_matched_controls, width, label='Controls (matched)')
ax.set_ylabel('Mean PGS Value')
ax.set_title('Mean PGS for Cases and Controls (Matched)')
ax.set_xticks(x)
ax.set_xticklabels(prs_labels, rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 17. Compute and plot standardized mean differences (SMD) for PRS
# ---------------------------------------------
def compute_smd(x1, x0):
    m1, m0 = np.nanmean(x1), np.nanmean(x0)
    s1, s0 = np.nanstd(x1), np.nanstd(x0)
    return np.abs(m1 - m0) / np.sqrt((s1**2 + s0**2) / 2)

smds_matched = []
for i, prs in enumerate(prs_labels):
    smds_matched.append(compute_smd(
        G[matched_treated_indices, i], G[matched_control_indices, i]
    ))

x = np.arange(len(prs_labels))
plt.figure(figsize=(16,6))
plt.bar(x, smds_matched, width=0.6)
plt.axhline(0.1, color='red', linestyle='--', label='SMD=0.1')
plt.xticks(x, prs_labels, rotation=90)
plt.ylabel('Standardized Mean Difference')
plt.title('SMD for PGS (Matched)')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 18. Build group DataFrames for summary tables
# ---------------------------------------------
treated_eids = list(treated_eids)
groups = {
    'Treated (all)': cov[cov['eid'].isin(treated_eids_matched)],
    'Control (all)': cov[cov['eid'].isin(untreated_eids)],
    'Treated (matched)': cov[cov['eid'].isin(treated_eids_matched)],
    'Control (matched)': cov[cov['eid'].isin(matched_control_eids)],
}

print(groups['Treated (all)'].shape)
print(groups['Control (all)'].shape)
print(groups['Treated (matched)'].shape)
print(groups['Control (matched)'].shape)
len(matched_control_eids)

missing = set(matched_control_eids) - set(cov['eid'])
print(f"Number of missing matched control EIDs: {len(missing)}")
print(list(missing)[:10])  # Show a few examples
print(f"Number of unique matched_control_eids: {len(set(matched_control_eids))}")
print(f"Total matched_control_eids: {len(matched_control_eids)}")

# ---------------------------------------------
# 19. Summarize categorical covariates by group
# ---------------------------------------------
categorical_covariates = ['Sex', 'SmokingStatusv2', 'prev_dm', 'prev_hl','race','prev_ht']
for covariate in categorical_covariates:
    # Get all possible categories
    categories = set()
    for df in groups.values():
        categories.update(df[covariate].dropna().unique())
    categories = sorted(categories)
    cat_summary = pd.DataFrame(index=categories)
    for group_name, df in groups.items():
        cat_summary[group_name] = df[covariate].value_counts(normalize=True).reindex(categories, fill_value=0)
    print(f"\n{covariate} distribution table:")
    display(cat_summary.round(3))

# ---------------------------------------------
# 20. Summarize continuous covariates by group
# ---------------------------------------------
covariates = ['age_at_enroll', 'tchol', 'hdl', 'SBP']
summary = pd.DataFrame(index=covariates)
groups = {
    'Treated (all)': cov[cov['eid'].isin(treated_eids_matched)],
    'Control (all)': cov[cov['eid'].isin(untreated_eids)],
    'Treated (matched)': cov[cov['eid'].isin(treated_eids_matched)],
    'Control (matched)': cov[cov['eid'].isin(matched_control_eids)],
}
for group_name, df in groups.items():
    summary[(group_name, 'mean')] = df[covariates].mean()
    summary[(group_name, 'std')] = df[covariates].std()
# Reorder columns for clarity
summary = summary.reindex(columns=pd.MultiIndex.from_product(
    [['Treated (all)', 'Control (all)', 'Treated (matched)', 'Control (matched)'], ['mean', 'std']]
))
# Display as a nice table
summary.round(2)

# ---------------------------------------------
# 21. Compute and plot SMD for phenotypic covariates (matched)
# ---------------------------------------------
treated_matched = cov[cov['eid'].isin(treated_eids_matched)]
control_matched = cov[cov['eid'].isin(matched_control_eids)]
def compute_smd(x1, x0):
    m1, m0 = np.nanmean(x1), np.nanmean(x0)
    s1, s0 = np.nanstd(x1), np.nanstd(x0)
    return np.abs(m1 - m0) / np.sqrt((s1**2 + s0**2) / 2)

covariates = ['age_enrolled', 'tchol', 'hdl', 'SBP', 'pce_goff','Sex','prev_hl','prev_dm','antihtnbase']
smds_matched = []
for covariate in covariates:
    smds_matched.append(compute_smd(
        treated_matched[covariate], control_matched[covariate]
    ))

x = np.arange(len(covariates))
plt.figure(figsize=(8,4))
plt.bar(x, smds_matched, width=0.6)
plt.axhline(0.1, color='red', linestyle='--', label='SMD=0.1')
plt.xticks(x, covariates, rotation=45)
plt.ylabel('Standardized Mean Difference')
plt.title('SMD for Phenotypic Covariates (Matched)')
plt.legend()
plt.tight_layout()
plt.show()

# Print SMDs for reference
for covariate, smd in zip(covariates, smds_matched):
    print(f"{covariate}: SMD = {smd:.3f}")