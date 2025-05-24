import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dt import run_digital_twin_matching_single_sig,run_digital_twin_matching
import torch
import pandas as pd

st.title("Digital Twin Matching App")

# --- Load core data ---
thetas = np.load("thetas.npy")  # shape: (N, K, T)
processed_ids = np.load("processed_patient_ids.npy").astype(int)  # ensure int

# Load Y (event tensor)
try:
    Y = np.load("Y.npy")
except Exception:
    Y = torch.load("/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/Y_tensor.pt")
    if hasattr(Y, 'detach'):
        Y = Y.detach().cpu().numpy()

# --- Disease selection (place here!) ---
disease_names_df = pd.read_csv("disease_names.csv")
disease_names = disease_names_df.iloc[:, 1].tolist()
disease_idx = st.sidebar.selectbox(
    "Select disease/event to measure",
    list(range(Y.shape[1])),
    format_func=lambda x: disease_names[x]
)

# --- Load covariate file (age, year of birth, etc.) ---
covariate_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/for_regenie/ukbb_covariates_400k.txt'
cov = pd.read_csv(covariate_path, sep=' ')
cov.columns = cov.columns.str.strip()
cov = cov.rename(columns={cov.columns[0]: 'eid', cov.columns[2]: 'year_of_birth'})
cov['eid'] = cov['eid'].astype(int)
cov['enrollment'] = pd.to_datetime(cov['enrollment'], errors='coerce')
cov['age_at_enroll'] = cov['enrollment'].dt.year - cov['year_of_birth']
age_at_enroll = dict(zip(cov['eid'], cov['age_at_enroll']))

# Build age_at_enroll and eid_to_yob
eid_to_yob = dict(zip(cov['eid'], cov['year_of_birth']))

# --- Load prescription file ---
prescription_path = 'prescriptions.csv'
df_treat = pd.read_csv(prescription_path)
df_treat['eid'] = df_treat['eid'].astype(int)
presc_col = 'from'

# Merge YOB into prescription DataFrame
df_treat = df_treat.merge(cov[['eid', 'year_of_birth']], on='eid', how='left')
yob_col = 'year_of_birth'

df_treat[presc_col] = pd.to_datetime(df_treat[presc_col], errors='coerce')
time_grid = np.arange(thetas.shape[2])

# --- Streamlit controls for drug class and signature matching ---
st.sidebar.header("Analysis Options")
drug_classes = ['All'] + sorted(df_treat['category'].dropna().unique().tolist())
drug_class = st.sidebar.selectbox("Select drug class (or All)", drug_classes)

sig_options = [f"Signature {k}" for k in range(thetas.shape[1])]
match_all_sigs = st.sidebar.checkbox("Match on all signatures (multivariate)", value=False)
sig_idx = st.sidebar.selectbox("Signature for Matching (if not all)", list(range(thetas.shape[1])), format_func=lambda x: f"Signature {x}")

if match_all_sigs:
    st.sidebar.info("Signature selection above is for plotting only. Matching uses all signatures.")
else:
    st.sidebar.info("Signature selection below is used for both matching and plotting.")

# --- Filter prescriptions by drug class ---
if drug_class == 'All':
    df_drug = df_treat.copy()
else:
    df_drug = df_treat[df_treat['category'] == drug_class].copy()

# Get first prescription date for each treated patient
first_presc = df_drug.groupby('eid')[presc_col].min()

# Build treated_time_idx using only those with the drug

def get_time_index(yob, presc_date, time_grid):
    if pd.isnull(yob) or pd.isnull(presc_date):
        return None
    # Ensure presc_date is a datetime object
    if isinstance(presc_date, str):
        presc_date = pd.to_datetime(presc_date, errors='coerce')
    if pd.isnull(presc_date):
        return None
    age_at_presc = presc_date.year - yob
    return int(np.argmin(np.abs(time_grid + 30 - age_at_presc)))

treated_time_idx = {}
for eid, presc_date in first_presc.items():
    yob = eid_to_yob.get(int(eid), None)
    t0 = get_time_index(yob, presc_date, time_grid)
    if t0 is not None:
        treated_time_idx[int(eid)] = t0

treated_eids = set(treated_time_idx.keys())
untreated_eids = [eid for eid in processed_ids if eid not in treated_eids]

# --- User controls (can be hardcoded or left as sidebar) ---
window = st.sidebar.number_input("Pre-treatment Window (years)", min_value=1, max_value=30, value=10)
window_post = st.sidebar.number_input("Post-treatment Window (years)", min_value=1, max_value=30, value=10)
sample_size = st.sidebar.number_input("Sample Size for Controls", min_value=10, max_value=5000, value=1000)
age_tolerance = st.sidebar.number_input("Age Tolerance (years)", min_value=0, max_value=10, value=2)
max_cases = st.sidebar.number_input("Max Treated Cases", min_value=1, max_value=10000, value=100)

# --- Run matching ---
if st.button("Run Digital Twin Matching"):
    if not treated_time_idx or not untreated_eids or not age_at_enroll:
        st.error("Please upload and configure all required data files and columns.")
        st.stop()
    with st.spinner("Matching..."):
        if match_all_sigs:
            # Use all signatures for matching (multivariate)
            results = run_digital_twin_matching(
                treated_time_idx,
                untreated_eids,
                processed_ids,
                thetas,
                Y,
                diabetes_idx=disease_idx,
                window=window,
                window_post=window_post,
                sig_idx=None,  # None means all sigs
                sample_size=sample_size,
                max_cases=max_cases,
                age_at_enroll=age_at_enroll,
                age_tolerance=age_tolerance
            )
        else:
            results = run_digital_twin_matching_single_sig(
                treated_time_idx,
                untreated_eids,
                processed_ids,
                thetas,
                Y,
                diabetes_idx=disease_idx,
                window=window,
                window_post=window_post,
                sig_idx=sig_idx,
                sample_size=sample_size,
                max_cases=max_cases,
                age_at_enroll=age_at_enroll,
                age_tolerance=age_tolerance
            )
    st.success("Matching complete!")

    # --- Show results ---
    st.write(f"Treated event rate (disease {disease_idx}):", results['treated_event_rate'])
    st.write(f"Control event rate (disease {disease_idx}):", results['control_event_rate'])
    st.write("Number of matched pairs:", len(results['matched_pairs']))
    if len(results['matched_pairs']) == 0:
        st.warning("No matched pairs found. Try relaxing matching parameters or check your data.")

    # --- Plot mean signature trajectories ---
    if results['trajectories_treated'].ndim == 3:
        # Multivariate matching: (n_pairs, n_signatures, window_post)
        mean_treated = results['trajectories_treated'][:, sig_idx, :].mean(axis=0)
        mean_control = results['trajectories_control'][:, sig_idx, :].mean(axis=0)
    elif results['trajectories_treated'].ndim == 2:
        # Single signature matching: (n_pairs, window_post)
        mean_treated = results['trajectories_treated'].mean(axis=0)
        mean_control = results['trajectories_control'].mean(axis=0)
    else:
        st.error("Unexpected shape for trajectories_treated.")
        mean_treated = mean_control = None

    if mean_treated is not None and mean_control is not None:
        # Squeeze in case only one pair
        mean_treated = np.squeeze(mean_treated)
        mean_control = np.squeeze(mean_control)
        fig, ax = plt.subplots()
        ax.plot(mean_treated, label='Treated')
        ax.plot(mean_control, label='Control')
        ax.set_title(f'Mean signature (sig_idx={sig_idx}) trajectory after t0')
        ax.legend()
        st.pyplot(fig)

    # --- Download matched pairs ---
    df_pairs = pd.DataFrame(results['matched_pairs'], columns=['Treated_idx', 'Control_idx', 't0'])
    st.download_button("Download Matched Pairs", df_pairs.to_csv(index=False), "matched_pairs.csv")

    # Store matched event rates for plotting
    matched_treated_rate = results['treated_event_rate'] * 100  # assuming it's a fraction
    matched_control_rate = results['control_event_rate'] * 100

    # Only plot if you have also calculated raw rates
    if 'raw_treated_rate' in locals() and 'raw_untreated_rate' in locals():
        fig, ax = plt.subplots(figsize=(8, 4))
        bar_labels = ['Treated', 'Untreated']
        x = np.arange(len(bar_labels))
        width = 0.35

        # Raw rates
        raw_rates = [raw_treated_rate, raw_untreated_rate]
        # Matched rates
        matched_rates = [matched_treated_rate, matched_control_rate]

        rects1 = ax.bar(x - width/2, raw_rates, width, label='Raw')
        rects2 = ax.bar(x + width/2, matched_rates, width, label='Matched')

        ax.set_ylabel('Event Rate (%)')
        ax.set_title(f'Event Rates for {disease_names[disease_idx]}')
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels)
        ax.legend()

        # Add value labels
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        st.pyplot(fig)
    else:
        st.info("Run 'Show Raw Event Rates' first to compare with matched rates.")

if st.sidebar.button("Show Raw Event Rates"):
    # Calculate raw event rates for treated and untreated
    treated_events = 0
    treated_total = 0
    for eid, t0 in treated_time_idx.items():
        if t0 + window_post < Y.shape[2]:
            treated_total += 1
            idx = np.where(processed_ids == int(eid))[0][0]
            if np.any(Y[idx, disease_idx, t0:t0+window_post] > 0):
                treated_events += 1

    untreated_events = 0
    untreated_total = 0
    for eid in untreated_eids:
        t0 = int(age_at_enroll.get(eid, 0) - 30)
        if t0 + window_post < Y.shape[2]:
            untreated_total += 1
            idx = np.where(processed_ids == int(eid))[0][0]
            if np.any(Y[idx, disease_idx, t0:t0+window_post] > 0):
                untreated_events += 1

    raw_treated_rate = (treated_events / treated_total * 100) if treated_total > 0 else 0
    raw_untreated_rate = (untreated_events / untreated_total * 100) if untreated_total > 0 else 0

    st.write(f"**Raw Treated Event Rate:** {raw_treated_rate:.2f}% (n={treated_total})")
    st.write(f"**Raw Untreated Event Rate:** {raw_untreated_rate:.2f}% (n={untreated_total})")

st.markdown("---")
st.write("This app lets you run digital twin matching and compare outcomes for treated vs. matched controls interactively.")
