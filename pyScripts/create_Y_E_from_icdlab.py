"""
Create Y and E tensors from existing icdlab RDS file
Adapted from sample_for_RAP.py to work with pre-processed icdlab data
"""

import pandas as pd
import numpy as np
import torch
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Activate pandas conversion
pandas2ri.activate()

# Load RDS file
print("="*80)
print("Loading icdlab RDS file...")
print("="*80)

base = importr('base')
icdlab = base.readRDS("~/Dropbox-Personal/icd10phe_lab.rds")

# Convert to pandas DataFrame
df = pandas2ri.rpy2py(icdlab)
print(f"Loaded {len(df):,} rows")
print(f"Columns: {list(df.columns)}")
print(f"Unique patients: {df.iloc[:, 0].nunique():,}")

# Assume columns are: patient_id, age_diag, phecode (diag_icd10)
# Adjust column names if different
col_names = list(df.columns)
patient_col = col_names[0]  # First column should be patient ID
age_col = [c for c in col_names if 'age' in c.lower()][0] if any('age' in c.lower() for c in col_names) else col_names[1]
phecode_col = [c for c in col_names if any(x in c.lower() for x in ['phecode', 'icd', 'diag'])][0] if any(any(x in c.lower() for x in ['phecode', 'icd', 'diag']) for c in col_names) else col_names[2]

print(f"\nUsing columns:")
print(f"  Patient: {patient_col}")
print(f"  Age: {age_col}")
print(f"  Phecode: {phecode_col}")

# Rename for consistency
df = df.rename(columns={
    patient_col: 'eid',
    age_col: 'age_diag',
    phecode_col: 'diag_icd10'
})

# Filter age range (29-80)
print(f"\nFiltering to age 29-80...")
df = df[df['age_diag'].between(29, 80, inclusive='both')].copy()
df['age_idx'] = (df['age_diag'].round().astype(int) - 29)
df = df[df['age_idx'] >= 0]  # Ensure non-negative

print(f"After filtering: {len(df):,} rows")

# Integer encode
print(f"\nCreating integer encodings...")
eids = df['eid'].astype(str).unique()
phecodes = df['diag_icd10'].astype(str).unique()

eid_index = {eid: idx for idx, eid in enumerate(eids)}
phe_index = {code: idx for idx, code in enumerate(phecodes)}

n_patients = len(eids)
n_diseases = len(phecodes)
n_timepoints = df['age_idx'].max() + 1

print(f"Patients: {n_patients:,}")
print(f"Diseases (phecodes): {n_diseases:,}")
print(f"Timepoints: {n_timepoints}")

# Create Y tensor
print(f"\nCreating Y tensor...")
Y = torch.zeros((n_patients, n_diseases, n_timepoints), dtype=torch.int8)

# Fill Y tensor
print("Filling Y tensor (this may take a while)...")
for idx, row in df.iterrows():
    eid_str = str(row['eid'])
    phecode_str = str(row['diag_icd10'])
    age_idx = int(row['age_idx'])
    
    if eid_str in eid_index and phecode_str in phe_index:
        Y[eid_index[eid_str], phe_index[phecode_str], age_idx] = 1
    
    if (idx + 1) % 100000 == 0:
        print(f"  Processed {idx + 1:,} / {len(df):,} rows")

print("✓ Y tensor created")

# Create E tensor (event times)
print(f"\nCreating E tensor...")
def create_event_matrix(Y_tensor):
    n_patients, n_diseases, n_times = Y_tensor.shape
    max_time = n_times - 1
    E = torch.full((n_patients, n_diseases), max_time, dtype=torch.int16)
    
    events = torch.nonzero(Y_tensor == 1, as_tuple=False)
    for event in events:
        patient, disease, time = event[0].item(), event[1].item(), event[2].item()
        if time < E[patient, disease].item():
            E[patient, disease] = time
    
    return E

E = create_event_matrix(Y)
print("✓ E tensor created")

# Save outputs
print(f"\nSaving outputs...")
output_dir = "~/Dropbox-Personal/data_for_running/"
import os
output_dir = os.path.expanduser(output_dir)
os.makedirs(output_dir, exist_ok=True)

torch.save(Y, os.path.join(output_dir, "Y_tensor.pt"))
torch.save(E, os.path.join(output_dir, "E_matrix.pt"))

# Also save as numpy format
np.savez_compressed(
    os.path.join(output_dir, "Y_tensor.npz"),
    Y=Y.numpy()
)
np.savez_compressed(
    os.path.join(output_dir, "E_matrix.npz"),
    E=E.numpy()
)

# Save metadata
import pickle
metadata = {
    'n_patients': n_patients,
    'n_diseases': n_diseases,
    'n_timepoints': n_timepoints,
    'eids': eids,
    'phecodes': phecodes,
    'eid_index': eid_index,
    'phecode_index': phe_index
}

with open(os.path.join(output_dir, "Y_E_metadata.pkl"), 'wb') as f:
    pickle.dump(metadata, f)

print("="*80)
print("COMPLETE")
print("="*80)
print(f"Y tensor shape: {Y.shape}")
print(f"E tensor shape: {E.shape}")
print(f"Output directory: {output_dir}")
print(f"\nFiles saved:")
print(f"  - Y_tensor.pt")
print(f"  - E_matrix.pt")
print(f"  - Y_tensor.npz")
print(f"  - E_matrix.npz")
print(f"  - Y_E_metadata.pkl")
print("="*80)

