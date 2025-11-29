import pandas as pd
import numpy as np
import torch

phe_path = "/Dropbox-Personal/icd10phe_lab.csv"
df = pd.read_csv(phe_path)

# keep phecode events after age 29 and before 81, match your earlier scripts
df = df[df["age_diag"].between(29, 80)]
df["age_idx"] = (df["age_diag"].round().astype(int) - 29)

# integer encode
eids = df["eid"].astype(str).unique()
phecodes = df["diag_icd10"].astype(str).unique()
eid_index = {eid: idx for idx, eid in enumerate(eids)}
phe_index = {code: idx for idx, code in enumerate(phecodes)}

n_patients = len(eids)
n_diseases = len(phecodes)
n_timepoints = df["age_idx"].max() + 1

Y = torch.zeros((n_patients, n_diseases, n_timepoints), dtype=torch.int8)

for _, row in df.iterrows():
    Y[eid_index[str(row["eid"])], phe_index[row["diag_icd10"]], row["age_idx"]] = 1

def create_event_matrix(Y_tensor):
    n_patients, n_diseases, n_times = Y_tensor.shape
    max_time = n_times - 1
    E = torch.full((n_patients, n_diseases), max_time, dtype=torch.int16)
    events = torch.nonzero(Y_tensor == 1, as_tuple=False)
    for patient, disease, time in events:
        if time < E[patient, disease]:
            E[patient, disease] = time.item()
    return E

E = create_event_matrix(Y)

torch.save(Y, "/mnt/project/exports/Y_tensor.pt")
torch.save(E, "/mnt/project/exports/E_matrix.pt")
np.savez_compressed("/mnt/project/exports/Y_tensor.npz", Y=Y.numpy())