"""
Convert icd10phe_lab.rds (R data frame) to E, G, and Y tensors for Aladynoulli.

Input format (from R):
  eid: patient ID
  diag_icd10: disease code (numeric phecode)
  age_diag: age at diagnosis

Output:
  Y: (N, D, T) tensor - disease occurrence over time
  E: (N, D) tensor - event times (first occurrence or censoring)
  G: (N, P) tensor - genetic/PRS features
  disease_names: list of disease names
"""

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List


def load_icd10phe_data(icd10phe_path: Path) -> pd.DataFrame:
    """Load icd10phe_lab.rds from R."""
    pandas2ri.activate()
    print(f"Loading {icd10phe_path}...")
    df = pd.DataFrame(robjects.r['readRDS'](str(icd10phe_path)))
    print(f"✓ Loaded {len(df):,} records")
    
    # Handle column names - R data frames may have integer column names
    # Expected: eid, diag_icd10, age_diag
    # If columns are integers, map them based on position or try to infer
    if len(df.columns) >= 3:
        # Check if columns are named correctly
        col_names = [str(col).lower() for col in df.columns]
        has_eid = any('eid' in name for name in col_names)
        has_age = any('age' in name for name in col_names)
        has_diag = any('diag' in name for name in col_names)
        
        if not (has_eid and has_age and has_diag):
            # Try to infer: typically eid (col 0), diag_icd10 (col 1), age_diag (col 2)
            # But based on user's example: eid (col 0), diag_icd10 (col 1), age_diag (col 2)
            print("⚠️  Column names not recognized. Assuming standard order:")
            print(f"   Column 0: eid")
            print(f"   Column 1: diag_icd10") 
            print(f"   Column 2: age_diag")
            df.columns = ['eid', 'diag_icd10', 'age_diag'] + list(df.columns[3:])
    
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


def create_Y_tensor(
    icd10phe: pd.DataFrame,
    eid_col: str = 'eid',
    disease_col: str = 'diag_icd10',
    age_col: str = 'age_diag',
    min_age: int = 30,
    max_age: int = 80
) -> Tuple[torch.Tensor, List, np.ndarray]:
    """
    Create Y tensor from icd10phe data frame.
    
    Matches R code logic:
    - age_idx = age_diag - 29 (so age 30 -> idx 1, age 31 -> idx 2, ..., age 81 -> idx 52)
    - Filter age_idx >= 1 and age_idx <= 52
    - Create 52 time slices (T=52)
    - Y[i, j, t] = 1 if patient i had disease j at time t
    
    Returns:
        Y: (N, D, T) tensor where T=52 (ages 30-81)
        disease_names: list of unique disease names
        age_bins: array of age bins (30-81, 52 timepoints)
    """
    print("\n" + "="*80)
    print("CREATING Y TENSOR")
    print("="*80)
    
    # Get unique patients and diseases
    unique_eids = sorted(icd10phe[eid_col].unique())
    unique_diseases = sorted(icd10phe[disease_col].unique())
    
    # Create age bins: 52 timepoints (ages 30-81)
    # R code: age_idx = age_diag - 29, filter age_idx >= 1 and <= 52
    # So age 30 -> idx 0, age 31 -> idx 1, ..., age 81 -> idx 51 (0-indexed in Python)
    age_bins = np.arange(min_age, min_age + 52)  # ages 30-81 (52 timepoints)
    
    N = len(unique_eids)
    D = len(unique_diseases)
    T = 52  # Fixed to 52 timepoints as in R code
    
    print(f"  N (patients): {N:,}")
    print(f"  D (diseases): {D:,}")
    print(f"  T (timepoints): {T} (ages {min_age}-{min_age+T-1})")
    
    # Create mappings (1-indexed in R, 0-indexed in Python)
    eid_to_idx = {eid: idx for idx, eid in enumerate(unique_eids)}
    disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
    
    # Initialize Y tensor
    Y = torch.zeros((N, D, T), dtype=torch.float32)
    
    print("\nPopulating Y tensor...")
    print("  Matching R logic: age_idx = age_diag - 29, filter age_idx >= 1 and <= 52")
    
    # Filter data: age_idx = age_diag - 29, keep where age_idx >= 1 and <= 52
    # In Python (0-indexed): age_idx = age_diag - 30, keep where age_idx >= 0 and < 52
    filtered_data = icd10phe.copy()
    filtered_data['age_idx'] = filtered_data[age_col] - 30  # age 30 -> idx 0, age 31 -> idx 1, etc.
    filtered_data = filtered_data[
        (filtered_data['age_idx'] >= 0) & 
        (filtered_data['age_idx'] < 52) &
        (~filtered_data[age_col].isna())
    ]
    
    print(f"  Filtered to {len(filtered_data):,} records with ages 30-81")
    
    n_records = len(filtered_data)
    for idx, row in filtered_data.iterrows():
        if idx % 100000 == 0 and n_records > 100000:
            print(f"  Processed {idx:,} / {n_records:,} records ({100*idx/n_records:.1f}%)...")
        
        eid = row[eid_col]
        age = row[age_col]
        disease = row[disease_col]
        age_idx = int(row['age_idx'])  # Already computed: age - 30
        
        # Get indices
        patient_idx = eid_to_idx[eid]
        disease_idx = disease_to_idx[disease]
        time_idx = age_idx  # age 30 -> 0, age 31 -> 1, ..., age 81 -> 51
        
        # Mark disease occurrence (set to 1 if occurred at this time)
        Y[patient_idx, disease_idx, time_idx] = 1.0
    
    print(f"\n✓ Y tensor created: {Y.shape}")
    print(f"  Total disease occurrences: {Y.sum().item():,}")
    print(f"  Patients with at least one disease: {(Y.sum(dim=(1,2)) > 0).sum().item():,}")
    
    disease_names = [str(d) for d in unique_diseases]
    return Y, disease_names, age_bins


def create_E_matrix(Y: torch.Tensor, T: int) -> torch.Tensor:
    """
    Create E matrix (event times) from Y tensor.
    
    E[i,d] = time of first occurrence of disease d for patient i, or T-1 if censored.
    """
    print("\n" + "="*80)
    print("CREATING E MATRIX")
    print("="*80)
    
    N, D = Y.shape[0], Y.shape[1]
    E = torch.full((N, D), T - 1, dtype=torch.long)
    
    print("Finding first occurrence of each disease for each patient...")
    for patient_idx in range(N):
        if patient_idx % 50000 == 0:
            print(f"  Processed {patient_idx:,} / {N:,} patients ({100*patient_idx/N:.1f}%)...")
        
        for disease_idx in range(D):
            # Find first time point where disease occurs
            disease_times = torch.where(Y[patient_idx, disease_idx, :] > 0)[0]
            if len(disease_times) > 0:
                E[patient_idx, disease_idx] = disease_times[0].item()
    
    print(f"\n✓ E matrix created: {E.shape}")
    print(f"  Patients with at least one event: {(E != (T-1)).any(dim=1).sum().item():,}")
    print(f"  Total events: {(E != (T-1)).sum().item():,}")
    print(f"  Average events per patient: {(E != (T-1)).sum().float() / N:.2f}")
    
    return E


def load_G_matrix(
    data_path: Path,
    N: int,
    eid_order: Optional[List] = None
) -> torch.Tensor:
    """
    Load G matrix (genetic/PRS features).
    
    Tries multiple common locations and formats.
    """
    print("\n" + "="*80)
    print("LOADING G MATRIX")
    print("="*80)
    
    g_paths = [
        data_path / "G_subset_forsparse.rds",
        data_path / "G_matrix.pt",
        data_path / "baselinagefamh_withpcs.csv"
    ]
    
    G = None
    for g_path in g_paths:
        if g_path.exists():
            print(f"  Found G data at: {g_path}")
            
            if g_path.suffix == '.rds':
                G = np.array(robjects.r['readRDS'](str(g_path)))
                # Transpose if needed (should be N x P)
                if G.shape[0] != N:
                    G = G.T
                    
            elif g_path.suffix == '.pt':
                G = torch.load(str(g_path), weights_only=False).numpy()
                
            elif g_path.suffix == '.csv':
                # Load CSV and extract PC columns
                baseline_df = pd.read_csv(g_path)
                pc_cols = [col for col in baseline_df.columns if 'PC' in col or 'pc' in col]
                if len(pc_cols) > 0:
                    G = baseline_df[pc_cols].values
                    # Match patients by eid if needed
                    if 'eid' in baseline_df.columns and eid_order is not None:
                        baseline_eids = baseline_df['eid'].values
                        eid_to_baseline_idx = {eid: idx for idx, eid in enumerate(baseline_eids)}
                        G_ordered = np.zeros((N, len(pc_cols)))
                        for idx, eid in enumerate(eid_order):
                            if eid in eid_to_baseline_idx:
                                G_ordered[idx] = G[eid_to_baseline_idx[eid]]
                        G = G_ordered
            
            if G is not None and G.shape[0] == N:
                print(f"  ✓ Loaded G matrix: {G.shape} (N x P)")
                break
    
    if G is None:
        print("  ⚠️  Could not find G matrix. Creating placeholder.")
        G = np.zeros((N, 10))  # Placeholder: 10 PCs
    
    G = torch.FloatTensor(G)
    P = G.shape[1]
    print(f"\n✓ G matrix: {G.shape} (N x P, P={P})")
    
    return G


def save_tensors(
    Y: torch.Tensor,
    E: torch.Tensor,
    G: torch.Tensor,
    disease_names: List[str],
    output_dir: Path,
    summary: Optional[dict] = None,
    overwrite: bool = False
):
    """Save all tensors and metadata."""
    print("\n" + "="*80)
    print("SAVING TENSORS")
    print("="*80)
    
    # HARDCODED PROTECTION: NEVER allow writing to data_for_running/ directory
    output_str = str(output_dir).lower()
    if 'data_for_running' in output_str and 'data_for_running_new' not in output_str and 'data_for_running_' not in output_str:
        print("\n" + "="*80)
        print("🚨🚨🚨 CRITICAL PROTECTION: BLOCKED WRITE TO data_for_running/ 🚨🚨🚨")
        print("="*80)
        print("THIS SCRIPT WILL NEVER WRITE TO data_for_running/ DIRECTORY!")
        print("This is a protected directory containing your original data files.")
        print(f"\nAttempted output directory: {output_dir}")
        print("\nPlease use a DIFFERENT output directory, such as:")
        print("  - data_for_running_new")
        print("  - data_for_running_YYYYMMDD")
        print("  - Any other directory name")
        print("\n" + "="*80)
        raise PermissionError(
            "BLOCKED: Cannot write to data_for_running/ directory. "
            "This directory is protected. Use a different output directory."
        )
    
    # Check for existing files
    existing_files = []
    for filename in ["Y_tensor.pt", "E_matrix.pt", "G_matrix.pt", "disease_names.pkl"]:
        if (output_dir / filename).exists():
            existing_files.append(filename)
    
    if existing_files and not overwrite:
        print("\n" + "="*80)
        print("🚨 CRITICAL ERROR: EXISTING FILES FOUND!")
        print("="*80)
        print("⚠️  WARNING: Existing files found in output directory:")
        for f in existing_files:
            file_path = output_dir / f
            size = file_path.stat().st_size if file_path.exists() else 0
            size_mb = size / (1024 * 1024)
            print(f"     - {f} ({size_mb:.2f} MB)")
        print("\n" + "="*80)
        print("ERROR: Would overwrite existing files!")
        print("="*80)
        print("To prevent accidental data loss, this script will NOT overwrite existing files.")
        print(f"\nOptions:")
        print(f"  1. Use a different --output_dir")
        print(f"  2. Move/rename existing files first")
        print(f"  3. Set overwrite=True in the function call (use with EXTREME caution!)")
        print("\n" + "="*80)
        print("ABORTING TO PREVENT DATA LOSS")
        print("="*80)
        raise FileExistsError(
            f"Cannot save: files already exist in {output_dir}. "
            "Set overwrite=True to override, or use a different output directory."
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(Y, output_dir / "Y_tensor.pt")
    print(f"  ✓ Saved Y: {Y.shape}")
    
    torch.save(E, output_dir / "E_matrix.pt")
    print(f"  ✓ Saved E: {E.shape}")
    
    torch.save(G, output_dir / "G_matrix.pt")
    print(f"  ✓ Saved G: {G.shape}")
    
    # Save disease names
    import pickle
    with open(output_dir / "disease_names.pkl", "wb") as f:
        pickle.dump(disease_names, f)
    print(f"  ✓ Saved disease names: {len(disease_names)} diseases")
    
    # Save summary
    if summary is None:
        summary = {
            'N': Y.shape[0],
            'D': Y.shape[1],
            'T': Y.shape[2],
            'P': G.shape[1],
            'total_events': (E != (Y.shape[2]-1)).sum().item(),
            'patients_with_events': (E != (Y.shape[2]-1)).any(dim=1).sum().item()
        }
    
    with open(output_dir / "data_summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    print(f"  ✓ Saved summary")
    
    print(f"\nAll files saved to: {output_dir}")


def main(
    icd10phe_path: Path,
    output_dir: Path,
    data_path: Optional[Path] = None,
    min_age: int = 30,
    max_age: int = 80,
    eid_col: str = 'eid',
    disease_col: str = 'diag_icd10',
    age_col: str = 'age_diag',
    overwrite: bool = False
):
    """
    Main function to create E, G, Y tensors from icd10phe_lab.rds.
    
    Parameters:
    -----------
    icd10phe_path : Path to icd10phe_lab.rds file
    output_dir : Directory to save output tensors
        ⚠️  WARNING: Cannot be 'data_for_running' - this directory is protected!
    data_path : Directory containing G matrix (if None, uses output_dir)
    min_age : Minimum age for time bins (default: 30)
    max_age : Maximum age for time bins (default: 80)
    eid_col : Column name for patient ID (default: 'eid')
    disease_col : Column name for disease code (default: 'diag_icd10')
    age_col : Column name for age at diagnosis (default: 'age_diag')
    overwrite : Whether to overwrite existing files (default: False)
        ⚠️  NOTE: Even if True, cannot write to data_for_running/ directory
    """
    # HARDCODED PROTECTION: Check output_dir BEFORE doing anything
    output_str = str(output_dir).lower()
    if 'data_for_running' in output_str and 'data_for_running_new' not in output_str and 'data_for_running_' not in output_str:
        print("\n" + "="*80)
        print("🚨🚨🚨 CRITICAL PROTECTION: BLOCKED WRITE TO data_for_running/ 🚨🚨🚨")
        print("="*80)
        print("THIS SCRIPT WILL NEVER WRITE TO data_for_running/ DIRECTORY!")
        print("This is a protected directory containing your original data files.")
        print(f"\nAttempted output directory: {output_dir}")
        print("\nPlease use a DIFFERENT output directory, such as:")
        print("  - data_for_running_new")
        print("  - data_for_running_YYYYMMDD")
        print("  - Any other directory name")
        print("\n" + "="*80)
        raise PermissionError(
            "BLOCKED: Cannot write to data_for_running/ directory. "
            "This directory is protected. Use a different output directory."
        )
    
    # Load data
    icd10phe = load_icd10phe_data(icd10phe_path)
    
    # Create Y tensor
    Y, disease_names, age_bins = create_Y_tensor(
        icd10phe,
        eid_col=eid_col,
        disease_col=disease_col,
        age_col=age_col,
        min_age=min_age,
        max_age=max_age
    )
    
    # Create E matrix
    E = create_E_matrix(Y, T=len(age_bins))
    
    # Load G matrix
    if data_path is None:
        data_path = output_dir
    unique_eids = sorted(icd10phe[eid_col].unique())
    G = load_G_matrix(data_path, N=Y.shape[0], eid_order=unique_eids)
    
    # Save everything (with safety check - won't overwrite existing files unless overwrite=True)
    save_tensors(Y, E, G, disease_names, output_dir, overwrite=overwrite)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Y tensor: {Y.shape} (patients × diseases × timepoints)")
    print(f"E matrix: {E.shape} (patients × diseases, event times)")
    print(f"G matrix: {G.shape} (patients × genetic features)")
    print(f"Disease names: {len(disease_names)} diseases")
    
    return Y, E, G, disease_names


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create E, G, Y tensors from icd10phe_lab.rds")
    parser.add_argument("--icd10phe_path", type=str, required=True,
                       help="Path to icd10phe_lab.rds file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save output tensors (WILL NOT overwrite existing files)")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Directory containing G matrix (default: same as output_dir)")
    parser.add_argument("--min_age", type=int, default=30,
                       help="Minimum age for time bins (default: 30)")
    parser.add_argument("--max_age", type=int, default=80,
                       help="Maximum age for time bins (default: 80)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Allow overwriting existing files (use with caution!)")
    
    args = parser.parse_args()
    
    main(
        icd10phe_path=Path(args.icd10phe_path),
        output_dir=Path(args.output_dir),
        data_path=Path(args.data_path) if args.data_path else None,
        min_age=args.min_age,
        max_age=args.max_age,
        overwrite=args.overwrite
    )

