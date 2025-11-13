"""
Comparison of Aladynoulli results with Delphi-2M (Shmatko et al., Nature 2025)

This script creates a comprehensive comparison table showing:
- Aladynoulli 5-year predictions
- Aladynoulli 10-year predictions  
- Aladynoulli 1-year washout (prospective prediction)
- Delphi-2M results (to be filled from Nature paper supplementary tables)

Reference: Shmatko et al. (2025) "Learning the natural history of human disease 
with generative transformers" Nature 647, 248-256
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Load your results
washout_df = pd.read_csv("/Users/sarahurbut/aladynoulli2/claudefile/output/washout_summary_table.csv")

# Try to load Delphi-2M results if available
delphi_results = {}
delphi_pkl = Path("/Users/sarahurbut/aladynoulli2/claudefile/output/delphi_results.pkl")
if delphi_pkl.exists():
    with open(delphi_pkl, "rb") as f:
        delphi_data = pickle.load(f)
        # Convert to simple format
        for disease, data in delphi_data.items():
            delphi_results[disease] = data.get('1yr', np.nan)
    print(f"Loaded Delphi-2M results for {len(delphi_results)} diseases")
else:
    print("Delphi-2M results not found. Run download_delphi_data.py first to download and process the data.")

# Load 5-year and 10-year results if available
output_dir = Path("/Users/sarahurbut/aladynoulli2/claudefile/output")

# Try to load pickled results
results_5yr = None
results_10yr = None

if (output_dir / "batch_results_fixed_5yr.pkl").exists():
    with open(output_dir / "batch_results_fixed_5yr.pkl", "rb") as f:
        batch_results_5yr = pickle.load(f)
    
    # Compute median AUCs for 5-year
    disease_aucs_5yr = {}
    for batch_res in batch_results_5yr.values():
        for disease, metrics in batch_res.items():
            disease_aucs_5yr.setdefault(disease, []).append(metrics.get("auc", np.nan))
    
    results_5yr = pd.DataFrame({
        "Disease": list(disease_aucs_5yr.keys()),
        "AUC_5yr": [np.nanmedian(vals) for vals in disease_aucs_5yr.values()],
        "IQR_5yr": [
            (np.nanpercentile(vals, 25), np.nanpercentile(vals, 75))
            for vals in disease_aucs_5yr.values()
        ]
    })

if (output_dir / "batch_results_fixed_10yr.pkl").exists():
    with open(output_dir / "batch_results_fixed_10yr.pkl", "rb") as f:
        batch_results_10yr = pickle.load(f)
    
    # Compute median AUCs for 10-year
    disease_aucs_10yr = {}
    for batch_res in batch_results_10yr.values():
        for disease, metrics in batch_res.items():
            disease_aucs_10yr.setdefault(disease, []).append(metrics.get("auc", np.nan))
    
    results_10yr = pd.DataFrame({
        "Disease": list(disease_aucs_10yr.keys()),
        "AUC_10yr": [np.nanmedian(vals) for vals in disease_aucs_10yr.values()],
        "IQR_10yr": [
            (np.nanpercentile(vals, 25), np.nanpercentile(vals, 75))
            for vals in disease_aucs_10yr.values()
        ]
    })

# Prepare washout results (0-year washout = immediate prediction, comparable to Delphi's 1-year gap)
washout_0yr = washout_df[["Disease", "0yr_AUC", "0yr_std"]].copy()
washout_0yr.columns = ["Disease", "AUC_0yr_immediate", "Std_0yr_immediate"]

# Merge all results
comparison_df = washout_df[["Disease"]].copy()

# Add 5-year results
if results_5yr is not None:
    comparison_df = comparison_df.merge(results_5yr[["Disease", "AUC_5yr"]], on="Disease", how="left")

# Add 10-year results  
if results_10yr is not None:
    comparison_df = comparison_df.merge(results_10yr[["Disease", "AUC_10yr"]], on="Disease", how="left")

# Add 0-year immediate prediction (comparable to Delphi's 1-year gap)
comparison_df = comparison_df.merge(washout_0yr[["Disease", "AUC_0yr_immediate"]], on="Disease", how="left")

# Add Delphi-2M results (1-year gap = prospective prediction, similar to our 1-year washout)
delphi_1yr_values = []
for disease in comparison_df["Disease"]:
    delphi_1yr_values.append(delphi_results.get(disease, np.nan))

comparison_df["Delphi_1yr"] = delphi_1yr_values
comparison_df["Delphi_5yr"] = np.nan  # Not available in Delphi supplementary data
comparison_df["Delphi_10yr"] = np.nan  # Not available in Delphi supplementary data

# Reorder columns for better readability
col_order = [
    "Disease",
    "AUC_0yr_immediate",
    "Delphi_1yr",
    "AUC_5yr",
    "AUC_10yr",
    "Delphi_5yr", 
    "Delphi_10yr"
]

# Only include columns that exist
col_order = [c for c in col_order if c in comparison_df.columns]
comparison_df = comparison_df[col_order]

# Sort by 5-year AUC (or 10-year if 5-year not available)
sort_col = "AUC_5yr" if "AUC_5yr" in comparison_df.columns else "AUC_10yr"
if sort_col in comparison_df.columns:
    comparison_df = comparison_df.sort_values(sort_col, ascending=False, na_position='last')

print("=" * 100)
print("ALADYNOULLI vs DELPHI-2M COMPARISON TABLE")
print("=" * 100)
print("\nNote: Delphi-2M results need to be extracted from:")
print("  - Nature 2025, Shmatko et al., Supplementary Table 2")
print("  - Main paper figures/tables")
print("\nFill in Delphi columns from the paper, then re-run to see full comparison.\n")

# Format for display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(comparison_df.to_string(index=False))

# Save to CSV
output_path = output_dir / "delphi_comparison_table.csv"
comparison_df.to_csv(output_path, index=False)
print(f"\n\nSaved comparison table to: {output_path}")

# Create a formatted version for publication
print("\n" + "=" * 100)
print("FORMATTED COMPARISON (for manuscript)")
print("=" * 100)

formatted_df = comparison_df.copy()
for col in ["AUC_0yr_immediate", "AUC_5yr", "AUC_10yr", "Delphi_1yr", "Delphi_5yr", "Delphi_10yr"]:
    if col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

print(formatted_df.to_string(index=False))

