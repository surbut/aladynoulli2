# New Repository Structure Plan

## Goal
Create a clean, self-contained repository for reviewer responses that includes everything needed.

---

## Proposed Structure

```
aladynoulli_reviewer_responses/
├── README.md                          # Main entry point
├── notebooks/                         # All analysis notebooks
│   ├── index.ipynb                    # Master index (renamed from REVIEWER_QUESTIONS_INDEX)
│   │
│   ├── R1/                            # Referee #1 notebooks
│   │   ├── R1_Q1_Selection_Bias.ipynb
│   │   ├── R1_Q2_Lifetime_Risk.ipynb
│   │   ├── R1_Q3_Clinical_Meaning.ipynb
│   │   ├── R1_Q7_Heritability.ipynb
│   │   ├── R1_Q9_AUC_Comparisons.ipynb
│   │   └── R1_Q10_Age_Specific.ipynb
│   │
│   ├── R2/                            # Referee #2 notebooks
│   │   └── R2_Temporal_Leakage.ipynb
│   │
│   ├── R3/                            # Referee #3 notebooks
│   │   ├── R3_Q1_Participation_Bias.ipynb
│   │   ├── R3_Q3_Washout_Windows.ipynb
│   │   ├── R3_Q4_Competing_Risks.ipynb
│   │   ├── R3_Q8_Heterogeneity.ipynb
│   │   ├── R3_Q12_Heritability_Methods.ipynb
│   │   ├── R3_Population_Stratification_Ancestry.ipynb
│   │   ├── R3_Fixed_vs_Joint_Phi_Comparison.ipynb
│   │   ├── R3_FullE_vs_ReducedE_Comparison.ipynb
│   │   └── R3_Linear_vs_NonLinear_Mixing.ipynb
│   │
│   ├── framework/                     # Framework/overview notebooks
│   │   └── Discovery_Prediction_Framework_Overview.ipynb
│   │
│   └── supporting/                    # Supporting analyses (referenced by main notebooks)
│       ├── fh_analysis_summary.ipynb
│       ├── pc_analysis_clean.ipynb
│       ├── ipw_analysis_summary.ipynb
│       ├── heritability_analysis_summary.ipynb
│       ├── heterogeneity_analysis_summary.ipynb
│       ├── delphicomp.ipynb
│       └── performancen_notebook_clean.ipynb
│
├── source/                            # Model source code (for reproducibility)
│   ├── clust_huge_amp.py              # Main model (joint phi estimation - discovery mode)
│   ├── clust_huge_amp_fixedPhi.py     # Fixed phi model (prediction mode)
│   ├── weighted_aladyn.py             # IPW weighted model (selection bias correction)
│   └── README.md                      # How to run the models
│
├── preprocessing/                      # Data preprocessing and initialization scripts
│   ├── compute_prevalence.py           # Calculate prevalence_t (weighted/unweighted)
│   ├── create_initial_clusters.py     # Spectral clustering for disease clusters
│   ├── create_initial_psi.py           # Calculate initial psi from clusters
│   ├── create_reference_trajectories.py # Calculate signature references
│   ├── create_model_essentials.py      # Create model_essentials.pt (disease names, etc.)
│   ├── compute_ipw_weights.py          # Calculate IPW weights (if separate script)
│   └── README.md                       # How to run preprocessing pipeline
│
├── scripts/                            # Supporting Python scripts
│   ├── analyze_fh_carriers_signature.py
│   ├── analyze_age_offset_signatures.py
│   ├── generate_washout_predictions.py
│   ├── generate_age_offset_predictions.py
│   ├── compare_with_external_scores.py
│   ├── visualize_prediction_drops.py
│   └── ... (other supporting scripts)
│
├── docs/                              # Documentation
│   ├── README_FOR_REVIEWERS.md
│   ├── REVIEWER_RESPONSE_ORGANIZATION.md
│   └── ... (other docs)
│
├── results/                           # Analysis results (CSVs, plots, etc.)
│   └── ...
│
└── data/                              # Data paths/config (if needed)
    └── paths_config.py                # Centralized path configuration
```

---

## What to Copy

### ✅ Must Copy (Core Analyses)
- All notebooks in `reviewer_responses/notebooks/`
- Supporting analysis notebooks:
  - `fh_analysis_summary.ipynb`
  - `pc_analysis_clean.ipynb`
  - `ipw_analysis_summary.ipynb`
  - `heritability_analysis_summary.ipynb`
  - `heterogeneity_analysis_summary.ipynb`
  - `delphicomp.ipynb`
  - `performancen_notebook_clean.ipynb`

### ✅ Must Copy (Source Code)
- `pyScripts_forPublish/clust_huge_amp.py` → `source/clust_huge_amp.py`
- `pyScripts_forPublish/clust_huge_amp_fixedPhi.py` → `source/clust_huge_amp_fixedPhi.py`
- `pyScripts_forPublish/weighted_aladyn.py` → `source/weighted_aladyn.py`
- Any other model files needed for reproducibility

### ✅ Must Copy (Preprocessing Scripts)
- `pyScripts_forPublish/weightedprev.py` → `preprocessing/compute_prevalence.py`
- Functions for creating reference trajectories (from `pyScripts/new_clust.py` or notebooks)
- Scripts for creating initial clusters (spectral clustering code)
- Scripts for creating initial psi (from clusters)
- Scripts for creating model_essentials.pt
- IPW weight calculation scripts (if separate)

### ✅ Must Copy (Scripts)
- All scripts referenced by notebooks
- Scripts in `reviewer_responses/` parent directory that are used

### ✅ Must Copy (Docs)
- All docs in `reviewer_responses/docs/`
- README files

### ✅ Must Copy (Results)
- `reviewer_responses/notebooks/results/` directory
- Any other results directories referenced

### ⚠️ Update Paths
- All notebooks will need path updates to work in new repo structure
- Create `data/paths_config.py` for centralized path management

---

## Steps to Create New Repo

1. **Create new directory structure**
2. **Copy all files** (use script below)
3. **Update paths** in notebooks to new structure
4. **Test** that notebooks run
5. **Create README** with clear instructions

---

## Copy Script (to be created)

```bash
#!/bin/bash
# copy_to_new_repo.sh

NEW_REPO="../aladynoulli_reviewer_responses"
mkdir -p "$NEW_REPO"

# Copy notebooks
cp -r reviewer_responses/notebooks/* "$NEW_REPO/notebooks/"

# Copy supporting analyses
cp fh_analysis_summary.ipynb "$NEW_REPO/notebooks/supporting/"
cp pc_analysis_clean.ipynb "$NEW_REPO/notebooks/supporting/"
cp ipw_analysis_summary.ipynb "$NEW_REPO/notebooks/supporting/"
cp heritability_analysis_summary.ipynb "$NEW_REPO/notebooks/supporting/"
cp heterogeneity_analysis_summary.ipynb "$NEW_REPO/notebooks/supporting/"
cp delphicomp.ipynb "$NEW_REPO/notebooks/supporting/"
cp performancen_notebook_clean.ipynb "$NEW_REPO/notebooks/supporting/"

# Copy source code
cp pyScripts_forPublish/clust_huge_amp.py "$NEW_REPO/source/"
cp pyScripts_forPublish/clust_huge_amp_fixedPhi.py "$NEW_REPO/source/"
cp pyScripts_forPublish/weighted_aladyn.py "$NEW_REPO/source/"

# Copy preprocessing scripts
cp pyScripts_forPublish/weightedprev.py "$NEW_REPO/preprocessing/compute_prevalence.py"
# TODO: Extract and copy other preprocessing functions (clusters, psi, reference trajectories, model_essentials)

# Copy scripts
cp *.py "$NEW_REPO/scripts/" 2>/dev/null || true

# Copy docs
cp -r reviewer_responses/docs/* "$NEW_REPO/docs/"

# Copy results
cp -r reviewer_responses/notebooks/results "$NEW_REPO/results/"

echo "Done! Now update paths in notebooks."
```

---

## Path Updates Needed

All notebooks reference paths like:
- `/Users/sarahurbut/aladynoulli2/...`
- `~/Dropbox/...`
- `../results/...`

**Options:**
1. **Keep absolute paths** (easiest, but less portable)
2. **Use relative paths** from repo root (better for sharing)
3. **Use config file** (`data/paths_config.py`) - best practice

---

## Questions to Decide

1. **Keep absolute paths or make relative?**
   - Absolute: Works immediately, but not portable
   - Relative: More portable, but needs path updates

2. **Include data files or just reference them?**
   - Reference: Smaller repo, but reviewers need data access
   - Include: Larger repo, but self-contained

3. **Git repo or just folder?**
   - Git: Version control, easier sharing
   - Folder: Simpler, but no version control

---

## Next Steps

1. Decide on path strategy (absolute vs relative vs config)
2. Create copy script
3. Run copy
4. Update paths
5. Test notebooks
6. Create final README

