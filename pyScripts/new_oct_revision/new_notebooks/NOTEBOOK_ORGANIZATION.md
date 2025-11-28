# Performance Notebook Organization

## Current Structure (Updated Nov 27)

```
new_notebooks/
├── performancen_notebook_clean.ipynb  # ✅ ACTIVE - Use this (formerly _cf)
└── archive/
    ├── performancen_notebook.ipynb       # Original backup (Nov 25, 1.7M)
    └── performancen_notebook_clean_old.ipynb # Old cleaned version (Nov 26, 837K)
```

## Active Notebook

### `performancen_notebook_clean.ipynb` ✅
- **Current main notebook** (335K, 3,970 lines)
- Includes comparison scripts:
  - `%run compare_age_offset_washout` - Compares age offset vs washout AUCs
  - `%run verify_offset0_equivalence` - Verifies E matrices match
  - `%run compare_local_batch_files` - Compares local batch files
- All standard analysis scripts included
- **Use this for all new work**

## Archived Notebooks

### `archive/performancen_notebook.ipynb`
- Original with all outputs (Nov 25, 1.7M, 30,931 lines)
- Contains full execution history
- **Reference only** - don't edit

### `archive/performancen_notebook_clean_old.ipynb`
- Old cleaned version (Nov 26, 837K, 10,632 lines)
- Missing comparison scripts
- **Superseded** by current clean version

## Key Differences

The current `performancen_notebook_clean.ipynb` (formerly `_cf`) includes:
- ✅ All comparison scripts for debugging age offset/washout differences
- ✅ Streamlined structure (smaller file size)
- ✅ Most recent updates

The archived `performancen_notebook_clean_old.ipynb` was missing:
- ❌ `compare_age_offset_washout`
- ❌ `verify_offset0_equivalence`
- ❌ `compare_local_batch_files`

## What "cf" Likely Means

- **Conflict-Free**: Resolved merge conflicts, clean version
- **Comparison-Focused**: Includes comparison analysis scripts
- **Cleaned Final**: Final cleaned version
