# Notebook Cell: Show Pathway Reproducibility

Run this cell in your notebook to show reproducibility between UKB and MGB:

```python
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from show_pathway_reproducibility import main

# Run reproducibility analysis (uses existing results by default)
results = main(force_rerun_mgb=False)
```

To force re-run MGB analysis:
```python
results = main(force_rerun_mgb=True)
```

This will:
1. Get pathway matches between UKB and MGB
2. Create a comprehensive reproducibility figure showing:
   - Pathway sizes for matched pathways
   - Pathway proportions correlation
   - Similarity scores
   - Age at onset comparison
   - Summary statistics
3. Print a reproducibility summary

The figure will be saved as `pathway_reproducibility_ukb_mgb.png`

