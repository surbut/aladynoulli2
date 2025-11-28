# Re-Knit After AWS Age Offset Results

## When AWS Age Offset Predictions Complete

Once all AWS age offset batches finish running (`forAWS_offsetmasterfix.py`), you should:

1. **Verify results are available**:
   ```bash
   # Check that all batches are complete
   ls /path/to/age_offset/results/pi_enroll_fixedphi_age_offset_*_*.pt
   ```

2. **Re-knit the notebooks**:
   ```bash
   cd reviewer_responses/
   ./knit_notebooks.sh
   ```

3. **Regenerate master HTML** (optional):
   ```bash
   python generate_master_document.py
   ```

## Notebooks That May Need Updates

- **R2_Temporal_Leakage.ipynb**: May compare age offset vs washout results
- Any notebooks that reference age offset AUCs or predictions

## Quick Check

After AWS completes, verify results exist:
```bash
# Check age offset results directory
ls -lh /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pi_offset_using_pooled_retrospective_local/
```

Then re-knit to include updated results in the HTML/PDF outputs.

