#!/bin/bash
# Script to generate gamma association plots for both regularized and unregularized batches

SCRIPT_PATH="/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/main_paper_figures/generate_prs_signature_plots.py"
BASE_OUTPUT="/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig4"

echo "================================================================================"
echo "GENERATING GAMMA ASSOCIATIONS: REGULARIZED (with lambda_reg)"
echo "================================================================================"
python3 "$SCRIPT_PATH" \
    --batch_dir "/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/" \
    --pattern "enrollment_model_W0.0001_batch_*_*.pt" \
    --output_dir "${BASE_OUTPUT}/regularized" \
    --fdr_threshold 0.05 \
    --n_top 30

echo ""
echo "================================================================================"
echo "GENERATING GAMMA ASSOCIATIONS: UNREGULARIZED (nolr)"
echo "================================================================================"
python3 "$SCRIPT_PATH" \
    --batch_dir "/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr/" \
    --pattern "enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt" \
    --output_dir "${BASE_OUTPUT}/unregularized" \
    --fdr_threshold 0.05 \
    --n_top 30

echo ""
echo "================================================================================"
echo "COMPARISON COMPLETE"
echo "================================================================================"
echo "Regularized results: ${BASE_OUTPUT}/regularized/"
echo "Unregularized results: ${BASE_OUTPUT}/unregularized/"
echo ""
echo "To compare, check:"
echo "  - Regularized: ${BASE_OUTPUT}/regularized/gamma_associations.csv"
echo "  - Unregularized: ${BASE_OUTPUT}/unregularized/gamma_associations.csv"
