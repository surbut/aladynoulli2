#!/bin/bash
# Export model parameters for all three cohorts (UKB, AOU, MGB)
#
# UKB: master_for_fitting_pooled_correctedE.pt (pooled from censor_e_batchrun_vectorized batches)
# AOU: aou_model_master_correctedE.pt (pooled from aou_batches)
# MGB: mgb_model_initialized.pt

# Set base directory
BASE_DIR="/Users/sarahurbut/aladynoulli2"
EXPORT_BASE_DIR="${BASE_DIR}/exported_parameters"

# Try to activate conda environment if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Try to activate common conda environments
if command -v conda &> /dev/null; then
    if conda env list | grep -q "new_env_pyro2"; then
        echo "Activating conda environment: new_env_pyro2"
        conda activate new_env_pyro2
    elif conda env list | grep -q "aladyn"; then
        echo "Activating conda environment: aladyn"
        conda activate aladyn
    fi
fi

echo "=========================================="
echo "Exporting ALADYNOULLI Parameters"
echo "=========================================="
echo ""

# UK Biobank (Primary model)
# Created from: /Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*_*.pt
echo "1. Exporting UK Biobank parameters..."
UKB_CHECKPOINT="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt"
UKB_OUTPUT="${EXPORT_BASE_DIR}/ukb"

if [ -f "$UKB_CHECKPOINT" ]; then
    python export_model_parameters.py \
        --checkpoint "$UKB_CHECKPOINT" \
        --output_dir "$UKB_OUTPUT" \
        --prevalence "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/prevalence_t_corrected.pt" 2>&1 | tee "${UKB_OUTPUT}_export.log" || echo "⚠️  UKB export failed (check log: ${UKB_OUTPUT}_export.log)"
else
    echo "⚠️  UKB checkpoint not found: $UKB_CHECKPOINT"
fi

echo ""
echo "2. Exporting All of Us parameters (for cross-cohort validation)..."
AOU_CHECKPOINT="${BASE_DIR}/aou_model_master_correctedE.pt"
AOU_OUTPUT="${EXPORT_BASE_DIR}/aou"

if [ -f "$AOU_CHECKPOINT" ]; then
    python export_model_parameters.py \
        --checkpoint "$AOU_CHECKPOINT" \
        --output_dir "$AOU_OUTPUT" 2>&1 | tee "${AOU_OUTPUT}_export.log" || echo "⚠️  AOU export failed (check log: ${AOU_OUTPUT}_export.log)"
else
    echo "⚠️  AOU checkpoint not found: $AOU_CHECKPOINT"
fi

echo ""
echo "3. Exporting MGB parameters (for cross-cohort validation)..."
MGB_CHECKPOINT="${BASE_DIR}/mgb_model_initialized.pt"
MGB_OUTPUT="${EXPORT_BASE_DIR}/mgb"

if [ -f "$MGB_CHECKPOINT" ]; then
    python export_model_parameters.py \
        --checkpoint "$MGB_CHECKPOINT" \
        --output_dir "$MGB_OUTPUT" 2>&1 | tee "${MGB_OUTPUT}_export.log" || echo "⚠️  MGB export failed (check log: ${MGB_OUTPUT}_export.log)"
else
    echo "⚠️  MGB checkpoint not found: $MGB_CHECKPOINT"
fi

# Optional: Run verification plots
if [ "$1" == "--verify" ]; then
    echo ""
    echo "=========================================="
    echo "Running Verification Plots"
    echo "=========================================="
    
    # Verify UKB
    if [ -f "$UKB_CHECKPOINT" ] && [ -d "$UKB_OUTPUT" ]; then
        echo "Verifying UKB plots..."
        python verify_exported_parameters.py \
            --export_dir "$UKB_OUTPUT" \
            --clusters /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt \
            --cohort UKB 2>&1 | tee "${UKB_OUTPUT}_verification.log" || echo "⚠️  UKB verification failed"
    fi
    
    # Verify AOU
    if [ -f "$AOU_CHECKPOINT" ] && [ -d "$AOU_OUTPUT" ]; then
        echo "Verifying AOU plots..."
        python verify_exported_parameters.py \
            --export_dir "$AOU_OUTPUT" \
            --clusters /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam_AOU.pt \
            --cohort AOU 2>&1 | tee "${AOU_OUTPUT}_verification.log" || echo "⚠️  AOU verification failed"
    fi
    
    # Verify MGB
    if [ -f "$MGB_CHECKPOINT" ] && [ -d "$MGB_OUTPUT" ]; then
        echo "Verifying MGB plots..."
        python verify_exported_parameters.py \
            --export_dir "$MGB_OUTPUT" \
            --clusters /Users/sarahurbut/aladynoulli2/mgb_model_initialized.pt \
            --cohort MGB 2>&1 | tee "${MGB_OUTPUT}_verification.log" || echo "⚠️  MGB verification failed"
    fi
    
    echo ""
    echo "Verification plots saved in each export directory as: verification_signatures.pdf"
fi

echo ""
echo "=========================================="
echo "Export Complete"
echo "=========================================="
echo "Exported parameters are in: $EXPORT_BASE_DIR"
echo "  - UKB (primary): $UKB_OUTPUT"
echo "  - AOU (validation): $AOU_OUTPUT"
echo "  - MGB (validation): $MGB_OUTPUT"
echo ""
echo "Note: UKB checkpoint was created by pooling phi from:"
echo "  /Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*_*.pt"
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
echo "To verify exported parameters match original plots, run:"
echo ""
echo "  # For UKB:"
echo "  python verify_exported_parameters.py \\"
echo "    --export_dir $UKB_OUTPUT \\"
echo "    --clusters /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt \\"
echo "    --cohort UKB"
echo ""
echo "  # For AOU (clusters are in the checkpoint):"
echo "  python verify_exported_parameters.py \\"
echo "    --export_dir $AOU_OUTPUT \\"
echo "    --clusters /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam_AOU.pt \\"
echo "    --cohort AOU"
echo ""
echo "  # For MGB (clusters are in the checkpoint):"
echo "  python verify_exported_parameters.py \\"
echo "    --export_dir $MGB_OUTPUT \\"
echo "    --clusters /Users/sarahurbut/aladynoulli2/mgb_model_initialized.pt \\"
echo "    --cohort MGB"

