#!/bin/bash
# Script to copy all figure files to Overleaf figures folder

OVERLEAF_FIGURES="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/Apps/Overleaf/Aladynoulli_Nature/figures"
SOURCE_DIR="pyScripts/new_oct_revision/new_notebooks/results"

# Create figures directory if it doesn't exist
mkdir -p "$OVERLEAF_FIGURES"

# Copy files
echo "Copying figures to Overleaf..."

cp "$SOURCE_DIR/ancestry_analysis/signature_space_by_population.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/chip_multiple_signatures/ASCVD_sig5/CHIP_signature_trajectory.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/analysis/multi_disease_patterns_visualization.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/comparisons/plots/external_scores_comparison.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/analysis/plots/true_washout_comparison_10yr_30yr.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/analysis/plots/precursor_comparison_ASCVD.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/ancestry_analysis/pc1_pc2_signature_loadings.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/analysis/subsequent_disease_temporal_patterns.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/comparisons/plots/delphi_comparison.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/analysis/disease_progression_crosstab_matrices.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/analysis/top_disease_progressions_by_horizon.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/analysis/disease_cooccurrence_heatmap.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/analysis/plots/model_learning_full_comparison_ASCVD.png" "$OVERLEAF_FIGURES/"
cp "$SOURCE_DIR/analysis/plots/three_washout_types_diagram.png" "$OVERLEAF_FIGURES/"

echo "Done! Copied 14 figure files to $OVERLEAF_FIGURES"
ls -lh "$OVERLEAF_FIGURES"/*.png | wc -l
echo "files in Overleaf figures folder"
