#!/bin/bash
# Script to update GitHub Pages with latest notebook HTML exports
# Usage: ./update_pages.sh

set -e

echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo "UPDATING GITHUB PAGES"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="

# 1. Re-render all notebooks to HTML
echo ""
echo "Step 1: Rendering notebooks to HTML..."
echo ""

find pyScripts/new_oct_revision/new_notebooks/reviewer_responses -name "*.ipynb" ! -path "*/archive/*" ! -name "index.ipynb" | while read notebook; do
    echo "  Processing: $(basename $notebook)"
    jupyter nbconvert --to html \
        --TagRemovePreprocessor.remove_input_tags='{"hide_input"}' \
        --TagRemovePreprocessor.remove_single_output_tags='{"hide_output"}' \
        "$notebook" > /dev/null 2>&1 || echo "    ⚠️  Warning: Failed to render $notebook"
done

echo "✓ All notebooks rendered"

# 2. Copy HTML files to docs/
echo ""
echo "Step 2: Copying HTML files to docs/..."
echo ""

# Clean docs/reviewer_responses
rm -rf docs/reviewer_responses
mkdir -p docs/reviewer_responses

# Copy all HTML files maintaining directory structure
find pyScripts/new_oct_revision/new_notebooks/reviewer_responses -name "*.html" -type f | while read html_file; do
    rel_path=${html_file#pyScripts/new_oct_revision/new_notebooks/reviewer_responses/}
    target_dir="docs/reviewer_responses/$(dirname "$rel_path")"
    mkdir -p "$target_dir"
    cp "$html_file" "docs/reviewer_responses/$rel_path"
    echo "  ✓ Copied: $rel_path"
done

echo "✓ All HTML files copied to docs/"

# 2.5. Remove HTML files from original notebook directories
echo ""
echo "Step 2.5: Removing HTML files from original notebook directories..."
echo ""

find pyScripts/new_oct_revision/new_notebooks/reviewer_responses -name "*.html" -type f -delete
echo "✓ HTML files removed from original directories"

# 3. Create index page
echo ""
echo "Step 3: Creating index page..."
echo ""

cat > docs/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Aladynoulli Reviewer Response Analyses</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; margin-top: 30px; }
        ul { list-style-type: none; padding-left: 0; }
        li { margin: 10px 0; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .section { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Aladynoulli: Reviewer Response Analyses</h1>
    <p>Interactive analyses addressing reviewer questions and concerns.</p>
    
    <div class="section">
        <h2>Referee #1 Analyses</h2>
        <ul>
            <li><a href="reviewer_responses/notebooks/R1/R1_Q1_Selection_Bias.html">Q1: Selection Bias</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Q2_Lifetime_Risk.html">Q2: Lifetime Risk</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Q3_Clinical_Meaning.html">Q3: Clinical Meaning</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Q3_ICD_vs_PheCode_Comparison.html">Q3: ICD vs PheCode Comparison</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Q7_Heritability.html">Q7: Heritability</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Q9_AUC_Comparisons.html">Q9: AUC Comparisons</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Q10_Age_Specific.html">Q10: Age-Specific Performance</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Biological_Plausibility_CHIP.html">Biological Plausibility: CHIP</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Clinical_Utility_Dynamic_Risk_Updating.html">Clinical Utility: Dynamic Risk Updating</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Genetic_Validation_GWAS.html">Genetic Validation: GWAS</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Multi_Disease_Patterns_Competing_Risks.html">Multi-Disease Patterns: Competing Risks</a></li>
            <li><a href="reviewer_responses/notebooks/R1/R1_Robustness_LOO_Validation.html">Robustness: Leave-One-Out Validation</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Referee #2 Analyses</h2>
        <ul>
            <li><a href="reviewer_responses/notebooks/R2/R2_R3_Model_Validity_Learning.html">Model Validity & Learning</a></li>
            <li><a href="reviewer_responses/notebooks/R2/R2_Temporal_Leakage.html">Temporal Leakage</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Referee #3 Analyses</h2>
        <ul>
            <li><a href="reviewer_responses/notebooks/R3/R3_Competing_Risks.html">Competing Risks</a></li>
            <li><a href="reviewer_responses/notebooks/R3/R3_Fixed_vs_Joint_Phi_Comparison.html">Fixed vs Joint Phi Comparison</a></li>
            <li><a href="reviewer_responses/notebooks/R3/R3_FullE_vs_ReducedE_Comparison.html">Full E vs Reduced E Comparison</a></li>
            <li><a href="reviewer_responses/notebooks/R3/R3_Linear_vs_NonLinear_Mixing.html">Linear vs Non-Linear Mixing</a></li>
            <li><a href="reviewer_responses/notebooks/R3/R3_Population_Stratification_Ancestry.html">Population Stratification: Ancestry</a></li>
            <li><a href="reviewer_responses/notebooks/R3/R3_Q8_Heterogeneity.html">Q8: Heterogeneity</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Framework Overview</h2>
        <ul>
            <li><a href="reviewer_responses/notebooks/framework/Discovery_Prediction_Framework_Overview.html">Discovery & Prediction Framework</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Preprocessing</h2>
        <ul>
            <li><a href="reviewer_responses/preprocessing/create_preprocessing_files.html">Create Preprocessing Files</a></li>
        </ul>
    </div>
</body>
</html>
EOF

echo "✓ Index page created"

# 4. Summary
echo ""
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo "COMPLETE!"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Commit: git add docs/ && git commit -m 'Update GitHub Pages'"
echo "  3. Push: git push"
echo ""
echo "After pushing, pages will be available at:"
echo "  https://yourusername.github.io/aladynoulli2/"
echo ""

