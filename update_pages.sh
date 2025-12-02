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
    <p><a href="reviewer_responses/README.md">📖 View full README with detailed navigation</a></p>
    
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
            <li><a href="reviewer_responses/notebooks/R2/R2_Washout_Continued.html">Washout Analysis: Continued</a></li>
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

# 3.5. Copy README.md to docs, update links to .html, and convert to HTML
echo ""
echo "Step 3.5: Copying README.md, updating links, and converting to HTML..."
echo ""

# First, copy README and convert .ipynb links to .html
sed 's/\.ipynb/\.html/g' pyScripts/new_oct_revision/new_notebooks/reviewer_responses/README.md > docs/reviewer_responses/README.md
echo "✓ README.md copied to docs/reviewer_responses/ (links updated to .html)"

# Convert README.md to HTML using pandoc (if available) or markdown
if command -v pandoc &> /dev/null; then
    pandoc docs/reviewer_responses/README.md -o docs/reviewer_responses/README.html \
        --standalone \
        --css=https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css \
        --metadata title="Reviewer Response Analyses" \
        --wrap=none \
        -V margin-top=0 \
        -V margin-bottom=0 \
        -V margin-left=0 \
        -V margin-right=0
    
    # Post-process to add better styling
    python3 << 'PYEOF'
import re
from pathlib import Path

html_file = Path('docs/reviewer_responses/README.html')
with open(html_file, 'r') as f:
    html = f.read()

# Add markdown-body wrapper and better styling
html = re.sub(
    r'<body>',
    r'''<body>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #24292e;
      max-width: 1200px;
      margin: 0 auto;
      padding: 40px 20px;
      background-color: #ffffff;
    }
    .markdown-body {
      box-sizing: border-box;
      min-width: 200px;
      max-width: 1200px;
      margin: 0 auto;
      padding: 45px;
      background-color: #ffffff;
    }
    .markdown-body table {
      border-spacing: 0;
      border-collapse: collapse;
      width: 100%;
      margin: 20px 0;
    }
    .markdown-body table th,
    .markdown-body table td {
      padding: 12px 15px;
      border: 1px solid #dfe2e5;
      text-align: left;
    }
    .markdown-body table th {
      background-color: #f6f8fa;
      font-weight: 600;
    }
    .markdown-body table tr:nth-child(even) {
      background-color: #f6f8fa;
    }
    .markdown-body h1 {
      border-bottom: 2px solid #eaecef;
      padding-bottom: 0.3em;
      margin-top: 24px;
      margin-bottom: 16px;
    }
    .markdown-body h2 {
      border-bottom: 1px solid #eaecef;
      padding-bottom: 0.3em;
      margin-top: 24px;
      margin-bottom: 16px;
    }
    .markdown-body code {
      background-color: rgba(27, 31, 35, 0.05);
      border-radius: 3px;
      font-size: 85%;
      padding: 0.2em 0.4em;
    }
    .markdown-body pre {
      background-color: #f6f8fa;
      border-radius: 6px;
      padding: 16px;
      overflow: auto;
    }
    .markdown-body a {
      color: #0366d6;
      text-decoration: none;
    }
    .markdown-body a:hover {
      text-decoration: underline;
    }
  </style>
  <article class="markdown-body">''',
    html
)

# Close the wrapper
html = re.sub(r'</body>', r'  </article>\n</body>', html)

with open(html_file, 'w') as f:
    f.write(html)
PYEOF
    echo "✓ README.md converted to HTML using pandoc with enhanced styling"
elif command -v markdown &> /dev/null; then
    markdown docs/reviewer_responses/README.md > docs/reviewer_responses/README.html
    echo "✓ README.md converted to HTML using markdown"
else
    # Fallback: Use Python markdown library
    python3 -c "
import markdown
from pathlib import Path

readme_md = Path('docs/reviewer_responses/README.md')
readme_html = Path('docs/reviewer_responses/README.html')

with open(readme_md, 'r') as f:
    md_content = f.read()

html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <title>Reviewer Response Analyses</title>
    <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css\">
    <style>
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 1200px;
            margin: 0 auto;
            padding: 45px;
        }}
    </style>
</head>
<body>
    <article class=\"markdown-body\">
{markdown.markdown(md_content, extensions=['tables', 'fenced_code'])}
    </article>
</body>
</html>'''

with open(readme_html, 'w') as f:
    f.write(html_content)
" 2>/dev/null && echo "✓ README.md converted to HTML using Python markdown" || echo "⚠️  Could not convert README.md to HTML (pandoc/markdown/Python markdown not available)"
fi

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

