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

# 3. Regenerate index page (preserving styled version)
echo ""
echo "Step 3: Regenerating index page (preserving styling)..."
echo ""

# Use the regenerate script to preserve styling
if [ -f "docs/regenerate_index_html.py" ]; then
    python3 docs/regenerate_index_html.py
    echo "✓ Index page regenerated with preserved styling"
else
    echo "⚠️  regenerate_index_html.py not found, skipping index regeneration"
    echo "   (index.html will remain unchanged)"
fi

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
      color: #212529;
      max-width: 1200px;
      margin: 0 auto;
      padding: 40px 20px;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .markdown-body {
      box-sizing: border-box;
      min-width: 200px;
      max-width: 1200px;
      margin: 0 auto;
      padding: 45px;
      background-color: #ffffff;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
      border-radius: 8px;
      border-top: 4px solid #0366d6;
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
      background: linear-gradient(135deg, #0366d6 0%, #0056b3 100%);
      color: #ffffff;
      font-weight: 600;
      text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .markdown-body table tr:nth-child(even) {
      background-color: #f8f9fa;
    }
    .markdown-body table tr:hover {
      background-color: #e3f2fd;
      transform: scale(1.01);
      transition: all 0.2s ease;
    }
    .markdown-body h1 {
      border-bottom: 4px solid #0366d6;
      padding-bottom: 0.3em;
      margin-top: 24px;
      margin-bottom: 20px;
      color: #0366d6;
      font-weight: 700;
      font-size: 2.2em;
    }
    .markdown-body h2 {
      border-bottom: 3px solid #28a745;
      padding-bottom: 0.3em;
      margin-top: 28px;
      margin-bottom: 16px;
      color: #24292e;
      font-weight: 600;
      font-size: 1.6em;
    }
    .markdown-body h3 {
      color: #586069;
      font-weight: 600;
      margin-top: 20px;
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
      font-weight: 500;
      border-bottom: 1px solid transparent;
      transition: all 0.2s ease;
    }
    .markdown-body a:hover {
      color: #0056b3;
      border-bottom: 1px solid #0056b3;
    }
    .markdown-body code {
      background-color: #f1f8ff;
      color: #e83e8c;
      border: 1px solid #c8e1ff;
      font-weight: 500;
    }
    .markdown-body pre {
      background-color: #f6f8fa;
      border-left: 4px solid #0366d6;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .markdown-body strong {
      color: #24292e;
      font-weight: 700;
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

