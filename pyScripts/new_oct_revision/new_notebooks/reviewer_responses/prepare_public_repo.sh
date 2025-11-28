#!/bin/bash
# Script to prepare reviewer materials for a public repository
# This creates a sanitized version with relative paths

set -e

PUBLIC_DIR="../aladynoulli-reviewer-materials"
REVIEWER_DIR="$(pwd)"

echo "================================================================================"
echo "PREPARING PUBLIC REPOSITORY FOR REVIEWER MATERIALS"
echo "================================================================================"

# Create public directory
mkdir -p "$PUBLIC_DIR"
cd "$PUBLIC_DIR"

# Copy notebooks (will need path sanitization)
echo "Copying notebooks..."
mkdir -p notebooks
cp -r "$REVIEWER_DIR/notebooks"/*.ipynb notebooks/

# Copy knitted HTML outputs
echo "Copying HTML outputs..."
mkdir -p knitted_output
cp -r "$REVIEWER_DIR/knitted_output"/*.html knitted_output/

# Copy documentation
echo "Copying documentation..."
mkdir -p docs
cp -r "$REVIEWER_DIR/docs"/*.md docs/
cp "$REVIEWER_DIR/README.md" .
cp "$REVIEWER_DIR/REVIEWER_RESPONSE_MASTER.html" .

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data (exclude sensitive paths)
results/
data/
*.pt
*.pkl
*.h5
*.hdf5

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
venv/
env/
EOF

# Create README for public repo
cat > README.md << 'EOF'
# Aladynoulli: Reviewer Response Materials

This repository contains materials for responding to reviewer comments on the Aladynoulli manuscript.

## 📋 Contents

- **`notebooks/`**: Interactive Jupyter notebooks addressing reviewer questions
- **`knitted_output/`**: HTML versions of notebooks (ready to view)
- **`docs/`**: Supporting documentation
- **`REVIEWER_RESPONSE_MASTER.html`**: Master document with all responses

## 🚀 Quick Start

1. **View HTML outputs**: Open any `.html` file in `knitted_output/` in your browser
2. **Run notebooks**: Open `.ipynb` files in Jupyter (note: requires data access)
3. **Read documentation**: See `docs/README_FOR_REVIEWERS.md` for navigation guide

## 📝 Note on Paths

These notebooks reference results files that are not included in this repository due to size and privacy constraints. The notebooks are provided to demonstrate the analysis methodology. Results summaries are included in the HTML outputs.

## 🔗 Links

- Main repository: [Private - available upon request]
- Paper: [Link when published]

## 📧 Contact

For questions about these analyses, please contact [your email].

---

**Last Updated**: $(date +"%B %d, %Y")
EOF

echo ""
echo "================================================================================"
echo "PUBLIC REPOSITORY PREPARED"
echo "================================================================================"
echo "Location: $PUBLIC_DIR"
echo ""
echo "Next steps:"
echo "1. Review the files in $PUBLIC_DIR"
echo "2. Sanitize paths in notebooks (replace absolute paths with relative/generic)"
echo "3. Create GitHub repo: gh repo create aladynoulli-reviewer-materials --public"
echo "4. Push: cd $PUBLIC_DIR && git init && git add . && git commit -m 'Initial' && git push"
echo ""
echo "⚠️  IMPORTANT: Review notebooks for any sensitive paths before pushing!"


