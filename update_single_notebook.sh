#!/bin/bash
# Script to update a single notebook's HTML export
# Usage: ./update_single_notebook.sh path/to/notebook.ipynb

set -e

if [ -z "$1" ]; then
    echo "Usage: ./update_single_notebook.sh path/to/notebook.ipynb"
    exit 1
fi

NOTEBOOK="$1"
NOTEBOOK_NAME=$(basename "$NOTEBOOK" .ipynb)
NOTEBOOK_DIR=$(dirname "$NOTEBOOK")

echo "Converting $NOTEBOOK_NAME to HTML..."

# Convert notebook to HTML
jupyter nbconvert --to html \
    --TagRemovePreprocessor.remove_input_tags='{"hide_input"}' \
    --TagRemovePreprocessor.remove_single_output_tags='{"hide_output"}' \
    "$NOTEBOOK"

# Determine target directory in docs/
if [[ "$NOTEBOOK" == *"reviewer_responses"* ]]; then
    REL_PATH=${NOTEBOOK#*reviewer_responses/}
    TARGET_DIR="docs/reviewer_responses/$(dirname "$REL_PATH")"
    mkdir -p "$TARGET_DIR"
    cp "${NOTEBOOK%.ipynb}.html" "$TARGET_DIR/"
    echo "✓ Copied to: $TARGET_DIR/$NOTEBOOK_NAME.html"
    
    # Remove HTML from original location
    rm "${NOTEBOOK%.ipynb}.html"
    echo "✓ Removed HTML from original location"
else
    echo "⚠️  Notebook not in reviewer_responses directory, skipping copy to docs/"
fi

echo "✓ Done!"

