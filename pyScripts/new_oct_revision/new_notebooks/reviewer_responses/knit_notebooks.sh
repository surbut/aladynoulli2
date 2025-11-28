#!/bin/bash
# Convert all reviewer question notebooks to HTML/PDF using nbconvert
# This is the Python equivalent of "knitting" R Markdown

set -e

NOTEBOOKS_DIR="notebooks"
OUTPUT_DIR="knitted_output"

echo "================================================================================"
echo "CONVERTING NOTEBOOKS TO HTML/PDF (Python 'Knitting')"
echo "================================================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if nbconvert is available
if ! python -c "import nbconvert" 2>/dev/null; then
    echo "Installing nbconvert..."
    pip install nbconvert
fi

# Convert each notebook to HTML
echo ""
echo "Converting notebooks to HTML..."
for notebook in "$NOTEBOOKS_DIR"/R*.ipynb; do
    if [ -f "$notebook" ]; then
        basename=$(basename "$notebook" .ipynb)
        echo "  Converting $basename..."
        jupyter nbconvert --to html "$notebook" --output-dir="$OUTPUT_DIR" --output="${basename}.html"
    fi
done

# Convert index notebook
if [ -f "$NOTEBOOKS_DIR/REVIEWER_QUESTIONS_INDEX.ipynb" ]; then
    echo "  Converting REVIEWER_QUESTIONS_INDEX..."
    jupyter nbconvert --to html "$NOTEBOOKS_DIR/REVIEWER_QUESTIONS_INDEX.ipynb" --output-dir="$OUTPUT_DIR"
fi

# Optionally convert to PDF (requires pandoc and LaTeX)
echo ""
read -p "Convert to PDF? (requires pandoc/LaTeX) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Converting notebooks to PDF..."
    for notebook in "$NOTEBOOKS_DIR"/R*.ipynb; do
        if [ -f "$notebook" ]; then
            basename=$(basename "$notebook" .ipynb)
            echo "  Converting $basename to PDF..."
            jupyter nbconvert --to pdf "$notebook" --output-dir="$OUTPUT_DIR" --output="${basename}.pdf" 2>&1 | grep -v "WARNING" || echo "    (PDF conversion may require LaTeX)"
        fi
    done
fi

echo ""
echo "================================================================================"
echo "CONVERSION COMPLETE"
echo "================================================================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR"/*.html 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
ls -lh "$OUTPUT_DIR"/*.pdf 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (No PDF files - install LaTeX for PDF conversion)"

