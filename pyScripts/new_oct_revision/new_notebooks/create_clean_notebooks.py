#!/usr/bin/env python3
"""
Create clean versions of notebooks for reviewers by hiding ancillary code/output.

This script:
1. Reads a notebook
2. Tags cells based on patterns (e.g., imports, data loading, intermediate outputs)
3. Creates a clean version with appropriate cells hidden
4. Can also create an HTML/PDF export with hidden cells removed

Usage:
    python create_clean_notebooks.py --notebook path/to/notebook.ipynb --output path/to/clean_notebook.ipynb
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any


def should_hide_cell(cell: Dict[str, Any], hide_patterns: List[str] = None):
    """
    Determine if a cell should have input/output hidden.
    
    Returns: (hide_input, hide_output)
    """
    if hide_patterns is None:
        hide_patterns = [
            r'^import\s+',  # Import statements
            r'^from\s+.*\s+import',  # From imports
            r'^#\s*Set\s+style',  # Style settings
            r'^sns\.set_style',  # Seaborn style
            r'^plt\.rcParams',  # Matplotlib params
            r'^print\(.*Loading',  # Loading messages
            r'^print\(.*Running',  # Running messages
            r'^print\(.*Skipping',  # Skipping messages
            r'^print\(.*Found',  # Found messages
            r'^print\(.*Warning',  # Warnings
            r'^print\(.*Error',  # Errors
            r'^print\(.*Processing',  # Processing messages
            r'^print\(.*Completed',  # Completion messages
            r'^print\(.*CALCULATING',  # Calculation headers
            r'^print\(.*RESULTS',  # Results headers
            r'^print\(.*='\s*\*',  # Separator lines
            r'^print\(f".*already exist',  # File exists messages
            r'^print\(f".*not found',  # File not found messages
        ]
    
    if cell['cell_type'] != 'code':
        return False, False
    
    source = ''.join(cell.get('source', []))
    
    # Check if it's a markdown cell with just a separator
    if cell['cell_type'] == 'markdown' and re.match(r'^---+\s*$', source.strip()):
        return False, True  # Hide output (there shouldn't be any)
    
    # Check patterns for hiding input
    hide_input = False
    for pattern in hide_patterns:
        if re.search(pattern, source, re.IGNORECASE | re.MULTILINE):
            hide_input = True
            break
    
    # Check if it's just setup/configuration code
    if re.match(r'^(import|from|# Set|sns\.|plt\.)', source.strip(), re.MULTILINE):
        hide_input = True
    
    # Check for verbose output patterns
    hide_output = False
    if 'outputs' in cell and cell['outputs']:
        for output in cell['outputs']:
            if 'text' in output:
                text = ''.join(output.get('text', []))
                # Hide verbose print statements
                if any(pattern in text.lower() for pattern in [
                    'loading', 'running', 'skipping', 'found', 'warning',
                    'processing', 'completed', 'calculating', 'results',
                    'already exist', 'not found'
                ]):
                    hide_output = True
                    break
    
    return hide_input, hide_output


def add_tags_to_notebook(notebook_path: Path, output_path: Path = None, 
                         auto_tag: bool = True, hide_input_tags: List[str] = None,
                         hide_output_tags: List[str] = None):
    """
    Add hide_input/hide_output tags to notebook cells.
    
    Args:
        notebook_path: Path to input notebook
        output_path: Path to output notebook (if None, overwrites input)
        auto_tag: Automatically tag cells based on patterns
        hide_input_tags: List of cell indices or patterns to hide input
        hide_output_tags: List of cell indices or patterns to hide output
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    if hide_input_tags is None:
        hide_input_tags = []
    if hide_output_tags is None:
        hide_output_tags = []
    
    for i, cell in enumerate(nb['cells']):
        # Initialize metadata if needed
        if 'metadata' not in cell:
            cell['metadata'] = {}
        if 'tags' not in cell['metadata']:
            cell['metadata']['tags'] = []
        
        # Auto-tag based on patterns
        if auto_tag:
            hide_input, hide_output = should_hide_cell(cell)
            if hide_input and 'hide_input' not in cell['metadata']['tags']:
                cell['metadata']['tags'].append('hide_input')
            if hide_output and 'hide_output' not in cell['metadata']['tags']:
                cell['metadata']['tags'].append('hide_output')
        
        # Manual tags
        if i in hide_input_tags or any(pattern in str(i) for pattern in hide_input_tags):
            if 'hide_input' not in cell['metadata']['tags']:
                cell['metadata']['tags'].append('hide_input')
        if i in hide_output_tags or any(pattern in str(i) for pattern in hide_output_tags):
            if 'hide_output' not in cell['metadata']['tags']:
                cell['metadata']['tags'].append('hide_output')
    
    output_path = output_path or notebook_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Tagged notebook saved to: {output_path}")
    print(f"  To view with hidden cells, use: jupyter nbconvert --to notebook --execute {output_path}")


def create_html_export(notebook_path: Path, output_html: Path, 
                       hide_tagged: bool = True):
    """
    Create HTML export with tagged cells removed.
    """
    import subprocess
    
    cmd = ['jupyter', 'nbconvert', '--to', 'html', 
           '--output', str(output_html.stem),
           '--output-dir', str(output_html.parent)]
    
    if hide_tagged:
        # Use a template that respects tags
        cmd.extend(['--template', 'classic'])  # or create custom template
    
    cmd.append(str(notebook_path))
    
    subprocess.run(cmd, check=True)
    print(f"✓ HTML export created: {output_html}")


def main():
    parser = argparse.ArgumentParser(description='Create clean notebook versions for reviewers')
    parser.add_argument('--notebook', type=str, required=True,
                       help='Path to input notebook')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output notebook (default: overwrites input with _clean suffix)')
    parser.add_argument('--html', type=str, default=None,
                       help='Also create HTML export at this path')
    parser.add_argument('--no-auto-tag', action='store_true',
                       help='Disable automatic tagging based on patterns')
    parser.add_argument('--hide-input', type=str, nargs='+', default=[],
                       help='Cell indices or patterns to hide input (e.g., "0,1,2" or "import")')
    parser.add_argument('--hide-output', type=str, nargs='+', default=[],
                       help='Cell indices or patterns to hide output')
    
    args = parser.parse_args()
    
    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        return
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = notebook_path.parent / f"{notebook_path.stem}_clean.ipynb"
    
    # Add tags
    add_tags_to_notebook(
        notebook_path, 
        output_path,
        auto_tag=not args.no_auto_tag,
        hide_input_tags=args.hide_input,
        hide_output_tags=args.hide_output
    )
    
    # Create HTML if requested
    if args.html:
        create_html_export(output_path, Path(args.html))


if __name__ == '__main__':
    main()

