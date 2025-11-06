#!/usr/bin/env python3
"""
Helper script to create new Jupyter notebooks

Usage:
    python create_notebook.py notebook_name.ipynb
    OR
    from create_notebook import create_notebook
    create_notebook('washout_analysis.ipynb', title='Washout Analysis')
"""

import json
import sys


def create_notebook(filename, title=None, initial_cells=None):
    """
    Create a new Jupyter notebook with optional title and initial cells
    
    Parameters:
    -----------
    filename : str
        Path to notebook file (should end in .ipynb)
    title : str, optional
        Title for the notebook (first markdown cell)
    initial_cells : list, optional
        List of dicts with 'cell_type' and 'source' keys
    """
    if not filename.endswith('.ipynb'):
        filename = filename + '.ipynb'
    
    # Default notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Add title cell if provided
    if title:
        title_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n"]
        }
        notebook["cells"].append(title_cell)
    
    # Add initial cells if provided
    if initial_cells:
        for cell in initial_cells:
            cell_dict = {
                "cell_type": cell.get("cell_type", "code"),
                "metadata": {},
                "source": cell.get("source", [])
            }
            if cell_dict["cell_type"] == "code":
                cell_dict["execution_count"] = None
                cell_dict["outputs"] = []
            notebook["cells"].append(cell_dict)
    
    # Add default setup cell if no initial cells
    if not initial_cells:
        setup_cell = {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Setup\n",
                "import sys\n",
                "import os\n",
                "sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import torch\n",
                "\n",
                "print('✅ Setup complete')"
            ]
        }
        notebook["cells"].append(setup_cell)
    
    # Write notebook
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Created notebook: {filename}")
    return filename


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_notebook.py notebook_name.ipynb [title]")
        sys.exit(1)
    
    filename = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else None
    
    create_notebook(filename, title=title)

