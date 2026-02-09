#!/usr/bin/env python3
"""
Git filter to strip metadata from Jupyter notebooks.
This prevents metadata-only changes (timestamps, kernel info, etc.) from showing up in git diffs.
"""

import json
import sys

def strip_notebook_metadata(nb_content):
    """Remove metadata that changes frequently but doesn't affect notebook content."""
    try:
        nb = json.loads(nb_content)
        
        # Remove metadata from notebook level
        if 'metadata' in nb:
            # Keep only essential metadata
            essential_metadata = {}
            if 'kernelspec' in nb['metadata']:
                essential_metadata['kernelspec'] = nb['metadata']['kernelspec']
            if 'language_info' in nb['metadata']:
                essential_metadata['language_info'] = nb['metadata']['language_info']
            nb['metadata'] = essential_metadata
        
        # Remove execution-related metadata from cells
        if 'cells' in nb:
            for cell in nb['cells']:
                if 'metadata' in cell:
                    # Keep only essential cell metadata
                    cell_metadata = {}
                    if 'tags' in cell['metadata']:
                        cell_metadata['tags'] = cell['metadata']['tags']
                    if 'jupyter' in cell['metadata']:
                        cell_metadata['jupyter'] = cell['metadata']['jupyter']
                    cell['metadata'] = cell_metadata
                
                # Remove execution_count and other execution-related fields
                if 'execution_count' in cell:
                    cell['execution_count'] = None
                if 'outputs' in cell:
                    # Keep outputs but remove execution metadata
                    for output in cell['outputs']:
                        if 'execution_count' in output:
                            del output['execution_count']
        
        # Remove other frequently-changing fields
        for key in ['last_modified', 'last_run', 'last_executed']:
            if key in nb:
                del nb[key]
        
        return json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
    except Exception as e:
        # If parsing fails, return original content
        sys.stderr.write(f"Error processing notebook: {e}\n")
        return nb_content

if __name__ == '__main__':
    # Read from stdin
    content = sys.stdin.read()
    # Process and write to stdout
    sys.stdout.write(strip_notebook_metadata(content))
