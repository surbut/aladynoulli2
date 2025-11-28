#!/usr/bin/env python3
"""
Generate a master HTML/PDF document from all reviewer question notebooks.

This script:
1. Reads all question notebooks
2. Extracts key sections (question, response, findings)
3. Generates a master HTML document
4. Can be converted to PDF using weasyprint or similar
"""

import json
import re
from pathlib import Path
from datetime import datetime

def extract_notebook_content(notebook_path):
    """Extract markdown and code cells from a notebook."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    content = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            content.append(('markdown', source))
        elif cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Only include if it's not just imports/setup
            if source.strip() and not source.strip().startswith('import'):
                content.append(('code', source))
    
    return content

def extract_question_and_response(notebook_path):
    """Extract reviewer question and response text from notebook."""
    content = extract_notebook_content(notebook_path)
    
    question = None
    response = None
    findings = []
    
    for cell_type, text in content:
        if cell_type == 'markdown':
            # Extract reviewer question
            if '## Reviewer Question' in text or '**Referee' in text:
                lines = text.split('\n')
                for line in lines:
                    if 'Referee' in line or 'Reviewer Question' in line:
                        question = line.strip()
                        break
            
            # Extract response
            if '### Response to Reviewer' in text or '> "We' in text:
                # Find the blockquote response
                match = re.search(r'>\s*"([^"]+)"', text, re.DOTALL)
                if match:
                    response = match.group(1)
            
            # Extract key findings
            if '## Key Findings' in text or '✅' in text:
                findings.append(text)
    
    return {
        'question': question,
        'response': response,
        'findings': '\n'.join(findings)
    }

def generate_html_master(notebooks_dir, output_path):
    """Generate master HTML document from all notebooks."""
    notebooks_dir = Path(notebooks_dir)
    
    # Find all question notebooks
    question_notebooks = sorted(notebooks_dir.glob('R*.ipynb'))
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Aladynoulli: Reviewer Response Document</title>
    <style>
        body {{
            font-family: 'Georgia', serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .question {{
            background-color: #ecf0f1;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
        .response {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #27ae60;
            margin: 20px 0;
        }}
        .findings {{
            background-color: #fff9e6;
            padding: 15px;
            border-left: 4px solid #f39c12;
            margin: 20px 0;
        }}
        .notebook-link {{
            color: #3498db;
            text-decoration: none;
        }}
        .notebook-link:hover {{
            text-decoration: underline;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <h1>Aladynoulli: Reviewer Response Document</h1>
    <p><em>Generated: {datetime.now().strftime('%B %d, %Y')}</em></p>
    
    <h2>Table of Contents</h2>
    <ul>
"""
    
    # Add TOC entries
    for nb_path in question_notebooks:
        nb_name = nb_path.stem
        if 'INDEX' not in nb_name:
            html += f'        <li><a href="#{nb_name}">{nb_name}</a></li>\n'
    
    html += """    </ul>
    
    <hr>
"""
    
    # Add content for each notebook
    for nb_path in question_notebooks:
        if 'INDEX' in nb_path.name:
            continue
            
        nb_name = nb_path.stem
        print(f"Processing {nb_name}...")
        
        try:
            info = extract_question_and_response(nb_path)
            
            html += f"""
    <section id="{nb_name}">
        <h2>{nb_name.replace('_', ' ')}</h2>
        
        <div class="question">
            <h3>Reviewer Question</h3>
            <p>{info['question'] or 'Question text extracted from notebook'}</p>
        </div>
        
        <div class="response">
            <h3>Our Response</h3>
            <p>{info['response'] or 'Response text extracted from notebook'}</p>
        </div>
        
        <div class="findings">
            <h3>Key Findings</h3>
            <div>{info['findings'] or 'See notebook for detailed findings'}</div>
        </div>
        
        <p><a href="notebooks/{nb_path.name}" class="notebook-link">→ Open interactive notebook</a></p>
        
        <hr>
    </section>
"""
        except Exception as e:
            print(f"Error processing {nb_name}: {e}")
            html += f"""
    <section id="{nb_name}">
        <h2>{nb_name}</h2>
        <p><em>Error loading notebook content. See <a href="notebooks/{nb_path.name}">notebook</a> directly.</em></p>
        <hr>
    </section>
"""
    
    html += """
    <footer>
        <p><em>For interactive analysis, see individual notebooks in the <code>notebooks/</code> directory.</em></p>
    </footer>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\n✅ Generated master HTML: {output_path}")

def main():
    notebooks_dir = Path(__file__).parent / 'notebooks'
    output_path = Path(__file__).parent / 'REVIEWER_RESPONSE_MASTER.html'
    
    print("="*80)
    print("GENERATING MASTER REVIEWER RESPONSE DOCUMENT")
    print("="*80)
    print(f"Notebooks directory: {notebooks_dir}")
    print(f"Output: {output_path}")
    print("="*80)
    
    generate_html_master(notebooks_dir, output_path)
    
    print(f"\n✅ Done! Open {output_path} in a browser.")
    print("\nTo convert to PDF:")
    print("  Option 1: Open in browser → Print → Save as PDF")
    print("  Option 2: Install weasyprint: pip install weasyprint")
    print("           Then: weasyprint REVIEWER_RESPONSE_MASTER.html REVIEWER_RESPONSE_MASTER.pdf")

if __name__ == '__main__':
    main()

