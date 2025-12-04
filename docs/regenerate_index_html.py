#!/usr/bin/env python3
"""
Script to regenerate index.html from index.md while preserving beautiful styling.

Usage:
    python regenerate_index_html.py

This script:
1. Reads index.md
2. Converts to HTML using pandoc
3. Wraps it in the beautiful template with all the custom CSS
4. Preserves the styling while updating content
"""

import subprocess
import re
from pathlib import Path

def get_template_styles():
    """Extract the style section from the current beautiful index.html, or use default"""
    index_html = Path('index.html')
    if index_html.exists():
        with open(index_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract the style section
        style_match = re.search(r'(<style>.*?</style>)', html_content, re.DOTALL)
        if style_match:
            return style_match.group(1)
    
    # Use default styling if index.html doesn't exist
    print("  ⚠️  index.html not found, using default styling...")
    return """<style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 15px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            font-size: 1.8em;
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }
        
        h3 {
            font-size: 1.4em;
            color: #555;
            margin-top: 25px;
            margin-bottom: 12px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
            transition: color 0.3s;
        }
        
        a:hover {
            color: #2980b9;
            text-decoration: underline;
        }
        
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
        }
        
        pre code {
            background: transparent;
            color: inherit;
            padding: 0;
        }
    </style>"""

def convert_markdown_to_html(md_file='index.md'):
    """Convert markdown to HTML body content using pandoc"""
    result = subprocess.run(
        ['pandoc', md_file, '-f', 'markdown', '-t', 'html', '--wrap=none'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Pandoc conversion failed: {result.stderr}")
    
    return result.stdout

def create_html_template(style_section, body_content):
    """Create the full HTML document with styling and content"""
    # Extract just the body content (remove any existing html/body tags from pandoc output)
    body_content = re.sub(r'^.*?<body[^>]*>', '', body_content, flags=re.DOTALL)
    body_content = re.sub(r'</body>.*?$', '', body_content, flags=re.DOTALL)
    body_content = re.sub(r'^.*?<html[^>]*>', '', body_content, flags=re.DOTALL)
    body_content = re.sub(r'</html>.*?$', '', body_content, flags=re.DOTALL)
    
    # Clean up any head sections
    body_content = re.sub(r'<head>.*?</head>', '', body_content, flags=re.DOTALL)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aladynoulli Documentation</title>
    {style_section}
</head>
<body>
    <div class="container">
        {body_content}
    </div>
</body>
</html>"""

def main():
    print("🔄 Regenerating index.html from index.md...")
    
    # Get the beautiful styling from current index.html
    print("  📋 Extracting styling from current index.html...")
    style_section = get_template_styles()
    
    # Convert markdown to HTML
    print("  📝 Converting index.md to HTML...")
    body_content = convert_markdown_to_html()
    
    # Create the full HTML document
    print("  🎨 Wrapping content with beautiful styling...")
    html_output = create_html_template(style_section, body_content)
    
    # Write to index.html
    output_file = Path('index.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"  ✅ Successfully regenerated {output_file}")
    print("  💡 Note: You may need to manually update table entries for new notebooks")

if __name__ == '__main__':
    main()

