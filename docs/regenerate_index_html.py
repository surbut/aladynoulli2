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
    """Extract the style section from the current beautiful index.html"""
    index_html = Path('index.html')
    if not index_html.exists():
        raise FileNotFoundError("index.html not found. Please ensure the beautiful version exists first.")
    
    with open(index_html, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract the style section
    style_match = re.search(r'(<style>.*?</style>)', html_content, re.DOTALL)
    if not style_match:
        raise ValueError("Could not find <style> section in index.html")
    
    return style_match.group(1)

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

