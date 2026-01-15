#!/usr/bin/env python3
"""
Extract signature colors used in Figure3_Individual_Trajectories.ipynb
for signatures 0-20 (21 signatures total).

The Figure3 notebook uses: colors = sns.color_palette("tab20", K_total)
where K_total = 21 (signatures 0-20).

Output formats:
1. RGB tuples (0-1 range) - for matplotlib/seaborn
2. RGB tuples (0-255 range) - standard RGB
3. Hex codes - for web/graphics software
4. Matplotlib RGBA - for matplotlib with alpha
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

# Number of signatures (0-20 = 21 signatures)
K = 21

# Method 1: Using seaborn (as in Figure3)
colors_seaborn = sns.color_palette("tab20", K)

# Method 2: Using matplotlib tab20 colormap (alternative)
colors_matplotlib = cm.get_cmap('tab20')(np.linspace(0, 1, K))

print("="*80)
print("SIGNATURE COLORS FOR FIGURE3 (Signatures 0-20)")
print("="*80)
print(f"\nTotal signatures: {K}")
print("\nColors extracted using: sns.color_palette('tab20', 21)")
print("\n" + "="*80)

# Generate all formats
print("\nFORMAT 1: RGB TUPLES (0-1 range) - For matplotlib/seaborn")
print("-"*80)
print("colors = [")
for i, color in enumerate(colors_seaborn):
    rgb = (color[0], color[1], color[2])
    print(f"    {rgb},  # Signature {i}")
print("]")

print("\n" + "="*80)
print("\nFORMAT 2: RGB TUPLES (0-255 range) - Standard RGB")
print("-"*80)
print("colors_rgb255 = [")
for i, color in enumerate(colors_seaborn):
    rgb255 = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
    print(f"    {rgb255},  # Signature {i}")
print("]")

print("\n" + "="*80)
print("\nFORMAT 3: HEX CODES - For web/graphics software")
print("-"*80)
print("colors_hex = [")
for i, color in enumerate(colors_seaborn):
    hex_code = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
    print(f"    '{hex_code}',  # Signature {i}")
print("]")

print("\n" + "="*80)
print("\nFORMAT 4: HEX CODES (uppercase) - Alternative format")
print("-"*80)
print("colors_hex_upper = [")
for i, color in enumerate(colors_seaborn):
    hex_code = f"#{int(color[0]*255):02X}{int(color[1]*255):02X}{int(color[2]*255):02X}"
    print(f"    '{hex_code}',  # Signature {i}")
print("]")

print("\n" + "="*80)
print("\nFORMAT 5: DICTIONARY - Signature index to color")
print("-"*80)
print("signature_colors = {")
for i, color in enumerate(colors_seaborn):
    rgb = (color[0], color[1], color[2])
    hex_code = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
    print(f"    {i}: {{'rgb': {rgb}, 'hex': '{hex_code}'}},")
print("}")

print("\n" + "="*80)
print("\nFORMAT 6: MATPLOTLIB RGBA (0-1 range with alpha=1.0)")
print("-"*80)
print("colors_rgba = [")
for i, color in enumerate(colors_matplotlib):
    rgba = (color[0], color[1], color[2], 1.0)
    print(f"    {rgba},  # Signature {i}")
print("]")

# Also save to a Python file that can be imported
output_file = "signature_colors.py"
with open(output_file, 'w') as f:
    f.write('"""\n')
    f.write('Signature colors for Figure3 (signatures 0-20)\n')
    f.write('Extracted from seaborn tab20 palette\n')
    f.write('"""\n\n')
    f.write('import numpy as np\n\n')
    
    f.write('# RGB tuples (0-1 range) - for matplotlib/seaborn\n')
    f.write('SIGNATURE_COLORS_RGB = [\n')
    for i, color in enumerate(colors_seaborn):
        rgb = (color[0], color[1], color[2])
        f.write(f"    {rgb},  # Signature {i}\n")
    f.write(']\n\n')
    
    f.write('# Hex codes\n')
    f.write('SIGNATURE_COLORS_HEX = [\n')
    for i, color in enumerate(colors_seaborn):
        hex_code = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
        f.write(f"    '{hex_code}',  # Signature {i}\n")
    f.write(']\n\n')
    
    f.write('# Dictionary mapping signature index to hex color\n')
    f.write('SIGNATURE_COLORS_DICT = {\n')
    for i, color in enumerate(colors_seaborn):
        hex_code = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
        f.write(f"    {i}: '{hex_code}',\n")
    f.write('}\n')

print(f"\n✓ Also saved colors to: {output_file}")
print("   (You can import this file: from signature_colors import SIGNATURE_COLORS_RGB, SIGNATURE_COLORS_HEX)")
print("\n" + "="*80)
print("\nSUMMARY FOR FIGURE GURU:")
print("-"*80)
print("• Python accepts: RGB tuples (0-1 range), hex codes, or matplotlib colors")
print("• Most common format: RGB tuples (0-1 range) or hex codes")
print("• Seaborn/matplotlib prefer: RGB tuples (0-1 range)")
print("• Web/graphics software prefer: Hex codes")
print("="*80)



