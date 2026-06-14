#!/usr/bin/env python3
"""
Convert icon.png from mobile/assets to PWA-compatible PNG icons (192x192 and 512x512).
"""
import os
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Installing...")
    import subprocess
    subprocess.check_call([__import__('sys').executable, '-m', 'pip', 'install', 'Pillow'])
    from PIL import Image

# Paths
source_icon = Path('mobile/assets/icon.png')
icons_dir = Path('backend/static/icons')
icons_dir.mkdir(parents=True, exist_ok=True)

if not source_icon.exists():
    print(f"❌ Source icon not found: {source_icon}")
    exit(1)

# Load source icon
img = Image.open(source_icon)
print(f"📦 Loaded source icon: {img.size} {img.mode}")

# Convert to RGBA if necessary (for PNG compatibility)
if img.mode != 'RGBA':
    img = img.convert('RGBA')
    print(f"   Converted to RGBA")

# Create 192x192
icon_192 = img.resize((192, 192), Image.Resampling.LANCZOS)
path_192 = icons_dir / 'icon-192.png'
icon_192.save(path_192, 'PNG', quality=95)
print(f"✅ Created: {path_192} (192x192)")

# Create 512x512
icon_512 = img.resize((512, 512), Image.Resampling.LANCZOS)
path_512 = icons_dir / 'icon-512.png'
icon_512.save(path_512, 'PNG', quality=95)
print(f"✅ Created: {path_512} (512x512)")

print("\n✨ Icon conversion complete!")
