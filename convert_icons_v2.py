#!/usr/bin/env python3
"""
Generate icon-192-v2.png and icon-512-v2.png from existing PNG icons.
Creates a visible border/background change to force Android/Chrome to treat them as new.
"""
from pathlib import Path
from PIL import Image, ImageDraw

icons_dir = Path('backend/static/icons')
icons_dir.mkdir(parents=True, exist_ok=True)

# Source priority: existing icon-512.png/icon-192.png, else mobile/assets/icon.png
src_candidates = [icons_dir / 'icon-512.png', icons_dir / 'icon-192.png', Path('mobile/assets/icon.png')]
source = None
for p in src_candidates:
    if p.exists():
        source = p
        break

if source is None:
    print('No source icon found. Expected one of:', src_candidates)
    raise SystemExit(1)

print('Using source:', source)

for size in (192, 512):
    src_path = icons_dir / f'icon-{size}.png'
    if not src_path.exists():
        # if specific size not exists, use general source
        src_path = source

    img = Image.open(src_path).convert('RGBA')
    # Ensure source is square by cropping centered
    w, h = img.size
    minside = min(w, h)
    left = (w - minside) // 2
    top = (h - minside) // 2
    img = img.crop((left, top, left + minside, top + minside))
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    # Create background with slightly different color to show change
    border_color = (6, 95, 70, 255)  # darker green
    canvas = Image.new('RGBA', (size, size), border_color)

    # Compute inner size to create visible border
    border_px = max(6, int(size * 0.06))
    inner_size = size - border_px * 2

    inner = img.resize((inner_size, inner_size), Image.Resampling.LANCZOS)
    offset = (border_px, border_px)

    # Paste inner onto canvas
    canvas.paste(inner, offset, inner)

    out_path = icons_dir / f'icon-{size}-v2.png'
    # Convert to RGB to remove alpha (avoid transparency issues)
    canvas_rgb = Image.new('RGB', canvas.size, (255, 255, 255))
    canvas_rgb.paste(canvas, mask=canvas.split()[3])
    canvas_rgb.save(out_path, format='PNG', optimize=True)
    print('Created', out_path)

print('Icon v2 generation complete.')
