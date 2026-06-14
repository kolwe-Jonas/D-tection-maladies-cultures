#!/usr/bin/env python3
"""
Generate professional PWA icons: icon-192.png and icon-512.png
Creates real PNG files with proper leaf design, visible on Android.
"""
import struct
import zlib
from pathlib import Path
import math

def create_png_leaf(width, height):
    """
    Create a professional leaf/plant icon PNG.
    Returns raw PNG bytes.
    """
    # Create pixel data (RGB, no alpha channel for better compatibility)
    pixels = []
    
    for y in range(height):
        row = bytearray()
        for x in range(width):
            # Normalize coordinates to center
            nx = (x / width) * 2 - 1
            ny = (y / height) * 2 - 1
            
            # Create a leaf shape using multiple circles/ellipses
            # Leaf outline: elongated in Y direction
            leaf_dist = (nx * nx * 0.6) + (ny * ny * 1.2)
            
            # Inner leaf (lighter green)
            inner_color = (144, 238, 144)  # light green
            # Outer leaf (darker green for outline)
            outer_color = (34, 139, 34)    # darker green
            # Background (off-white)
            bg_color = (248, 249, 250)
            
            # Antialias the leaf boundary
            if leaf_dist < 0.5:
                # Solid leaf
                r, g, b = inner_color
            elif leaf_dist < 0.65:
                # Transition zone (blend)
                alpha = (0.65 - leaf_dist) / 0.15
                r = int(inner_color[0] * alpha + outer_color[0] * (1 - alpha))
                g = int(inner_color[1] * alpha + outer_color[1] * (1 - alpha))
                b = int(inner_color[2] * alpha + outer_color[2] * (1 - alpha))
            else:
                # Background
                r, g, b = bg_color
            
            row.extend([r, g, b])
        
        pixels.append(bytes(row))
    
    # Encode as PNG
    def chunk(chunk_type, data):
        """Create a PNG chunk."""
        crc = zlib.crc32(chunk_type + data) & 0xffffffff
        return struct.pack('>I', len(data)) + chunk_type + data + struct.pack('>I', crc)
    
    # PNG signature
    png = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk: 13 bytes
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)  # RGB, 8-bit
    png += chunk(b'IHDR', ihdr)
    
    # IDAT chunk: compressed image data
    # Each row starts with filter type 0 (None)
    idat_raw = b''
    for row in pixels:
        idat_raw += b'\x00' + row  # 0 = no filter
    
    idat_compressed = zlib.compress(idat_raw, level=9)
    png += chunk(b'IDAT', idat_compressed)
    
    # IEND chunk: empty chunk marks end
    png += chunk(b'IEND', b'')
    
    return png

# Create icons directory
icons_dir = Path('backend/static/icons')
icons_dir.mkdir(parents=True, exist_ok=True)

# Generate 192x192 icon
png_192 = create_png_leaf(192, 192)
path_192 = icons_dir / 'icon-192.png'
path_192.write_bytes(png_192)
print(f'✅ Created {path_192} ({len(png_192)} bytes)')

# Generate 512x512 icon
png_512 = create_png_leaf(512, 512)
path_512 = icons_dir / 'icon-512.png'
path_512.write_bytes(png_512)
print(f'✅ Created {path_512} ({len(png_512)} bytes)')

print('✨ Icon generation complete!')
