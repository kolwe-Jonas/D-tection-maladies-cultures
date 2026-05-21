#!/usr/bin/env python
"""
Simple test to verify image_analysis and disease_detector work correctly.
"""

import sys
import os
import json
import base64
from io import BytesIO
from PIL import Image

# Add services to path
sys.path.insert(0, os.path.dirname(__file__))

from services.image_analysis import analyze_leaf
from services.disease_detector import find_best_match

def create_test_image():
    """Create a minimal test image (1x1 green pixel)."""
    img = Image.new('RGB', (100, 100), color=(0, 128, 0))  # Green
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()

def test_detection():
    """Test the full detection pipeline."""
    print("=" * 60)
    print("Simple Detection Pipeline Test")
    print("=" * 60)
    
    # Create test image
    print("\n[1] Creating test image...")
    image_bytes = create_test_image()
    print(f"    ✓ Image size: {len(image_bytes)} bytes")
    
    # Save to file
    print("\n[2] Saving test image...")
    test_path = "test_image.jpg"
    with open(test_path, "wb") as f:
        f.write(image_bytes)
    print(f"    ✓ Saved to {test_path}")
    
    # Analyze
    print("\n[3] Analyzing image...")
    try:
        analysis = analyze_leaf(test_path)
        print(f"    ✓ Analysis complete")
        print(f"       - dominant_color: {analysis.get('dominant_color_name')}")
        print(f"       - percent_yellow: {analysis.get('percent_yellow')}")
        print(f"       - percent_brown: {analysis.get('percent_brown')}")
        print(f"       - percent_unhealthy: {analysis.get('percent_unhealthy')}")
    except Exception as e:
        print(f"    ✗ Analysis failed: {e}")
        return False
    
    # Detect disease
    print("\n[4] Detecting disease...")
    try:
        detection = find_best_match(analysis)
        print(f"    ✓ Detection complete")
        print(f"       - disease: {detection.get('disease_name')}")
        print(f"       - confidence: {detection.get('confidence_score')}%")
        print(f"       - scientific: {detection.get('scientific_name')}")
    except Exception as e:
        print(f"    ✗ Detection failed: {e}")
        return False
    
    # Cleanup
    os.remove(test_path)
    
    print("\n" + "=" * 60)
    print("✓ All tests PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_detection()
    sys.exit(0 if success else 1)
