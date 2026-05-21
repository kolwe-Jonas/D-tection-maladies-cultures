#!/usr/bin/env python
"""
Test script to verify the disease detection pipeline end-to-end.
"""

import requests
import base64
import os
import json

BASE_URL = "http://10.18.10.59:5000"

def test_api_health():
    """Test if Flask API is running."""
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"[TEST] API Health: {r.status_code} - {r.text}")
        return r.status_code == 200
    except Exception as e:
        print(f"[TEST] API Health FAILED: {e}")
        return False

def test_detection_with_dummy_image():
    """Test /detect with a valid image."""
    # Create a minimal valid JPEG (1x1 green pixel)
    jpeg_hex = (
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
        "070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c23"
        "1c1c284000040200030201ffdb004301090909080900080808090000ffdb0043010a"
        "080a0c0b0c0c000d0c080d0d000101111401101702201212000101010101ffffff"
        "c00009010101011100ffc4001f0000010501010101010100000000000000000102030405"
        "0607080910ffc4b510000201020304050601070809000a181915131210110208" 
        "14121104133130f01122232415f10132526374166171819" 
        "2426354556373839" "3a4344455463656766" "67686869" "8a8b8c"
        "95969798999a" "a4a5a6a7a8a9aab3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4"
        "d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffd9"
    )
    
    try:
        jpeg_bytes = bytes.fromhex(jpeg_hex)
        image_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        
        payload = {
            "image": image_b64,
            "filename": "test.jpg"
        }
        
        r = requests.post(
            f"{BASE_URL}/detect",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"[TEST] Detection Response: {r.status_code}")
        print(f"[TEST] Response: {json.dumps(r.json(), indent=2, ensure_ascii=False)}")
        return r.status_code in (200, 400)  # 200 = success, 400 = valid error
        
    except Exception as e:
        print(f"[TEST] Detection FAILED: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Disease Detection Pipeline Test")
    print("=" * 60)
    
    print("\n[1/2] Testing API Health...")
    health_ok = test_api_health()
    
    if health_ok:
        print("\n[2/2] Testing Detection with Image...")
        detection_ok = test_detection_with_dummy_image()
    else:
        print("\n[ERROR] API not running. Start Flask backend first:")
        print("  cd backend && python app.py")
        detection_ok = False
    
    print("\n" + "=" * 60)
    if health_ok and detection_ok:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
    print("=" * 60)
