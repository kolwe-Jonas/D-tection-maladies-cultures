"""Test de l'endpoint /detect avec une vraie image"""
import requests
import base64
import cv2
import numpy as np

print("\n" + "="*60)
print("🧪 TEST DU ENDPOINT /detect")
print("="*60 + "\n")

# Créer une image de test simple (100x100 pixels, verte)
test_image = np.ones((100, 100, 3), dtype=np.uint8)
test_image[:, :, 0] = 50   # Blue
test_image[:, :, 1] = 150  # Green
test_image[:, :, 2] = 50   # Red

# Encoder en JPEG
success, encoded_image = cv2.imencode('.jpg', test_image)
image_bytes = encoded_image.tobytes()
b64 = base64.b64encode(image_bytes).decode()

print(f"✓ Image de test créée: {test_image.shape}")
print(f"✓ Convertie en base64: {len(b64)} caractères\n")

# Test 1: Requête valide
print("Test 1️⃣ : Requête valide avec image base64")
print("-" * 60)
try:
    resp = requests.post('http://127.0.0.1:5000/detect', 
                        json={'image': b64},
                        timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}\n")
except Exception as e:
    print(f"❌ Erreur: {e}\n")

# Test 2: Requête sans image
print("Test 2️⃣ : Requête sans clé 'image'")
print("-" * 60)
try:
    resp = requests.post('http://127.0.0.1:5000/detect', 
                        json={'notimage': b64},
                        timeout=5)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}\n")
except Exception as e:
    print(f"❌ Erreur: {e}\n")

# Test 3: Requête avec mauvais Content-Type
print("Test 3️⃣ : Content-Type invalide")
print("-" * 60)
try:
    resp = requests.post('http://127.0.0.1:5000/detect', 
                        data={'image': b64},
                        timeout=5)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}\n")
except Exception as e:
    print(f"❌ Erreur: {e}\n")

# Test 4: Route / (interface web)
print("Test 4️⃣ : Accès à l'interface web (/)")
print("-" * 60)
try:
    resp = requests.get('http://127.0.0.1:5000/')
    print(f"Status: {resp.status_code}")
    print(f"HTML length: {len(resp.text)} caractères")
    print(f"Contains 'Détecteur': {'Détecteur' in resp.text}\n")
except Exception as e:
    print(f"❌ Erreur: {e}\n")

print("="*60)
print("✅ Tests terminés - Vérifiez les logs du serveur")
print("="*60 + "\n")
