"""Test de la nouvelle route /detect avec files"""
import requests
import cv2
import numpy as np
import os
import tempfile

print("\n" + "="*60)
print("🧪 TEST DE LA ROUTE /detect (MULTIPART FILES)")
print("="*60 + "\n")

# Utiliser le répertoire temporaire du système
temp_dir = tempfile.gettempdir()

# Créer une image de test simple (100x100 pixels)
test_image = np.ones((100, 100, 3), dtype=np.uint8)
test_image[:, :, 0] = 50   # Blue
test_image[:, :, 1] = 150  # Green
test_image[:, :, 2] = 50   # Red

# Sauvegarder temporairement l'image
temp_image_path = os.path.join(temp_dir, "test_image.jpg")
cv2.imwrite(temp_image_path, test_image)
print(f"✓ Image de test créée: {temp_image_path}")

# Test 1: Upload valide
print("\nTest 1️⃣ : Upload valide avec fichier image")
print("-" * 60)
try:
    with open(temp_image_path, 'rb') as f:
        files = {'image': f}
        resp = requests.post('http://127.0.0.1:5000/detect', files=files, timeout=10)
    
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Success: {data.get('success', False)}")
    if 'detection' in data:
        detection = data['detection']
        print(f"Disease: {detection.get('disease_name', 'N/A')}")
        print(f"Confidence: {detection.get('confidence_score', 'N/A')}%")
    print()
except Exception as e:
    print(f"❌ Erreur: {e}\n")

# Test 2: Sans fichier
print("Test 2️⃣ : Requête sans fichier")
print("-" * 60)
try:
    resp = requests.post('http://127.0.0.1:5000/detect', timeout=5)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}\n")
except Exception as e:
    print(f"❌ Erreur: {e}\n")

# Test 3: Fichier invalide (texte au lieu d'image)
print("Test 3️⃣ : Fichier invalide (texte)")
print("-" * 60)
try:
    temp_text_path = os.path.join(temp_dir, "test.txt")
    with open(temp_text_path, 'w') as f:
        f.write("Ceci n'est pas une image")
    
    with open(temp_text_path, 'rb') as f:
        files = {'image': f}
        resp = requests.post('http://127.0.0.1:5000/detect', files=files, timeout=5)
    
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}\n")
except Exception as e:
    print(f"❌ Erreur: {e}\n")

# Test 4: Mauvais nom de champ
print("Test 4️⃣ : Mauvais nom de champ")
print("-" * 60)
try:
    with open(temp_image_path, 'rb') as f:
        files = {'wrongfield': f}
        resp = requests.post('http://127.0.0.1:5000/detect', files=files, timeout=5)
    
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}\n")
except Exception as e:
    print(f"❌ Erreur: {e}\n")

# Nettoyer
try:
    os.remove(temp_image_path)
    if 'temp_text_path' in locals():
        os.remove(temp_text_path)
except:
    pass

print("="*60)
print("✅ Tests terminés - Vérifiez les logs du serveur")
print("="*60 + "\n")
