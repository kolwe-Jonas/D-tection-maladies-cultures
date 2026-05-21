"""Test final avec affichage des logs détaillés"""
import requests
import cv2
import numpy as np
import os
import tempfile

print("\n" + "="*80)
print("🎯 TEST FINAL - AFFICHAGE DES LOGS DÉTAILLÉS DU SERVEUR")
print("="*80)
print("\n📋 Vérifiez les logs du serveur Flask ci-dessous:\n")

# Créer une image de test
temp_dir = tempfile.gettempdir()
test_image = np.ones((100, 100, 3), dtype=np.uint8)
test_image[:, :, 0] = 50
test_image[:, :, 1] = 150
test_image[:, :, 2] = 50

temp_image_path = os.path.join(temp_dir, "final_test.jpg")
cv2.imwrite(temp_image_path, test_image)

print("="*80)
print("▶️ ENVOI D'UNE REQUÊTE VALIDE...")
print("="*80 + "\n")

try:
    with open(temp_image_path, 'rb') as f:
        files = {'image': f}
        resp = requests.post('http://127.0.0.1:5000/detect', files=files, timeout=10)
    
    print(f"\n{'='*80}")
    print("✅ RÉPONSE REÇUE")
    print(f"{'='*80}")
    print(f"Status Code: {resp.status_code}")
    print(f"\nRéponse JSON:")
    
    data = resp.json()
    import json
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
except Exception as e:
    print(f"\n❌ ERREUR: {e}")

# Nettoyer
try:
    os.remove(temp_image_path)
except:
    pass

print(f"\n{'='*80}")
print("✅ Test terminé")
print("="*80 + "\n")
