# Guide d'Installation et Test de la PWA AgriDetect

## ✅ Checklist Pré-Installation (Étapes Requises)

### 1. Déploiement sur Render (HTTPS obligatoire)
```bash
# Assurez-vous que votre app est déployée sur Render
# URL doit être: https://votreapp.onrender.com
# (pas http://, HTTPS est obligatoire pour PWA)
```

### 2. Vérifier que les fichiers PWA sont accessibles

Via curl ou navigateur:
```
https://votreapp.onrender.com/static/manifest.json  → doit afficher JSON
https://votreapp.onrender.com/static/service-worker.js → doit afficher JavaScript
https://votreapp.onrender.com/static/icons/icon-192.png → doit afficher image PNG
https://votreapp.onrender.com/static/icons/icon-512.png → doit afficher image PNG
```

Tous doivent retourner **200 OK** (pas 404).

---

## 📱 Test sur Android (Chrome)

### Étape 1: Ouvrir le site et attendre le chargement
1. Ouvrir Chrome sur Android
2. Aller à `https://votreapp.onrender.com` (HTTPS obligatoire)
3. **Attendre 10-15 secondes** le chargement complet et enregistrement du SW

### Étape 2: Vérifier le Service Worker
1. Ouvrir DevTools Remote:
   - Sur PC, ouvrir Chrome → `chrome://inspect/#devices`
   - Connecter téléphone Android en USB
   - Sur téléphone, autoriser débogage USB
   - Sur PC, cliquer sur "inspect" pour ouvrir DevTools
2. Aller à onglet **Application**
3. Dans **Service Workers** → doit voir:
   - **Status: activated and running** (pas "installing" ni "waiting")
   - **Scope: /** (contrôle toute l'app)

### Étape 3: Vérifier le Manifest
1. Dans DevTools, onglet **Manifest**
2. Vérifier:
   - ✅ name: "AgriDetect"
   - ✅ start_url: "/"
   - ✅ display: "standalone"
   - ✅ theme_color: "#2e7d32"
   - ✅ Icons: 192x192 et 512x512 avec **status: ✓ OK**
   - ✅ scope: "/"

### Étape 4: Attendre le Prompt Installation
- Chrome affiche automatiquement une barre "Ajouter à l'écran d'accueil" → **cliquer dessus**
- OU le bouton "Installer l'application" apparaît sur la page → **cliquer dessus**

Si rien n'apparaît après 15 sec:
- Consulter la section **Dépannage** ci-dessous

### Étape 5: Installer l'App
1. Cliquer "Installer" (ou "Ajouter à l'écran d'accueil")
2. Attendre quelques secondes
3. L'app apparaît sur l'écran d'accueil avec l'icône **feuille verte**

### Étape 6: Lancer l'App
1. Cliquer sur l'icône "AgriDetect" sur écran d'accueil
2. L'app **s'ouvre en plein écran** (standalone) sans barre Chrome
3. Tester:
   - 📷 Caméra → doit ouvrir appareil photo
   - 🖼️ Galerie → doit ouvrir galerie photos
   - 🔍 Analyse → doit traiter l'image

---

## 🔧 Dépannage

### Problème: "Ajouter à l'écran d'accueil" n'apparaît pas

**Cause 1: Non-HTTPS**
- Vérifier URL commence par `https://`
- Chrome refuse PWA en HTTP

**Cause 2: Service Worker n'est pas activé**
- DevTools → Application → Service Workers
- Si "waiting" ou "installing" → forcer rafraîchir (Ctrl+Maj+R)
- Si erreur → vérifier console pour détails

**Cause 3: Manifest invalide**
- DevTools → Application → Manifest
- Doit montrer ✓ au lieu d'erreurs
- Vérifier `/static/manifest.json` est accessible

**Cause 4: Icons inaccessibles**
- DevTools → Network → filtrer "icon"
- Tous les icons doivent avoir status 200
- Pas de 404

### Problème: L'icône affichée est grise ou vide

**Cause: Format ou qualité PNG**
- Icon-192.png et icon-512.png doivent être des vraies images PNG
- Taille exacte: 192x192 et 512x512 pixels
- Vérifier: `file icon-192.png` → doit être "PNG image data"

**Solution:**
- Régénérer les icons avec le script Python:
```bash
python generate_icons.py
```

### Problème: "Ce site n'est pas installable"

**Cause: Fichier manifest manquant ou 404**
- Vérifier:
```bash
curl -v https://votreapp.onrender.com/static/manifest.json
```
- Doit retourner HTTP 200 et JSON valide

### Problème: App ouverte en mode navigateur (barre Chrome visible)

**Cause: display n'est pas "standalone"**
- Vérifier manifest.json contient:
```json
"display": "standalone"
```
- Désinstaller et réinstaller l'app

### Problème: Caméra ou upload ne fonctionne pas

**Cause 1: HTTPS requis pour caméra**
- Camera API demande HTTPS
- Vérifier URL est https://

**Cause 2: Permissions manquantes**
- Android → Paramètres → Applis → AgriDetect → Permissions
- Autoriser "Caméra" et "Stockage"

---

## 📊 Vérification complète (Console JavaScript)

Ouvrir **Console** (DevTools → Console) et vérifier les logs:

```
[PWA] Registering service worker from: /static/service-worker.js
[PWA] Service Worker registered successfully
[PWA] Scope: /
[PWA] Service Worker is ACTIVE and controlling this page
[PWA] beforeinstallprompt event fired - app is installable
[PWA] App is running in BROWSER mode (not installed yet)
```

Si vous voyez:
- ✅ Tous les logs → PWA fonctionne correctement
- ❌ Erreurs → consulter le message d'erreur

---

## 🚀 Après Installation

Une fois installée:

**En standalone:**
```
[PWA] App is running in STANDALONE mode (installed PWA)
```

**Tester offline:**
1. Éteindre internet
2. L'app doit rester accessible (cache SW)
3. Caméra et upload doivent fonctionner
4. API `/detect` va échouer (normal, pas de réseau)

---

## 📝 Notes Importantes

- **HTTPS obligatoire:** PWA ne fonctionne qu'en HTTPS (Render: automatique)
- **Service Worker scope: "/"** contrôle toute l'app
- **Icons maskable:** Android recadre en cercle/teardrop - design robuste
- **Manifest description:** Affiché lors de l'installation
- **Cache offline:** Service worker cache les assets, pas l'API

---

## 🐛 Si tout échoue: Diagnostic complet

1. **Console:**
   - DevTools → Console
   - Copier TOUS les logs `[PWA]`

2. **Manifest:**
   - DevTools → Application → Manifest
   - Copier le contenu complet

3. **Service Workers:**
   - DevTools → Application → Service Workers
   - Screenshot du statut

4. **Network:**
   - DevTools → Network
   - Recharger la page
   - Vérifier statuts de tous les /static/...

5. **Envoyer pour diagnostique:**
   - Screenshots DevTools
   - URL du site
   - Modèle téléphone Android
   - Version Chrome
