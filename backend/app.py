import os
import uuid
import base64
import io
import logging
import socket
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from flask_cors import CORS
from werkzeug.utils import secure_filename

from services.image_analysis import analyze_leaf
from services.disease_detector import find_best_match


# ===== CONFIGURATION =====
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def home():
    logger.info("✅ Route '/' appelée - retour de l'interface web")
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Vérification rapide : ouvrir http://<IP_PC>:5000/health depuis le téléphone."""
    return jsonify({
        "status": "ok",
        "message": "Serveur Flask actif — utiliser http:// (pas https://)",
        "detect_endpoint": "POST /detect (multipart, champ image)",
    }), 200


@app.route("/detect", methods=["POST"])
def detect():
    """
    Endpoint de détection de maladie
    Accepte: Formulaire multipart avec fichier 'image'
    Retourne: JSON avec analysis et detection
    """
    
    logger.info(f"📥 Requête /detect reçue - {request.remote_addr}")
    logger.info(f"   Content-Type: {request.content_type}")
    
    # ===== VALIDATION DU FICHIER =====
    try:
        # Vérifier que le fichier est présent
        if "image" not in request.files:
            logger.warning("   ❌ Clé 'image' manquante dans request.files")
            logger.warning(f"   Fichiers reçus: {list(request.files.keys())}")
            return jsonify({
                "error": "Aucun fichier image fourni",
                "received_files": list(request.files.keys())
            }), 400

        file = request.files["image"]
        logger.info(f"   ✓ Fichier reçu: {file.filename}")

        # Vérifier que le fichier n'est pas vide
        if file.filename == "":
            logger.warning("   ❌ Nom de fichier vide")
            return jsonify({
                "error": "Le fichier ne doit pas être vide"
            }), 400

        # Vérifier l'extension
        if not allowed_file(file.filename):
            logger.warning(f"   ❌ Extension non autorisée: {file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'N/A'}")
            return jsonify({
                "error": f"Format non autorisé. Acceptés: {', '.join(ALLOWED_EXT)}"
            }), 400

        logger.info(f"   ✓ Extension valide: {file.filename.rsplit('.', 1)[1].lower()}")

        # ===== SAUVEGARDE DU FICHIER =====
        try:
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(UPLOADS_DIR, unique_filename)
            file.save(filepath)
            logger.info(f"   ✓ Fichier sauvegardé: {unique_filename} ({os.path.getsize(filepath)} bytes)")
        except Exception as e:
            logger.error(f"   ❌ Erreur sauvegarde fichier: {str(e)}")
            return jsonify({
                "error": "Erreur lors de la sauvegarde du fichier",
                "details": str(e)
            }), 500

        # ===== CHARGEMENT IMAGE AVEC OPENCV =====
        try:
            image = cv2.imread(filepath)
            
            if image is None:
                logger.error(f"   ❌ cv2.imread a retourné None")
                os.remove(filepath)  # Nettoyer le fichier invalide
                return jsonify({
                    "error": "Image invalide ou corrompue",
                    "details": "Impossible de lire l'image avec OpenCV"
                }), 400
                
            logger.info(f"   ✓ Image chargée: shape={image.shape}, dtype={image.dtype}")
        except Exception as e:
            logger.error(f"   ❌ Erreur chargement image: {str(e)}")
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({
                "error": "Erreur lors du chargement de l'image",
                "details": str(e)
            }), 500

        # ===== RÉCEPTION DU TYPE DE PLANTE =====
        plant_type_raw = request.form.get('plant_type', '').strip().lower()
        plant_type_map = {
            'mais': 'maïs',
            'manioc': 'manioc',
            'tomate': 'tomate',
            'riz': 'riz',
        }
        plant_type = plant_type_map.get(plant_type_raw, None)
        if plant_type_raw and plant_type is None:
            logger.warning(f"   ❌ plant_type invalide reçu: {plant_type_raw}")
            return jsonify({
                "error": "Valeur plant_type invalide",
                "plant_type_received": plant_type_raw
            }), 400

        # ===== ANALYSE DE L'IMAGE =====
        try:
            logger.info("   🔍 Analyse de l'image en cours...")
            analysis = analyze_leaf(image)
            analysis['plant_type'] = plant_type
            logger.info("   ✓ Analyse complétée")
            logger.info("     - Couleur dominante: %s", analysis.get("dominant_color_name", "N/A"))
            logger.info("     - Zones affectées: %s%%", analysis.get("percent_unhealthy", "N/A"))
            logger.info("     - Gravité image: %s", analysis.get("severity", "N/A"))
        except Exception as e:
            logger.error(f"   ❌ Erreur analyse: {str(e)}", exc_info=True)
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({
                "error": "Erreur lors de l'analyse de l'image",
                "details": str(e)
            }), 500

        # ===== DÉTECTION MALADIE =====
        try:
            logger.info("   🔎 Recherche de la maladie...")
            match = find_best_match(analysis, plant_type=plant_type)
            logger.info(f"   ✓ Maladie détectée: {match.get('disease_name', 'Inconnue')}")
            logger.info(f"     - Confiance: {match.get('confidence_score', 'N/A')}%")
        except Exception as e:
            logger.error(f"   ❌ Erreur détection: {str(e)}", exc_info=True)
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({
                "error": "Erreur lors de la détection de la maladie",
                "details": str(e)
            }), 500

        logger.info(f"   ✅ Analyse complète - Réponse envoyée")
        
        return jsonify({
            "success": True,
            "analysis": analysis,
            "detection": match
        }), 200

    except Exception as e:
        logger.error(f"   ❌ Erreur générale non gérée: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Erreur serveur interne",
            "details": str(e)
        }), 500


def _get_local_ip() -> str:
    """Retourne l'IPv4 locale du PC (réseau Wi-Fi / Ethernet)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    local_ip = _get_local_ip()
    logger.info("=" * 60)
    logger.info("SERVEUR FLASK EN DEMARRAGE")
    logger.info("=" * 60)
    logger.info("Ecoute: 0.0.0.0:5000 (HTTP uniquement — pas HTTPS)")
    logger.info("PC (test):     http://127.0.0.1:5000")
    logger.info("Reseau (tel.): http://%s:5000", local_ip)
    logger.info("Test sante:    http://%s:5000/health", local_ip)
    logger.info("IMPORTANT: ne pas utiliser https:// sur le port 5000")
    logger.info("=" * 60)
    logger.info("SERVER RUNNING")
    logger.info("=" * 60)

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=True,
            use_reloader=True
        )
    except Exception as e:
        logger.error(f"Erreur au demarrage du serveur: {str(e)}", exc_info=True)