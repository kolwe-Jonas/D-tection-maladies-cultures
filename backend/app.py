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

from services.image_analysis import analyze_leaf, optimize_image_for_analysis, compute_leaf_score, validate_plant_match
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
                os.remove(filepath)
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

        # ===== OPTIMISATION MÉMOIRE (resize + débruitage) =====
        try:
            image = optimize_image_for_analysis(image, max_dim=1024)
            logger.info(f"   ✓ Image optimisée: shape={image.shape}")
        except Exception as e:
            logger.error(f"   ❌ Erreur optimisation image: {str(e)}")
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({
                "error": "Erreur lors de l'optimisation de l'image",
                "details": str(e)
            }), 500

        # ===== TYPE DE PLANTE (extrait tôt pour la validation structurelle) =====
        plant_type_raw = request.form.get('plant_type', '').strip().lower()
        plant_type_map = {
            'mais': 'maïs',
            'manioc': 'manioc',
            'tomate': 'tomate',
            'riz': 'riz',
            'ble': 'blé',
            'mil': 'mil',
            'sorgho': 'sorgho',
            'coton': 'coton',
        }
        plant_type = plant_type_map.get(plant_type_raw, None)
        if plant_type_raw and plant_type is None:
            logger.warning(f"   ❌ plant_type invalide reçu: {plant_type_raw}")
            return jsonify({
                "error": "Valeur plant_type invalide",
                "plant_type_received": plant_type_raw
            }), 400

        # ===== VALIDATION FEUILLE (score global 0-100, 4 critères pondérés) =====
        # Règle : aucun critère seul ne rejette. Seul le score global décide.
        #   < 45  → rejet (pas une feuille)
        #   45-64 → feuille probable, analyse autorisée avec warning
        #   ≥ 65  → feuille valide
        try:
            leaf_check = compute_leaf_score(image, plant_type=plant_type)
            logger.info(
                f"   🌿 Score feuille: {leaf_check.get('leaf_score', 0):.1f}/100 "
                f"is_leaf={leaf_check['is_leaf']} "
                f"low_confidence={leaf_check.get('low_confidence_leaf', False)} "
                f"(couleur={leaf_check.get('color_score', 0):.1f}, "
                f"texture={leaf_check.get('texture_score', 0):.1f}, "
                f"forme={leaf_check.get('shape_score', 0):.1f}, "
                f"contours={leaf_check.get('contour_score', 0):.1f}, "
                f"veg={leaf_check.get('veg_percent', 0):.1f}%)"
            )
            if not leaf_check["is_leaf"]:
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                logger.warning(
                    f"   ❌ Image rejetée — score={leaf_check.get('leaf_score', 0):.1f}/100 "
                    f"— {leaf_check.get('reason', '')}"
                )
                if plant_type == "manioc":
                    error_msg = (
                        "❌ Image invalide : aucune feuille de manioc clairement structurée détectée. "
                        "Veuillez photographier une feuille complète avec ses lobes visibles."
                    )
                else:
                    error_msg = "❌ Aucune feuille de plante détectée. Veuillez prendre une photo claire d'une feuille centrée."
                return jsonify({
                    "success": False,
                    "is_leaf": False,
                    "error": error_msg,
                    "leaf_score": leaf_check.get("leaf_score", 0),
                    "color_score": leaf_check.get("color_score", 0),
                    "texture_score": leaf_check.get("texture_score", 0),
                    "shape_score": leaf_check.get("shape_score", 0),
                    "contour_score": leaf_check.get("contour_score", 0),
                    "veg_percent": leaf_check.get("veg_percent", 0),
                    "reason": leaf_check.get("reason", ""),
                }), 400

            # Feuille probable (45 ≤ score < 65) — analyse autorisée avec avertissement
            leaf_warning = None
            if leaf_check.get("low_confidence_leaf"):
                leaf_warning = (
                    "⚠️ Feuille détectée avec une confiance limitée "
                    "(possible mauvaise lumière, flou ou feuille partiellement visible). "
                    "Le résultat peut être moins précis."
                )
                logger.info(
                    f"   ⚠️ Feuille probable acceptée avec avertissement "
                    f"(score={leaf_check.get('leaf_score', 0):.1f}/100)"
                )

        except Exception as e:
            logger.error(f"   ❌ Validation feuille — erreur inattendue: {str(e)}", exc_info=True)
            try:
                os.remove(filepath)
            except Exception:
                pass
            return jsonify({
                "success": False,
                "error": "❌ Aucune feuille de plante détectée. Veuillez prendre une photo claire d'une feuille centrée.",
                "details": str(e),
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

        # ===== VALIDATION COHÉRENCE PLANTE (BLOQUANTE) =====
        if plant_type:
            try:
                plant_check = validate_plant_match(analysis, plant_type)
                logger.info(
                    f"   🌱 Validation plante: match={plant_check['plant_match']} "
                    f"confidence={plant_check['confidence']} reason={plant_check['reason']}"
                )
                if not plant_check["plant_match"]:
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
                    logger.warning(f"   ❌ Incohérence plante '{plant_type}' — {plant_check.get('reason')}")
                    return jsonify({
                        "success": False,
                        "is_leaf": True,
                        "plant_match": False,
                        "error": "❌ L'image ne correspond pas au type de plante sélectionné.",
                        "expected": plant_check.get("expected", ""),
                        "detected_shape": plant_check.get("detected_shape", ""),
                        "confidence": plant_check.get("confidence", 0),
                        "reason": plant_check.get("reason", ""),
                    }), 400
            except Exception as e:
                logger.warning(f"   ⚠️ Validation plante échouée (non bloquante): {str(e)}")

        logger.info("   ✅ Analyse complète - Préparation de la réponse JSON")

        # === Normaliser et garantir les champs de sortie demandés ===
        def _norm_text(v, default):
            if v is None:
                return default
            if isinstance(v, str):
                s = v.strip()
                if s == "" or s.lower() in ("unknown", "unknown disease", "incertain", "inconnue"):
                    return default
                return s
            # lists or other -> stringify safely
            if isinstance(v, (list, tuple)):
                if not v:
                    return default
                return ", ".join(str(x) for x in v)
            return str(v)

        # Defaults cohérents
        DEFAULTS = {
            "disease_name": "Maladie non identifiée",
            "symptoms": "Symptômes non disponibles.",
            "causes": "Causes non disponibles.",
            "treatment": "Traitement non spécifié.",
            "prevention": "Prévention non spécifiée.",
        }

        detection = match or {}

        # confidence normalization: prefer confidence_score (percent), else confidence (0-1)
        raw_conf = detection.get("confidence_score") or detection.get("confidence") or detection.get("confidence_pct")
        try:
            # If value between 0 and 1, convert to percent
            conf = float(raw_conf)
            if conf <= 1.0:
                conf = conf * 100.0
            confidence = round(max(0.0, min(100.0, conf)), 2)
        except Exception:
            confidence = 0.0

        # Ensure disease_name comes from DB and is not a placeholder
        disease_name = _norm_text(detection.get("disease_name") or detection.get("scientific_name"), DEFAULTS["disease_name"])

        # If somehow disease_name is still the generic default, try to salvage from analysis (plant_type hint)
        if disease_name == DEFAULTS["disease_name"]:
            alt = analysis.get("plant_type") or analysis.get("probable_plant")
            if alt:
                disease_name = f"Possible: {alt} (à confirmer)"

        # Textual fields
        symptoms = _norm_text(detection.get("symptoms"), DEFAULTS["symptoms"])
        causes = _norm_text(detection.get("causes"), DEFAULTS["causes"])
        treatment = _norm_text(detection.get("treatment"), DEFAULTS["treatment"])
        prevention = _norm_text(detection.get("prevention"), DEFAULTS["prevention"])

        # If confidence very low (<50 after normalization) prefer to still return top DB match
        # (find_best_match already returns a DB entry; keep match but flag low confidence)
        low_confidence = confidence < 50.0

        response_detection = {
            "disease_name": disease_name,
            "scientific_name": _norm_text(detection.get("scientific_name"), ""),
            "confidence": confidence,
            "confidence_score": confidence,
            "symptoms": symptoms,
            "causes": causes,
            "treatment": treatment,
            "prevention": prevention,
            # keep original detailed info for debugging/consumers that need it
            "_raw_detection": detection,
            "_low_confidence": low_confidence,
        }

        logger.info("   ✓ Réponse JSON normalisée prête (low_confidence=%s)", low_confidence)

        return jsonify({
            "success": True,
            "analysis": analysis,
            "detection": response_detection,
            "leaf_warning": leaf_warning,
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
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )