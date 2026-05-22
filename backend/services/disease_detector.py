"""
Détection de maladies par comparaison multi-critères.

Compare les caractéristiques extraites par image_analysis.analyze_leaf()
avec les profils dérivés des maladies en base (leaf_color, leaf_texture,
symptoms, severity).

Critères : couleur précise, texture, type/disposition/densité/taille
des taches, contours, gravité — avec pénalités anti-confusion
(taches localisées vs mosaïque/chlorose/rouille).
"""

import logging
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Pondération globale des critères (somme = 1.0)
CRITERION_WEIGHTS = {
        "db_profile": 0.22,
        "color_precise": 0.14,
        "texture": 0.08,
        "spot_type": 0.14,
        "spot_layout": 0.12,
        "spot_density": 0.08,
        "spot_size": 0.08,
        "contour": 0.06,
        "severity": 0.05,
        "veins": 0.03,
}

# Mapping type_taches (base) → profil interne
TYPE_TACHES_TO_PATTERN = {
        "taches rondes": "localized_spots",
        "taches rondes localisées": "localized_spots",
        "taches rondes concentriques": "localized_spots",
        "taches angulaires": "localized_spots",
        "taches angulaires localisées": "localized_spots",
        "taches en losange": "localized_spots",
        "taches ovales": "localized_spots",
        "taches irrégulières": "localized_spots",
        "taches irrégulières fusionnées": "blight",
        "taches allongées": "blight",
        "taches allongées fusionnées": "blight",
        "pustules": "rust",
        "pustules nombreuses": "rust",
        "mosaïque": "mosaic",
        "mosaïque diffuse": "mosaic",
        "mosaïque et rayures": "mosaic",
        "rayures et mosaïque": "mosaic",
        "rayures": "mosaic",
        "pas de taches": "chlorosis",
        "pas de taches, jaunissement": "chlorosis",
        "pas de taches, jaunissement uniforme": "chlorosis",
        "pas de taches foliaires": "wilt",
        "pas de taches nettes": "wilt",
        "pas de taches typiques": "chlorosis",
        "jaunissement uniforme": "chlorosis",
        "poudre blanche": "powdery",
        "poudre blanche sans taches": "powdery",
        "excroissances": "general",
        "excroissances/galles": "general",
        "galle": "general",
}

TAILLE_TACHES_RANGES = {
        "aucune": (0.0, 30.0),
        "très petite": (5.0, 150.0),
        "petite": (15.0, 400.0),
        "moyenne": (80.0, 900.0),
        "grande": (400.0, 5000.0),
        "moyenne à grande": (200.0, 3000.0),
        "petite à moyenne": (15.0, 900.0),
        "couvre la surface": (500.0, 8000.0),
}

PROGRESSION_TO_UNHEALTHY = {
        "lente": (3.0, 18.0),
        "lente à modérée": (8.0, 28.0),
        "modérée": (12.0, 35.0),
        "modérée à rapide": (18.0, 50.0),
        "progressive": (10.0, 40.0),
        "rapide": (25.0, 70.0),
        "rapide en conditions humides": (20.0, 65.0),
}


def _get_db_path() -> str:
        base = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(base, "diseases.db")


def _connect(db_path: str = None) -> sqlite3.Connection:
        if db_path is None:
                db_path = _get_db_path()
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn


def load_all_diseases(conn: sqlite3.Connection) -> List[Dict]:
        cur = conn.execute("SELECT * FROM diseases ORDER BY id")
        return [dict(r) for r in cur.fetchall()]


def _text_blob(disease: Dict) -> str:
        parts = [
                disease.get("disease_name", ""),
                disease.get("symptoms", ""),
                disease.get("leaf_color", ""),
                disease.get("leaf_texture", ""),
                disease.get("type_taches", ""),
                disease.get("couleur_taches", ""),
                disease.get("couleur_generale", ""),
                disease.get("texture_feuille", ""),
        ]
        return " ".join(parts).lower()


def _text_contains_any(text: str, words: List[str]) -> bool:
        if not text:
                return False
        t = text.lower()
        return any(w in t for w in words)


def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))


def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0


def _normalize_confidence_score(raw_percent: float) -> float:
        """Garantit un score affiché entre 50 % et 100 %.

        Si le score brut est < 50 % : confidence = 50 + (score_brut / 2).
        """
        raw = max(0.0, min(100.0, float(raw_percent)))
        if raw < 50.0:
                return round(50.0 + raw / 2.0, 2)
        return round(raw, 2)


def _compute_urgency(severity: str, pct_unhealthy: float, pattern: str) -> str:
        """Calcule le niveau d'urgence d'intervention agricole."""
        s = (severity or "").strip().lower()
        if s.startswith("sév") or s.startswith("sev") or pct_unhealthy >= 40.0:
                return "Immédiate — intervenir dans les 24h"
        if (s.startswith("élev") or s.startswith("elev")) or pct_unhealthy >= 22.0:
                return "Urgente — intervenir dans 2 à 3 jours"
        if s.startswith("mod") or pct_unhealthy >= 10.0:
                return "Modérée — surveiller et traiter sous 1 semaine"
        if pattern in ("wilt",):
                return "Urgente — flétrissement souvent irréversible"
        return "Faible — surveiller l'évolution"


def _compute_propagation_risk(pattern: str, pct_unhealthy: float, severity: str) -> str:
        """Évalue le risque de propagation vers d'autres plantes."""
        s = (severity or "").strip().lower()
        high_risk_patterns = {"rust", "mosaic", "blight", "powdery"}
        if pattern in high_risk_patterns or s.startswith("élev") or s.startswith("elev"):
                return "Élevé — isoler les plants atteints immédiatement"
        if pct_unhealthy >= 25.0 or s.startswith("mod"):
                return "Modéré — éviter les outils et contacts entre plants"
        return "Faible — surveiller les plants voisins"


def _disease_to_result(
        disease: Dict,
        confidence_pct: float,
        plant_type: Optional[str],
        plant_confidence: float,
        best_detail: Dict,
        image_pattern: str,
        score_diff: float,
        features: Optional[Dict] = None,
) -> Dict:
        """Construit la réponse finale avec tous les champs obligatoires."""
        pattern = best_detail.get("pattern", "general")
        severity = disease.get("severity") or ""
        pct_unhealthy = float((features or {}).get("percent_unhealthy", 0)) if features else 0.0
        urgency = _compute_urgency(severity, pct_unhealthy, pattern)
        propagation_risk = _compute_propagation_risk(pattern, pct_unhealthy, severity)
        return {
                "disease_name": disease.get("disease_name") or "Maladie non nommée",
                "scientific_name": disease.get("scientific_name") or "",
                "symptoms": disease.get("symptoms") or "Symptômes décrits en base de données.",
                "causes": disease.get("causes") or "Causes décrites en base de données.",
                "treatment": disease.get("treatment") or "Traitement décrit en base de données.",
                "prevention": disease.get("prevention") or "Prévention décrite en base de données.",
                "severity": severity,
                "urgency": urgency,
                "propagation_risk": propagation_risk,
                "plant_type": plant_type or disease.get("plant_type") or "",
                "plant_confidence_score": round(float(plant_confidence * 100.0), 2),
                "confidence_score": confidence_pct,
                "score_breakdown": best_detail.get("breakdown"),
                "match_reasons": best_detail.get("reasons"),
                "disease_pattern": best_detail.get("pattern"),
                "image_pattern": image_pattern,
                "score_difference": round(score_diff, 2),
        }


def _range_similarity(value: float, low: float, high: float, margin: float = 0.0) -> float:
        """Score 1.0 si value ∈ [low, high], décroît linéairement hors zone (+ marge)."""
        if low > high:
                low, high = high, low
        low -= margin
        high += margin
        if value < low:
                dist = low - value
                span = max(high - low, 1.0)
                return _clamp01(1.0 - dist / span)
        if value > high:
                dist = value - high
                span = max(high - low, 1.0)
                return _clamp01(1.0 - dist / span)
        return 1.0


def _has_descriptive_db(disease: Dict) -> bool:
        return bool((disease.get("type_taches") or "").strip())


def _pattern_from_type_taches(type_taches: str) -> Optional[str]:
        """Convertit le champ type_taches de la base en profil interne."""
        if not type_taches:
                return None
        t = type_taches.lower().strip()
        for key, pattern in TYPE_TACHES_TO_PATTERN.items():
                if key in t:
                        return pattern
        if "pustule" in t:
                return "rust"
        if "mosaïque" in t or "mosaique" in t:
                return "mosaic"
        if "tache" in t and "rond" in t:
                return "localized_spots"
        if "jaunissement" in t or "chlorose" in t:
                return "chlorosis"
        if "poudre" in t:
                return "powdery"
        if "flétr" in t or "fletri" in t:
                return "wilt"
        return None


def _token_overlap_score(db_text: str, observation_tokens: List[str]) -> float:
        """Score de recouvrement entre tokens de la base et observations."""
        if not db_text:
                return 0.5
        db_tokens = [w.strip() for w in db_text.lower().replace(",", " ").replace("/", " ").split() if len(w.strip()) > 2]
        if not db_tokens:
                return 0.5
        hits = sum(1 for tok in db_tokens if any(tok in obs for obs in observation_tokens))
        return _clamp01(hits / max(len(db_tokens), 1))


def _observation_tokens(features: Dict) -> List[str]:
        """Tokens décrivant l'image analysée (pour matching textuel)."""
        tokens = [
                (features.get("dominant_color_name") or "").lower(),
                (features.get("texture_label") or "").lower(),
                (features.get("spot_distribution") or "").lower(),
                (features.get("severity") or "").lower(),
        ]
        if float(features.get("percent_jaune_pale", 0)) > 12:
                tokens.extend(["jaune", "jaune pâle", "jaune pale"])
        if float(features.get("percent_brun_clair", 0)) > 8:
                tokens.extend(["brun", "brun clair"])
        if float(features.get("percent_brun_fonce", 0)) > 8:
                tokens.extend(["brun foncé", "brun fonce"])
        if float(features.get("percent_vert_clair", 0)) > 15:
                tokens.extend(["vert", "vert clair"])
        if float(features.get("percent_noir", 0)) > 5:
                tokens.append("noir")
        if float(features.get("spot_edge_percent", 0)) > 25:
                tokens.append("bords")
        if float(features.get("spot_center_percent", 0)) > 30:
                tokens.append("centre")
        if float(features.get("spot_count", 0)) >= 5:
                tokens.extend(["taches", "taches rondes"])
        if float(features.get("spot_mean_circularity", 0)) >= 0.5:
                tokens.append("rondes")
        return tokens


def _score_database_profile(features: Dict, disease: Dict, image_pattern: str) -> Tuple[float, Dict[str, float], List[str]]:
        """Compare les 8 champs descriptifs de la base avec l'image analysée."""
        if not _has_descriptive_db(disease):
                return 0.5, {"info": 0.5}, ["profil base : champs descriptifs absents"]

        sub: Dict[str, float] = {}
        reasons: List[str] = []
        obs_tokens = _observation_tokens(features)

        # 1. Type de taches
        type_taches = (disease.get("type_taches") or "").lower()
        db_pattern = _pattern_from_type_taches(type_taches)
        if db_pattern:
                sub["type_taches"] = 1.0 if db_pattern == image_pattern else 0.25 if db_pattern in ("general",) else 0.4
                if sub["type_taches"] >= 0.9:
                        reasons.append(f"type taches compatible ({type_taches})")
        elif type_taches:
                sub["type_taches"] = _token_overlap_score(type_taches, obs_tokens)

        # 2. Couleur des taches
        couleur_taches = disease.get("couleur_taches") or ""
        sub["couleur_taches"] = _token_overlap_score(couleur_taches, obs_tokens)

        # 3. Taille des taches
        taille = (disease.get("taille_taches") or "").lower()
        area = float(features.get("spot_mean_area", 0))
        matched_range = None
        for label, (lo, hi) in TAILLE_TACHES_RANGES.items():
                if label in taille:
                        matched_range = (lo, hi)
                        break
        if matched_range:
                sub["taille_taches"] = _range_similarity(area, *matched_range, margin=80.0)
        elif "aucune" in taille or "pas de" in taille:
                sub["taille_taches"] = 1.0 if area < 50 else 0.3
        else:
                sub["taille_taches"] = _token_overlap_score(taille, obs_tokens)

        # 4. Disposition
        disposition = disease.get("disposition_taches") or ""
        dist = (features.get("spot_distribution") or "").lower()
        disp_score = _token_overlap_score(disposition, obs_tokens + [dist])
        if "uniforme" in disposition.lower() and dist == "uniforme":
                disp_score = max(disp_score, 0.95)
        if "regroup" in disposition.lower() and dist in ("regroupée", "concentrée_bords", "concentrée_centre"):
                disp_score = max(disp_score, 0.9)
        if "bord" in disposition.lower() and float(features.get("spot_edge_percent", 0)) > 20:
                disp_score = max(disp_score, 0.85)
        sub["disposition_taches"] = disp_score

        # 5. Texture feuille
        texture_db = disease.get("texture_feuille") or disease.get("leaf_texture") or ""
        label = (features.get("texture_label") or "").lower()
        tex_map = {
                "lisse": ["lisse", "chlorose", "flétri"],
                "rugueuse": ["rugueuse", "moyenne", "déform"],
                "poudreuse": ["rugueuse", "moyenne"],
                "huileuse": ["moyenne", "rugueuse"],
                "nécrotique": ["moyenne", "rugueuse"],
                "flétrie": ["lisse", "moyenne"],
        }
        tex_score = _token_overlap_score(texture_db, [label] + obs_tokens)
        for key, allowed in tex_map.items():
                if key in texture_db.lower() and label in allowed:
                        tex_score = max(tex_score, 0.88)
        sub["texture_feuille"] = tex_score

        # 6. Couleur générale
        couleur_gen = disease.get("couleur_generale") or disease.get("leaf_color") or ""
        sub["couleur_generale"] = _token_overlap_score(couleur_gen, obs_tokens)

        # 7. Zones atteintes
        zones = disease.get("zones_atteintes") or ""
        zone_score = _token_overlap_score(zones, obs_tokens)
        if "nervure" in zones.lower() and float(features.get("vein_affected_percent", 0)) > 10:
                zone_score = max(zone_score, 0.85)
        if "feuille entière" in zones.lower() or "entière" in zones.lower():
                if float(features.get("percent_unhealthy", 0)) > 15:
                        zone_score = max(zone_score, 0.8)
        sub["zones_atteintes"] = zone_score

        # 8. Progression
        prog = (disease.get("progression_maladie") or "").lower()
        pct = float(features.get("percent_unhealthy", 0))
        prog_range = None
        for label, bounds in PROGRESSION_TO_UNHEALTHY.items():
                if label in prog:
                        prog_range = bounds
                        break
        if prog_range:
                sub["progression_maladie"] = _range_similarity(pct, *prog_range, margin=12.0)
        else:
                sub["progression_maladie"] = _token_overlap_score(prog, obs_tokens)

        if not sub:
                return 0.5, {"default": 0.5}, reasons

        avg = _mean(list(sub.values()))
        strong = [k for k, v in sub.items() if v >= 0.8]
        if strong:
                reasons.append(f"champs base forts : {', '.join(strong[:4])}")
        return avg, {k: round(v, 3) for k, v in sub.items()}, reasons


def _infer_disease_pattern(disease: Dict) -> str:
        """Déduit le profil visuel attendu (priorité aux champs descriptifs de la base)."""
        if _has_descriptive_db(disease):
                p = _pattern_from_type_taches(disease.get("type_taches", ""))
                if p:
                        return p

        text = _text_blob(disease)
        if disease.get("couleur_generale"):
                text += " " + (disease.get("couleur_generale") or "").lower()
        if disease.get("texture_feuille"):
                text += " " + (disease.get("texture_feuille") or "").lower()

        if _text_contains_any(text, ["mosaïque", "mosaique", "marbré", "marbre", "virus"]):
                return "mosaic"
        if _text_contains_any(text, ["chlorose", "jaunissement général", "carence", "jaune pâle"]):
                return "chlorosis"
        if _text_contains_any(text, ["rouille", "pustule", "orange"]):
                return "rust"
        if _text_contains_any(text, ["oïdium", "oidium", "poudre", "farineux", "poudreux"]):
                return "powdery"
        if _text_contains_any(text, ["flétr", "fletri", "fusario", "pourriture des racines"]):
                return "wilt"
        if _text_contains_any(text, ["cercospor", "alternario", "tache bact", "tache foliaire", "halo", "anneau", "concentrique"]):
                return "localized_spots"
        if _text_contains_any(text, ["mildiou", "duvet", "huileux"]):
                return "blight"
        if _text_contains_any(text, ["tache", "nécrot", "necrot", "brun foncé", "brun sombre"]):
                return "localized_spots"
        if _text_contains_any(text, ["jaune", "jaun"]):
                if _text_contains_any(text, ["tache", "spot", "pustule", "brun"]):
                        return "localized_spots"
                return "chlorosis"
        return "general"


def _build_expected_profile(pattern: str, disease: Dict) -> Dict[str, Tuple[float, float]]:
        """Plages attendues (min, max) par critère selon le profil de maladie."""
        severity = (disease.get("severity") or "").lower()
        if severity.startswith("faible"):
                unhealthy = (2.0, 18.0)
        elif severity.startswith("mod"):
                unhealthy = (12.0, 35.0)
        elif severity.startswith("élev") or severity.startswith("elev"):
                unhealthy = (25.0, 55.0)
        elif severity.startswith("sév") or severity.startswith("sev"):
                unhealthy = (40.0, 80.0)
        else:
                unhealthy = (8.0, 40.0)

        profiles = {
                "localized_spots": {
                        "percent_jaune_pale": (5.0, 30.0),
                        "percent_brun_clair": (8.0, 35.0),
                        "percent_brun_fonce": (5.0, 30.0),
                        "percent_vert_clair": (15.0, 60.0),
                        "percent_vert_fonce": (10.0, 50.0),
                        "percent_noir": (0.0, 12.0),
                        "spot_count": (3.0, 80.0),
                        "spot_mean_circularity": (0.45, 1.0),
                        "spot_mean_area": (15.0, 1200.0),
                        "spot_density": (5.0, 45.0),
                        "spot_edge_percent": (10.0, 70.0),
                        "spot_center_percent": (15.0, 75.0),
                        "spot_cluster_score": (20.0, 85.0),
                        "texture_roughness": (5.0, 35.0),
                        "percent_unhealthy": unhealthy,
                },
                "mosaic": {
                        "percent_jaune_pale": (12.0, 55.0),
                        "percent_vert_clair": (15.0, 50.0),
                        "percent_brun_clair": (0.0, 15.0),
                        "percent_brun_fonce": (0.0, 12.0),
                        "spot_count": (0.0, 8.0),
                        "spot_mean_circularity": (0.0, 0.45),
                        "spot_mean_area": (0.0, 200.0),
                        "spot_density": (3.0, 30.0),
                        "spot_spatial_uniformity": (50.0, 100.0),
                        "texture_roughness": (15.0, 60.0),
                        "percent_unhealthy": unhealthy,
                },
                "chlorosis": {
                        "percent_yellow": (20.0, 70.0),
                        "percent_jaune_pale": (18.0, 65.0),
                        "percent_brun_clair": (0.0, 12.0),
                        "percent_brun_fonce": (0.0, 10.0),
                        "spot_count": (0.0, 6.0),
                        "spot_mean_circularity": (0.0, 0.40),
                        "spot_density": (0.0, 18.0),
                        "texture_roughness": (0.0, 22.0),
                        "percent_unhealthy": (5.0, 35.0),
                },
                "rust": {
                        # Rouille : orange-brun dominant + micro-pustules nombreuses
                        "percent_brun_clair": (8.0, 50.0),
                        "percent_brun_fonce": (3.0, 25.0),
                        "percent_jaune_pale": (2.0, 30.0),
                        "spot_count": (5.0, 200.0),      # nombreuses micro-pustules
                        "spot_mean_circularity": (0.30, 0.95),  # pustules rondes à ovales
                        "spot_mean_area": (4.0, 600.0),  # micro-pustules (très petites)
                        "spot_median_area": (3.0, 300.0),
                        "spot_density": (8.0, 60.0),
                        "spot_edge_percent": (20.0, 95.0),  # souvent sur toute la surface
                        "spot_cluster_score": (30.0, 90.0),  # dispersées mais denses
                        "texture_roughness": (8.0, 55.0),
                        "percent_unhealthy": unhealthy,
                },
                "powdery": {
                        "percent_vert_clair": (20.0, 70.0),
                        "percent_brun_clair": (0.0, 10.0),
                        "spot_count": (0.0, 15.0),
                        "spot_density": (0.0, 20.0),
                        "texture_roughness": (8.0, 40.0),
                        "texture_variance": (50.0, 500.0),
                        "percent_unhealthy": (3.0, 25.0),
                },
                "wilt": {
                        "percent_yellow": (25.0, 65.0),
                        "percent_jaune_pale": (15.0, 50.0),
                        "percent_dry": (10.0, 45.0),
                        "spot_count": (0.0, 10.0),
                        "spot_density": (0.0, 22.0),
                        "texture_roughness": (0.0, 20.0),
                        "percent_unhealthy": (15.0, 50.0),
                },
                "blight": {
                        "percent_brun_fonce": (10.0, 45.0),
                        "percent_brun_clair": (5.0, 30.0),
                        "percent_unhealthy": (20.0, 60.0),
                        "spot_count": (2.0, 40.0),
                        "spot_density": (10.0, 50.0),
                        "texture_roughness": (10.0, 45.0),
                },
                "general": {
                        "percent_unhealthy": unhealthy,
                        "spot_count": (1.0, 60.0),
                        "spot_density": (3.0, 50.0),
                },
        }
        return profiles.get(pattern, profiles["general"])


def _detect_image_pattern(features: Dict) -> str:
        """Déduit le profil visuel observé sur l'image analysée."""
        spot_count = float(features.get("spot_count", 0))
        circularity = float(features.get("spot_mean_circularity", 0))
        spot_density = float(features.get("spot_density", 0))
        pct_yellow = float(features.get("percent_yellow", 0))
        pct_jaune_pale = float(features.get("percent_jaune_pale", 0))
        pct_brown = float(features.get("percent_brown", 0))
        pct_brun_clair = float(features.get("percent_brun_clair", 0))
        pct_brun_fonce = float(features.get("percent_brun_fonce", 0))
        uniformity = float(features.get("spot_spatial_uniformity", 0))
        distribution = (features.get("spot_distribution") or "").lower()
        pct_burned = float(features.get("percent_burned_edges", 0))
        pct_humid = float(features.get("percent_humid", 0))
        texture_roughness = float(features.get("texture_roughness", 0))
        pct_vert_clair = float(features.get("percent_vert_clair", 0))

        # Brûlures marginales : nécrose prononcée des bords
        if pct_burned >= 35.0 and (pct_brun_fonce + pct_brun_clair) >= 12.0:
                return "blight"

        # Oïdium / Poudre : texture rugueuse, peu de taches, vert clair dominant
        if texture_roughness >= 22.0 and spot_count <= 10 and pct_vert_clair >= 20.0 and pct_brown < 10.0:
                return "powdery"

        # Mildiou / aspect huileux : humidité + brun + taches fusionnées
        if pct_humid >= 15.0 and spot_density >= 8.0 and pct_brun_fonce >= 8.0:
                return "blight"

        # Taches localisées rondes (cercosporiose, alternariose, tache bactérienne…)
        if spot_count >= 3 and circularity >= 0.45 and spot_density >= 4.0:
                if pct_brown >= 5.0 or pct_brun_clair >= 5.0:
                        return "localized_spots"
        # Also: petit nombre de taches bien définies même si peu brunes
        if spot_count >= 5 and circularity >= 0.55 and spot_density >= 6.0:
                return "localized_spots"

        # Mosaïque : jaune/vert diffus, peu de taches distinctes
        if pct_jaune_pale >= 12.0 and spot_count <= 8 and circularity < 0.45:
                if uniformity >= 45.0 or distribution in ("uniforme", "concentrée_centre"):
                        return "mosaic"

        # Chlorose : jaunissement global, peu de brun/taches
        if pct_yellow >= 18.0 and pct_brown < 12.0 and spot_count <= 6:
                return "chlorosis"

        # Rouille : conditions STRICTES et SIMULTANÉES requises
        # Nombreuses petites pustules + couleur orange-brun + densité + bords couverts
        pct_brun_clair_r = float(features.get("percent_brun_clair", 0))
        pct_brun_fonce_r = float(features.get("percent_brun_fonce", 0))
        rust_color_signal = pct_brun_clair_r + pct_brun_fonce_r
        rust_circularity = float(features.get("spot_mean_circularity", 0))
        rust_edge_pct = float(features.get("spot_edge_percent", 0))
        # Toutes les conditions doivent être réunies (ET logique strict)
        rust_multi_cond = (
                spot_count >= 10              # nombreuses pustules
                and spot_density >= 12.0      # densité significative
                and rust_color_signal >= 12.0 # couleur brun/orange présente
                and rust_edge_pct >= 25.0     # distribution sur toute la surface
                and spot_count >= 8           # confirmation taches multiples
        )
        if rust_multi_cond:
                return "rust"

        # Flétrissement : jaunissement + sécheresse, peu de taches
        pct_dry = float(features.get("percent_dry", 0))
        if pct_dry >= 12.0 and pct_yellow >= 15.0 and spot_count <= 8:
                return "wilt"

        return "general"


def _score_texture(features: Dict, pattern: str) -> Tuple[float, Dict[str, float]]:
        label = (features.get("texture_label") or "").lower()
        roughness = float(features.get("texture_roughness", 0))
        variance = float(features.get("texture_variance", 0))
        pct_dry = float(features.get("percent_dry", 0))

        sub: Dict[str, float] = {}
        if pattern == "mosaic":
                sub["label"] = 1.0 if label in ("rugueuse", "moyenne") else 0.4 if label == "lisse" else 0.5
                sub["roughness"] = _range_similarity(roughness, 12.0, 55.0, margin=15.0)
        elif pattern == "chlorosis":
                sub["label"] = 1.0 if label in ("lisse", "moyenne") else 0.35
                sub["roughness"] = _range_similarity(roughness, 0.0, 25.0, margin=12.0)
        elif pattern == "localized_spots":
                sub["label"] = 0.85 if label in ("moyenne", "rugueuse") else 0.6
                sub["roughness"] = _range_similarity(roughness, 6.0, 40.0, margin=18.0)
        elif pattern == "rust":
                sub["roughness"] = _range_similarity(roughness, 10.0, 50.0, margin=15.0)
                sub["label"] = 0.75 if label != "lisse" else 0.45
        elif pattern == "wilt":
                sub["dry"] = _range_similarity(pct_dry, 8.0, 45.0, margin=15.0)
                sub["label"] = 1.0 if label == "lisse" else 0.5
        else:
                sub["roughness"] = _range_similarity(roughness, 5.0, 45.0, margin=20.0)
                sub["variance"] = _range_similarity(variance, 20.0, 800.0, margin=200.0)

        if not sub:
                sub["default"] = 0.5
        return _mean(list(sub.values())), sub


def _score_color_precise(features: Dict, expected: Dict[str, Tuple[float, float]]) -> Tuple[float, Dict[str, float]]:
        color_keys = [
                "percent_vert_clair", "percent_vert_fonce", "percent_jaune_pale",
                "percent_brun_clair", "percent_brun_fonce", "percent_noir",
                "percent_yellow", "percent_brown", "percent_necrotic",
        ]
        sub: Dict[str, float] = {}
        for key in color_keys:
                if key not in expected:
                        continue
                low, high = expected[key]
                value = float(features.get(key, 0.0))
                sub[key] = _range_similarity(value, low, high, margin=max(5.0, (high - low) * 0.35))

        if not sub:
                dom = (features.get("dominant_color_name") or "").lower()
                sub["dominant_fallback"] = 0.6 if dom else 0.3
        return _mean(list(sub.values())), sub


def _score_from_expected(features: Dict, expected: Dict, keys: List[str], margin_factor: float = 0.3) -> Tuple[float, Dict[str, float]]:
        sub: Dict[str, float] = {}
        for key in keys:
                if key not in expected:
                        continue
                low, high = expected[key]
                value = float(features.get(key, 0.0))
                margin = max(3.0, (high - low) * margin_factor)
                sub[key] = _range_similarity(value, low, high, margin=margin)
        if not sub:
                return 0.5, {"default": 0.5}
        return _mean(list(sub.values())), sub


def _score_spot_layout(features: Dict, pattern: str, expected: Dict) -> Tuple[float, Dict[str, float]]:
        keys = ["spot_edge_percent", "spot_center_percent", "spot_spatial_uniformity", "spot_cluster_score"]
        sub: Dict[str, float] = {}

        for key in keys:
                if key in expected:
                        low, high = expected[key]
                        sub[key] = _range_similarity(float(features.get(key, 0)), low, high, margin=max(8.0, (high - low) * 0.4))

        distribution = (features.get("spot_distribution") or "").lower()
        if pattern == "localized_spots":
                # Taches localisées : souvent regroupées ou sur les bords, pas un jaunissement uniforme
                sub["distribution"] = 1.0 if distribution in ("regroupée", "concentrée_bords", "concentrée_centre", "groupée") else 0.55 if distribution == "uniforme" else 0.7
        elif pattern == "mosaic":
                sub["distribution"] = 1.0 if distribution in ("uniforme", "concentrée_centre") else 0.4
        elif pattern == "chlorosis":
                sub["distribution"] = 1.0 if distribution in ("uniforme", "concentrée_centre") else 0.5
        elif pattern == "rust":
                sub["distribution"] = 1.0 if float(features.get("spot_edge_percent", 0)) >= 15.0 else 0.45

        if not sub:
                return 0.5, {"default": 0.5}
        return _mean(list(sub.values())), sub


def _anti_confusion_penalty(features: Dict, pattern: str, image_pattern: str) -> Tuple[float, List[str]]:
        """Pénalités pour éviter les confusions classiques entre profils."""
        penalty = 0.0
        reasons: List[str] = []

        spot_count = float(features.get("spot_count", 0))
        circularity = float(features.get("spot_mean_circularity", 0))
        pct_yellow = float(features.get("percent_yellow", 0))
        pct_brown = float(features.get("percent_brown", 0))
        pct_burned = float(features.get("percent_burned_edges", 0))
        pct_humid = float(features.get("percent_humid", 0))
        spot_density = float(features.get("spot_density", 0))
        spot_area = float(features.get("spot_mean_area", 0))
        texture_roughness = float(features.get("texture_roughness", 0))

        # --- Image = taches localisées, maladie = mosaïque/chlorose ---
        if image_pattern == "localized_spots" and pattern in ("mosaic", "chlorosis"):
                penalty += 0.35
                reasons.append(
                        f"pénalité : taches rondes (n={spot_count:.0f}, circ={circularity:.2f}) "
                        f"incompatible avec {pattern}"
                )

        # --- Image = mosaïque/chlorose diffuse, maladie = taches localisées ---
        if image_pattern in ("mosaic", "chlorosis") and pattern == "localized_spots":
                if spot_count <= 6 and circularity < 0.4:
                        penalty += 0.30
                        reasons.append(
                                f"pénalité : jaunissement diffus (jaune={pct_yellow:.1f}%) sans taches nettes"
                        )

        # --- Image = chlorose, maladie = rouille ---
        if image_pattern == "chlorosis" and pattern == "rust":
                if pct_brown < 10.0 and spot_count < 8:
                        penalty += 0.28
                        reasons.append("pénalité : peu de brun/taches, incompatible avec rouille")

        # --- Rouille : densité insuffisante ---
        if image_pattern == "localized_spots" and pattern == "rust":
                if spot_density < 8.0 and spot_count < 8:
                        penalty += 0.15
                        reasons.append("pénalité : densité faible pour rouille")

        # --- Mosaïque : taches circulaires nettes (plutôt cercosporiose) ---
        if pattern == "mosaic" and circularity >= 0.55 and spot_count >= 5:
                penalty += 0.25
                reasons.append("pénalité : taches circulaires nettes atypiques pour mosaïque")

        # --- Cercosporiose vs Alternariose : taille des taches ---
        # Cercosporiose = petites taches (< 200), Alternariose = grandes avec halos (> 300)
        if pattern == "localized_spots" and image_pattern == "localized_spots":
                db_name = ""  # handled below via feature signals
                if spot_area > 350.0 and texture_roughness < 10.0:
                        # Image montre grosses taches lisses → plus Alternariose que Cercosporiose
                        pass  # bonus géré par score

        # --- Oïdium / powdery : si image très brune, pas oïdium ---
        if pattern == "powdery" and pct_brown >= 20.0:
                penalty += 0.20
                reasons.append("pénalité : trop de brun pour oïdium/poudre blanche")

        # --- Mildiou / blight : si pas humide/huileux et peu nécrose ---
        if pattern == "blight" and image_pattern not in ("blight", "localized_spots"):
                if pct_humid < 5.0 and pct_burned < 10.0:
                        penalty += 0.18
                        reasons.append("pénalité : aucune humidité/brûlure typique du mildiou")

        # --- Flétrissement : si taches rondes bien présentes, pas flétrissement ---
        if pattern == "wilt" and spot_count >= 8 and circularity >= 0.5:
                penalty += 0.22
                reasons.append("pénalité : taches rondes multiples atypiques pour flétrissement")

        # --- Carence vs chlorose vs mosaïque ---
        if pattern == "chlorosis" and image_pattern == "mosaic":
                penalty += 0.18
                reasons.append("pénalité : mosaïque détectée, pas une chlorose uniforme")

        return penalty, reasons


def _severity_to_expected_unhealthy(severity: str) -> float:
        s = (severity or "").strip().lower()
        if s.startswith("faible"):
                return 8.0
        if s.startswith("mod"):
                return 25.0
        if s.startswith("élev") or s.startswith("elev"):
                return 55.0
        if s.startswith("sév") or s.startswith("sev"):
                return 75.0
        return 20.0


def _score_for_disease(features: Dict, disease: Dict) -> Tuple[float, Dict[str, object]]:
        """Calcule le score global (0-1), le détail par critère et les raisons."""
        pattern = _infer_disease_pattern(disease)
        expected = _build_expected_profile(pattern, disease)
        image_pattern = _detect_image_pattern(features)

        breakdown: Dict[str, Dict[str, object]] = {}
        reasons: List[str] = [f"profil maladie={pattern}"]

        # --- Profil descriptif base de données (8 champs) ---
        db_score, db_sub, db_reasons = _score_database_profile(features, disease, image_pattern)
        breakdown["db_profile"] = {"score": round(db_score, 3), "details": db_sub}
        reasons.extend(db_reasons)

        # --- Couleur précise ---
        color_score, color_sub = _score_color_precise(features, expected)
        breakdown["color_precise"] = {"score": round(color_score, 3), "details": {k: round(v, 3) for k, v in color_sub.items()}}

        # --- Texture ---
        texture_score, texture_sub = _score_texture(features, pattern)
        breakdown["texture"] = {"score": round(texture_score, 3), "details": {k: round(v, 3) for k, v in texture_sub.items()}}

        # --- Type de taches ---
        spot_type_keys = ["spot_count", "spot_mean_circularity", "spot_density"]
        spot_type_score, spot_type_sub = _score_from_expected(features, expected, spot_type_keys)
        breakdown["spot_type"] = {"score": round(spot_type_score, 3), "details": {k: round(v, 3) for k, v in spot_type_sub.items()}}

        # --- Disposition ---
        spot_layout_score, spot_layout_sub = _score_spot_layout(features, pattern, expected)
        breakdown["spot_layout"] = {"score": round(spot_layout_score, 3), "details": {k: round(v, 3) for k, v in spot_layout_sub.items()}}

        # --- Densité ---
        spot_density_score, spot_density_sub = _score_from_expected(features, expected, ["spot_density", "percent_unhealthy"])
        breakdown["spot_density"] = {"score": round(spot_density_score, 3), "details": {k: round(v, 3) for k, v in spot_density_sub.items()}}

        # --- Taille ---
        spot_size_score, spot_size_sub = _score_from_expected(features, expected, ["spot_mean_area", "spot_median_area", "spot_max_area"])
        breakdown["spot_size"] = {"score": round(spot_size_score, 3), "details": {k: round(v, 3) for k, v in spot_size_sub.items()}}

        # --- Contours ---
        contour_keys = ["spot_mean_circularity", "spot_contour_count", "spot_mean_perimeter"]
        contour_score, contour_sub = _score_from_expected(features, expected, contour_keys[:2])
        if "spot_mean_circularity" in expected:
                low, high = expected["spot_mean_circularity"]
                contour_sub["circularity"] = _range_similarity(float(features.get("spot_mean_circularity", 0)), low, high, margin=0.25)
        contour_score = _mean(list(contour_sub.values())) if contour_sub else 0.5
        breakdown["contour"] = {"score": round(contour_score, 3), "details": {k: round(v, 3) for k, v in contour_sub.items()}}

        # --- Gravité ---
        pct_unhealthy = float(features.get("percent_unhealthy", 0))
        expected_unhealthy = _severity_to_expected_unhealthy(disease.get("severity", ""))
        severity_score = _clamp01(1.0 - abs(pct_unhealthy - expected_unhealthy) / 60.0)
        breakdown["severity"] = {
                "score": round(severity_score, 3),
                "details": {"percent_unhealthy": pct_unhealthy, "expected": expected_unhealthy},
        }

        # --- Nervures ---
        vein_score = _range_similarity(
                float(features.get("vein_affected_percent", 0)),
                0.0, 40.0 if pattern in ("mosaic", "chlorosis") else 60.0,
                margin=20.0,
        )
        breakdown["veins"] = {
                "score": round(vein_score, 3),
                "details": {
                        "vein_affected_percent": float(features.get("vein_affected_percent", 0)),
                        "spot_on_veins_percent": float(features.get("spot_on_veins_percent", 0)),
                },
        }

        # --- Score de confiance rouille (rust_confidence_score) ---
        # La rouille ne peut bénéficier d'un bonus que si TOUTES les conditions
        # suivantes sont réunies simultanément. Une seule condition ne suffit pas.
        rust_confidence_score = 0.0
        rust_visual_bonus = 0.0
        if pattern == "rust":
                pct_bc   = float(features.get("percent_brun_clair", 0))
                pct_bf   = float(features.get("percent_brun_fonce", 0))
                s_count  = float(features.get("spot_count", 0))
                s_dens   = float(features.get("spot_density", 0))
                s_circ   = float(features.get("spot_mean_circularity", 0))
                s_area   = float(features.get("spot_mean_area", 9999))
                tex_r    = float(features.get("texture_roughness", 0))
                edge_pct = float(features.get("spot_edge_percent", 0))

                # Composantes du score de confiance (0-1 chacune)
                rust_color_score   = min(1.0, (pct_bc + pct_bf) / 30.0)        # couleur brun/orange
                rust_density_score = min(1.0, s_dens / 20.0)                   # densité pustules
                rust_texture_score = min(1.0, tex_r / 25.0)                    # texture poudreuse
                # Taille micro-pustules : score max si petites (30-400px), décroit sinon
                if s_area < 30:
                        rust_size_score = 0.3
                elif s_area <= 400:
                        rust_size_score = 1.0 - (s_area - 30) / 600.0
                else:
                        rust_size_score = max(0.0, 1.0 - (s_area - 400) / 1000.0)
                # Dispersion (nombreuses taches + couvrant la surface)
                rust_dispersion_score = min(1.0, s_count / 15.0) * min(1.0, edge_pct / 40.0)
                # Cohérence couleur (score existant comparé au profil DB)
                rust_coherence_score  = color_score

                # rust_confidence_score = moyenne pondérée des 6 composantes
                rust_confidence_score = (
                        rust_color_score     * 0.25
                        + rust_density_score * 0.25
                        + rust_texture_score * 0.15
                        + rust_size_score    * 0.15
                        + rust_dispersion_score * 0.10
                        + rust_coherence_score  * 0.10
                )

                # Debug logs (demandés)
                print(f"Rust confidence:  {rust_confidence_score:.3f}")
                print(f"Rust texture:     {rust_texture_score:.3f}")
                print(f"Rust color:       {rust_color_score:.3f}")
                print(f"Rust density:     {rust_density_score:.3f}")
                print(f"Final rust score: {rust_confidence_score:.3f}")

                # Bonus UNIQUEMENT si confiance forte ET toutes les conditions présentes
                # Seuil : rust_confidence_score > 0.55 (pas de bonus si score moyen)
                RUST_CONFIDENCE_THRESHOLD = 0.55
                if rust_confidence_score > RUST_CONFIDENCE_THRESHOLD:
                        # Bonus proportionnel à la confiance (max +0.08)
                        rust_visual_bonus = min(0.08, (rust_confidence_score - RUST_CONFIDENCE_THRESHOLD) * 0.36)
                        reasons.append(
                                f"confiance rouille: {rust_confidence_score:.0%} "
                                f"(couleur={rust_color_score:.0%}, densité={rust_density_score:.0%})"
                        )
                # Sinon : aucun bonus (le score brut doit suffire)

        # --- Score pondéré ---
        weighted = (
                CRITERION_WEIGHTS["db_profile"] * db_score
                + CRITERION_WEIGHTS["color_precise"] * color_score
                + CRITERION_WEIGHTS["texture"] * texture_score
                + CRITERION_WEIGHTS["spot_type"] * spot_type_score
                + CRITERION_WEIGHTS["spot_layout"] * spot_layout_score
                + CRITERION_WEIGHTS["spot_density"] * spot_density_score
                + CRITERION_WEIGHTS["spot_size"] * spot_size_score
                + CRITERION_WEIGHTS["contour"] * contour_score
                + CRITERION_WEIGHTS["severity"] * severity_score
                + CRITERION_WEIGHTS["veins"] * vein_score
        )
        weighted = _clamp01(weighted + rust_visual_bonus)

        penalty, penalty_reasons = _anti_confusion_penalty(features, pattern, image_pattern)
        final_score = _clamp01(weighted - penalty)

        if penalty > 0:
                reasons.extend(penalty_reasons)
        breakdown["penalty"] = {"value": round(penalty, 3), "reasons": penalty_reasons}
        breakdown["image_pattern"] = image_pattern
        breakdown["disease_pattern"] = pattern
        breakdown["weighted_raw"] = round(weighted, 3)
        breakdown["final"] = round(final_score, 3)

        # Raisons positives (critères forts)
        best_criteria = sorted(
                [(k, v["score"]) for k, v in breakdown.items() if isinstance(v, dict) and "score" in v],
                key=lambda x: x[1],
                reverse=True,
        )[:3]
        for crit, sc in best_criteria:
                if sc >= 0.75:
                        reasons.append(f"bon match {crit} ({sc:.0%})")

        return final_score, {"breakdown": breakdown, "reasons": reasons, "pattern": pattern}


def detect_plant_type(features: Dict) -> Tuple[str, float]:
        """Détecte le type de plante probable à partir des caractéristiques de forme."""
        aspect = float(features.get("leaf_aspect_ratio", 0.0))
        compactness = float(features.get("leaf_compactness", 0.0))
        percent_area = float(features.get("percent_leaf_area", 0.0))
        probable = (features.get("probable_plant") or "").strip()
        color = (features.get("dominant_color_name") or "").lower()

        score = 0.55
        plant = "tomate"
        reasons: List[str] = ["plante par défaut : tomate"]

        if probable and probable.lower() not in ("inconnu", "unknown plant", ""):
                plant = probable
                score = 0.82
                reasons = [f"analyse image : {probable}"]

        if aspect >= 1.8 and "vert" in color:
                plant, score, reasons = "maïs", 0.88, ["feuille très allongée et verte"]
        elif compactness > 0.72 and 0.75 <= aspect <= 1.35 and "vert" in color:
                plant, score, reasons = "tomate", 0.85, ["forme compacte et arrondie"]
        elif percent_area >= 12.0 and 0.9 <= aspect <= 2.0 and "vert" in color:
                plant, score, reasons = "manioc", 0.80, ["grand port et feuille large"]
        elif aspect >= 1.6 and 0.60 <= compactness <= 0.75:
                plant, score = "maïs", max(score, 0.68)
                reasons.append("allongée probable")
        elif 0.85 <= aspect <= 1.4 and compactness >= 0.65:
                plant, score = "tomate", max(score, 0.62)
                reasons.append("feuille ronde probable")

        plant_score = min(1.0, max(0.0, score))
        logger.info("Détection plante : %s (%.1f%%) — %s", plant, plant_score * 100, ", ".join(reasons))
        return plant, plant_score


def _log_scoring_details(scored: List[Tuple[Dict, float, Dict]], image_pattern: str) -> None:
        logger.info("Profil visuel image détecté : %s", image_pattern)
        logger.info("=== SCORES DE TOUTES LES MALADIES ===")
        for i, (disease, score, detail) in enumerate(scored, start=1):
                name = disease.get("disease_name", "?")
                plant = disease.get("plant_type", "?")
                pattern = detail.get("pattern", "?")
                logger.info("TOP %d: %s (%s) → %.1f%% | profil=%s", i, name, plant, score * 100, pattern)
                breakdown = detail.get("breakdown", {})
                for crit, data in breakdown.items():
                        if isinstance(data, dict) and "score" in data:
                                logger.info("       %-14s %.0f%%", crit, data["score"] * 100)
                                if crit == "db_profile" and isinstance(data.get("details"), dict):
                                        for field, val in data["details"].items():
                                                if isinstance(val, (int, float)):
                                                        logger.info("         · %-18s %.0f%%", field, float(val) * 100)
                for reason in detail.get("reasons", [])[:4]:
                        logger.info("       → %s", reason)


def _log_winner(best: Dict, best_score: float, detail: Dict, score_diff: float) -> None:
        logger.info("--- MALADIE RETENUE ---")
        logger.info(
                "%s (%.1f%%) | écart TOP1-TOP2: %.1f pts | profil=%s",
                best.get("disease_name"),
                best_score * 100,
                score_diff,
                detail.get("pattern"),
        )
        for reason in detail.get("reasons", []):
                logger.info("  Raison : %s", reason)
        bd = detail.get("breakdown", {})
        if bd.get("final"):
                logger.info("  Score final après pénalités : %.1f%%", bd["final"] * 100)


def _verify_rust_signals(features: Dict) -> bool:
        """Vérifie que l'image présente de vrais signaux de rouille biologiques.

        Conditions CUMULATIVES requises pour valider une prédiction rouille :
        - présence de végétation (couleurs biologiques)
        - couleurs orange/brun significatives
        - taches présentes (micro-pustules)
        - score plante suffisant (si disponible)

        Retourne False si l'image est clairement non végétale ou sans signaux rouille réels.
        """
        pct_brun_clair  = float(features.get("percent_brun_clair", 0.0))
        pct_brun_fonce  = float(features.get("percent_brun_fonce", 0.0))
        pct_vert_clair  = float(features.get("percent_vert_clair", 0.0))
        pct_vert_fonce  = float(features.get("percent_vert_fonce", 0.0))
        spot_count      = float(features.get("spot_count", 0.0))
        pct_unhealthy   = float(features.get("percent_unhealthy", 0.0))
        plant_conf      = float(features.get("_plant_confidence_score", 1.0))
        leaf_score      = float(features.get("_leaf_score", 100.0))

        # 1. Score plante trop bas → image non végétale → rouille impossible
        if plant_conf < 0.30 or leaf_score < 30.0:
                logger.warning(
                        "RUST GUARD: score plante trop bas (conf=%.2f, leaf=%.1f) — rouille bloquée",
                        plant_conf, leaf_score,
                )
                return False

        # 2. Aucune végétation → rouille impossible
        veg_total = pct_vert_clair + pct_vert_fonce + pct_brun_clair + pct_brun_fonce
        if veg_total < 4.0 and pct_unhealthy < 6.0:
                logger.warning(
                        "RUST GUARD: aucune végétation détectée (veg=%.1f%%) — rouille bloquée",
                        veg_total,
                )
                return False

        # 3. Aucune couleur orange/brun → pustules rouille impossibles
        rust_color = pct_brun_clair + pct_brun_fonce
        if rust_color < 2.0:
                logger.warning(
                        "RUST GUARD: aucune couleur brun/orange (%.1f%%) — rouille bloquée",
                        rust_color,
                )
                return False

        # 4. Aucune tache détectée → pas de pustules
        if spot_count < 2.0 and pct_unhealthy < 5.0:
                logger.warning(
                        "RUST GUARD: aucune tache/pustule (spots=%d, unhealthy=%.1f%%) — rouille bloquée",
                        int(spot_count), pct_unhealthy,
                )
                return False

        return True


def find_best_match(
        features: Dict,
        conn: sqlite3.Connection = None,
        plant_type: str = None,
) -> Dict:
        """Trouve la meilleure maladie correspondant aux caractéristiques analysées."""
        close_conn = False
        if conn is None:
                conn = _connect()
                close_conn = True

        plant_type_norm = (plant_type or "").strip().lower() or None
        detected_type = None
        plant_score = 0.0

        if plant_type_norm is None:
                detected_type, plant_score = detect_plant_type(features)
                plant_type_norm = (detected_type or "tomate").strip().lower()
        else:
                plant_score = 1.0
                detected_type = plant_type_norm

        diseases = load_all_diseases(conn)
        image_pattern = _detect_image_pattern(features)

        filtered = [
                d for d in diseases
                if (d.get("plant_type") or "").strip().lower() == plant_type_norm
        ]
        if filtered:
                logger.info("%d maladies filtrées pour plant_type=%s", len(filtered), plant_type_norm)
        else:
                logger.warning(
                        "Aucune maladie pour plant_type='%s' — recherche sur toutes les entrées (%d)",
                        plant_type_norm,
                        len(diseases),
                )
                filtered = diseases

        scored: List[Tuple[Dict, float, Dict]] = []
        for disease in filtered:
                score, detail = _score_for_disease(features, disease)
                scored.append((disease, score, detail))

        # ── Normalisation : empêche une maladie d'écraser toutes les autres ──
        # On applique une normalisation douce (softmax atténué) :
        # si l'écart TOP1-TOP2 est > 30 pts et TOP1 > 0.85, on ramène TOP1 à max 0.85
        # pour garantir que les autres maladies restent comparables.
        if scored:
                raw_scores = [s for _, s, _ in scored]
                max_raw = max(raw_scores)
                # Normalisation linéaire : on remet les scores dans [0, 1] relativement
                # mais on les plafonne à 0.95 pour éviter la domination écrasante
                if max_raw > 0.0:
                        scored_norm: List[Tuple[Dict, float, Dict]] = []
                        for disease_i, score_i, detail_i in scored:
                                # Si le score est très dominant, on l'atténue légèrement
                                if score_i >= 0.90 and max_raw >= 0.90:
                                        score_adj = 0.88 + (score_i - 0.90) * 0.4   # atténuation douce
                                else:
                                        score_adj = score_i
                                scored_norm.append((disease_i, _clamp01(score_adj), detail_i))
                        scored = scored_norm

        scored.sort(key=lambda item: item[1], reverse=True)

        # ── GARDE ANTI-FAUX-POSITIF ROUILLE ──────────────────────────────────
        # Si la maladie gagnante est "rouille" mais que les signaux biologiques
        # ne valident pas ce diagnostic → descendre sa note et re-trier.
        if scored:
                top_disease, top_score, top_detail = scored[0]
                if top_detail.get("pattern") == "rust":
                        if not _verify_rust_signals(features):
                                penalized_score = top_score * 0.40
                                logger.warning(
                                        "RUST GUARD: rouille pénalisée (%.1f%% → %.1f%%) — "
                                        "signaux biologiques insuffisants",
                                        top_score * 100, penalized_score * 100,
                                )
                                scored[0] = (top_disease, _clamp01(penalized_score), top_detail)
                                scored.sort(key=lambda item: item[1], reverse=True)

        # ── GARDE GLOBALE NON-VÉGÉTAL ─────────────────────────────────────────
        # Si le score plante passé dans les features est trop bas,
        # aucune maladie ne doit être retournée avec confiance élevée.
        plant_conf_score = float(features.get("_plant_confidence_score", 1.0))
        leaf_sc = float(features.get("_leaf_score", 100.0))
        if plant_conf_score < 0.30 or leaf_sc < 30.0:
                logger.warning(
                        "NON-VEGETAL GUARD: image non végétale dans find_best_match "
                        "(plant_conf=%.2f, leaf_score=%.1f) — confiance plafonnée à 50%%",
                        plant_conf_score, leaf_sc,
                )
                scored = [
                        (d, min(s, 0.45), det)
                        for d, s, det in scored
                ]

        # Debug log : tous les scores pour comparer
        disease_scores = {
                d.get("disease_name", "?"): round(s * 100, 1)
                for d, s, _ in scored
        }
        print(f"Other disease scores: {disease_scores}")

        _log_scoring_details(scored, image_pattern)

        if not scored:
                if close_conn:
                        conn.close()
                raise ValueError("Aucune maladie dans la base de données.")

        best, best_score, best_detail = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        score_diff = (best_score - second_score) * 100.0

        raw_confidence_pct = round(best_score * 100.0, 2)
        confidence_pct = _normalize_confidence_score(raw_confidence_pct)
        plant_confidence = round(plant_score * 100.0, 2)

        logger.info(
                "Score brut TOP1: %.1f%% → confidence affichée: %.1f%%",
                raw_confidence_pct,
                confidence_pct,
        )
        _log_winner(best, confidence_pct / 100.0, best_detail, score_diff)

        if close_conn:
                conn.close()

        result = _disease_to_result(
                best,
                confidence_pct,
                plant_type_norm or detected_type,
                plant_score,
                best_detail,
                image_pattern,
                score_diff,
                features=features,
        )
        logger.info(
                "Maladie finale choisie : %s (%.1f%%)",
                result["disease_name"],
                result["confidence_score"],
        )
        return result


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        # Taches localisées : petites taches rondes, bord brun, centre jaune
        localized_features = {
                "dominant_color_name": "vert",
                "percent_yellow": 12.0,
                "percent_brown": 18.0,
                "percent_unhealthy": 22.0,
                "percent_jaune_pale": 14.0,
                "percent_brun_clair": 16.0,
                "percent_brun_fonce": 8.0,
                "percent_vert_clair": 35.0,
                "percent_vert_fonce": 25.0,
                "spot_count": 15,
                "spot_mean_circularity": 0.72,
                "spot_mean_area": 120.0,
                "spot_density": 18.0,
                "spot_edge_percent": 35.0,
                "spot_center_percent": 40.0,
                "spot_distribution": "regroupée",
                "spot_cluster_score": 55.0,
                "spot_spatial_uniformity": 40.0,
                "texture_label": "moyenne",
                "texture_roughness": 18.0,
                "leaf_aspect_ratio": 1.1,
                "leaf_compactness": 0.72,
                "percent_leaf_area": 25.0,
                "probable_plant": "tomate",
        }

        print("\n--- Test taches localisées (Cercosporiose attendue) ---")
        match = find_best_match(localized_features, plant_type="tomate")
        print(f"Résultat : {match['disease_name']} ({match['confidence_score']}%)")
        print(f"Motifs : {match.get('match_reasons')}")
