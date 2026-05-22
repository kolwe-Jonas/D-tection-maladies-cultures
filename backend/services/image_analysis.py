"""
Analyse d'images de feuilles avec OpenCV — extraction avancée de caractéristiques.

Techniques utilisées :
- Segmentation feuille/fond (HSV + seuillage adaptatif + composantes connexes)
- Détection de taches (morphologie, contours, connected components)
- Analyse de texture (variance locale, Laplacien, zones sèches/nécrotiques)
- Analyse spatiale (bords, centre, quadrants, nervures)
- Analyse couleur avancée (6 zones HSV)
- Heatmap des zones malades

Point d'entrée principal : analyze_leaf()
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Chargement et prétraitement
# ---------------------------------------------------------------------------

def load_image(path_or_array: object) -> np.ndarray:
        """Charge une image depuis un chemin ou accepte un tableau NumPy (BGR)."""
        if isinstance(path_or_array, str):
                img = cv2.imread(path_or_array)
                if img is None:
                        raise FileNotFoundError(f"Image introuvable : {path_or_array}")
                return img
        if isinstance(path_or_array, np.ndarray):
                return path_or_array.copy()
        raise TypeError("path_or_array doit être un chemin ou un numpy.ndarray")


def _ensure_small(img: np.ndarray, max_dim: int = 800) -> np.ndarray:
        """Réduit l'image si une dimension dépasse max_dim (conserve le ratio)."""
        h, w = img.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img


def optimize_image_for_analysis(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
        """Redimensionne et optimise une image pour l'analyse.
        Gère les photos haute résolution (téléphone, HD) et évite les crashs mémoire.
        - Limite la largeur/hauteur à max_dim (défaut 1024px)
        - Filtre bilatéral léger : réduit le bruit, préserve les contours de lésions
        """
        if img is None or not isinstance(img, np.ndarray):
                raise ValueError("Image invalide ou vide")
        h, w = img.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w, 1))
        if scale < 1.0:
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = cv2.bilateralFilter(img, d=5, sigmaColor=20, sigmaSpace=20)
        return img


def _morph_mask(
        mask: np.ndarray,
        kernel_size: Tuple[int, int] = (7, 7),
        close_iter: int = 2,
        open_iter: int = 1,
) -> np.ndarray:
        """Nettoie un masque binaire (fermeture puis ouverture morphologique)."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        return mask


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
        """Conserve uniquement la plus grande composante connexe du masque."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
                return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == largest_idx] = 255
        return out


# ---------------------------------------------------------------------------
# Segmentation feuille / fond
# ---------------------------------------------------------------------------

def _segment_leaf_precise(img: np.ndarray, hsv: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Segmentation multi-sources : HSV, canal vert, seuillage adaptatif."""
        h, w = img.shape[:2]
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Masque HSV : pixels végétaux probables
        mask_hsv = (
                ((sat > 35) & (val > 35))
                | ((hue >= 25) & (hue <= 95) & (sat > 30) & (val > 25))
                | ((hue >= 10) & (hue <= 55) & (sat > 20) & (val > 40))
        ).astype(np.uint8) * 255

        # Excess Green Index simplifié
        b, g, r = cv2.split(img.astype(np.float32))
        exg = 2.0 * g - r - b
        exg_norm = np.clip(((exg - exg.min()) / (exg.max() - exg.min() + 1e-6)) * 255, 0, 255).astype(np.uint8)
        _, mask_exg = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Seuillage adaptatif sur le canal vert
        green_ch = img[:, :, 1]
        adaptive = cv2.adaptiveThreshold(
                green_ch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 7
        )
        adaptive = cv2.bitwise_not(adaptive)

        # Fusion des masques
        combined = cv2.bitwise_or(mask_hsv, mask_exg)
        combined = cv2.bitwise_or(combined, adaptive)
        combined = _morph_mask(combined, kernel_size=(11, 11), close_iter=4, open_iter=2)
        combined = _keep_largest_component(combined)

        # Remplissage des trous internes
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
                largest = max(contours, key=cv2.contourArea)
                filled = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
                combined = filled

        combined = _morph_mask(combined, kernel_size=(7, 7), close_iter=2, open_iter=1)

        leaf_area = float(np.count_nonzero(combined))
        image_area = float(h * w)
        overlap_hsv = float(np.count_nonzero(cv2.bitwise_and(mask_hsv, combined)))
        overlap_exg = float(np.count_nonzero(cv2.bitwise_and(mask_exg, combined)))

        quality = {
                "leaf_segmentation_quality": round(100.0 * overlap_hsv / leaf_area, 2) if leaf_area else 0.0,
                "leaf_segmentation_overlap_percent": round(100.0 * overlap_exg / leaf_area, 2) if leaf_area else 0.0,
                "refined_leaf_area": leaf_area,
                "percent_leaf_area": round(100.0 * leaf_area / image_area, 2),
        }
        return combined, quality


# ---------------------------------------------------------------------------
# Couleur
# ---------------------------------------------------------------------------

def _map_hue_to_color_name(hue: int) -> str:
        """Convertit une teinte OpenCV (0-179) en nom lisible."""
        if hue < 10 or hue >= 170:
                return "rouge/rose"
        if 10 <= hue < 25:
                return "marron/orangé"
        if 25 <= hue < 40:
                return "jaune"
        if 40 <= hue < 85:
                return "vert"
        if 85 <= hue < 110:
                return "cyan/bleu-vert"
        if 110 <= hue < 170:
                return "bleu/violet"
        return "inconnu"


def _color_zone_percentages(
        hue: np.ndarray, sat: np.ndarray, val: np.ndarray, leaf_bool: np.ndarray
) -> Dict[str, float]:
        """Répartit les pixels de la feuille en 6 zones couleur avancées."""
        zones = {
                "vert_clair": ((hue >= 35) & (hue <= 85) & (sat > 45) & (val > 110)),
                "vert_fonce": ((hue >= 35) & (hue <= 85) & (sat > 75) & (val < 105)),
                "jaune_pale": ((hue >= 20) & (hue <= 45) & (sat > 20) & (val > 150)),
                "brun_clair": ((hue >= 10) & (hue <= 30) & (sat > 25) & (val > 110)),
                "brun_fonce": ((hue >= 5) & (hue <= 30) & (sat > 35) & (val < 100)),
                "noir": (val < 40),
        }
        total = float(np.count_nonzero(leaf_bool))
        result: Dict[str, float] = {}
        for label, zone_mask in zones.items():
                count = float(np.count_nonzero(zone_mask & leaf_bool))
                result[f"percent_{label}"] = round(100.0 * count / total, 2) if total > 0 else 0.0
        return result


# ---------------------------------------------------------------------------
# Masques de symptômes (jaune, brun, sombre, pâle, sec, nécrose)
# ---------------------------------------------------------------------------

def _build_symptom_masks(
        hsv: np.ndarray, leaf_mask: np.ndarray
) -> Dict[str, np.ndarray]:
        """Construit les masques binaires pour chaque type de symptôme visible."""
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        raw_masks = {
                "yellow_mask": (
                        ((hue >= 15) & (hue <= 45) & (sat > 40) & (val > 70))
                        | ((hue >= 20) & (hue <= 60) & (sat > 20) & (val > 100))
                ).astype(np.uint8) * 255,
                "brown_mask": (
                        (((hue >= 5) & (hue <= 30) & (sat > 35) & (val < 140))
                         | ((hue >= 0) & (hue <= 20) & (sat > 25) & (val < 100)))
                ).astype(np.uint8) * 255,
                "dark_mask": (
                        (((val < 65) & (sat > 20)) | ((val < 45) & (sat > 10)))
                ).astype(np.uint8) * 255,
                "pale_mask": (
                        ((sat < 70) & (val > 90) & (val < 240) & (hue >= 15) & (hue <= 90))
                ).astype(np.uint8) * 255,
                "dry_mask": (
                        ((sat < 55) & (val > 80) & (val < 200) & (hue >= 18) & (hue <= 50))
                ).astype(np.uint8) * 255,
                "necrotic_mask": (
                        (((hue >= 0) & (hue <= 25) & (sat > 30) & (val < 80)) | (val < 35))
                ).astype(np.uint8) * 255,
        }

        cleaned: Dict[str, np.ndarray] = {}
        for name, mask in raw_masks.items():
                mask = cv2.bitwise_and(mask, leaf_mask)
                cleaned[name] = _morph_mask(mask, kernel_size=(5, 5), close_iter=2, open_iter=1)
        return cleaned


# ---------------------------------------------------------------------------
# Analyse des taches (connected components + contours)
# ---------------------------------------------------------------------------

def _analyze_spots(spot_mask: np.ndarray, leaf_area: float) -> Dict[str, object]:
        """Analyse détaillée des taches via composantes connexes et contours."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                spot_mask, connectivity=8
        )

        areas: List[float] = []
        circularities: List[float] = []
        perimeters: List[float] = []
        contour_count = 0

        for label_id in range(1, num_labels):
                area = float(stats[label_id, cv2.CC_STAT_AREA])
                if area < 8.0:
                        continue
                areas.append(area)
                component = (labels == label_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_count += len(contours)
                for cnt in contours:
                        perim = float(cv2.arcLength(cnt, True))
                        perimeters.append(perim)
                        if perim > 0:
                                circ = 4.0 * np.pi * area / (perim * perim)
                                circularities.append(min(1.0, circ))

        spot_area = float(np.count_nonzero(spot_mask))
        spot_count = len(areas)

        # Zones fusionnées : dilatation morphologique regroupe les taches proches
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        merged_mask = cv2.dilate(spot_mask, kernel, iterations=2)
        merged_mask = _morph_mask(merged_mask, kernel_size=(9, 9), close_iter=1, open_iter=1)
        num_merged, _, merged_stats, _ = cv2.connectedComponentsWithStats(merged_mask, connectivity=8)
        merged_areas = [
                float(merged_stats[i, cv2.CC_STAT_AREA])
                for i in range(1, num_merged)
                if merged_stats[i, cv2.CC_STAT_AREA] >= 20
        ]
        merged_count = len(merged_areas)
        merged_total_area = float(np.count_nonzero(merged_mask))

        # Densité et regroupement spatial des centroïdes
        cluster_score = 0.0
        if spot_count >= 2:
                centers = np.array([
                        centroids[i] for i in range(1, num_labels)
                        if stats[i, cv2.CC_STAT_AREA] >= 8
                ])
                if len(centers) >= 2:
                        dists = []
                        for i in range(len(centers)):
                                for j in range(i + 1, len(centers)):
                                        dists.append(float(np.linalg.norm(centers[i] - centers[j])))
                        mean_dist = float(np.mean(dists))
                        h, w = spot_mask.shape
                        diag = float(np.hypot(h, w))
                        cluster_score = round(max(0.0, 100.0 * (1.0 - mean_dist / (diag * 0.5))), 2)

        return {
                "spot_count": spot_count,
                "spot_mean_area": round(float(np.mean(areas)), 2) if areas else 0.0,
                "spot_median_area": round(float(np.median(areas)), 2) if areas else 0.0,
                "spot_area_std": round(float(np.std(areas)), 2) if areas else 0.0,
                "spot_max_area": round(float(max(areas)), 2) if areas else 0.0,
                "spot_min_area": round(float(min(areas)), 2) if areas else 0.0,
                "spot_mean_circularity": round(float(np.mean(circularities)), 3) if circularities else 0.0,
                "spot_circularity_std": round(float(np.std(circularities)), 3) if circularities else 0.0,
                "spot_mean_perimeter": round(float(np.mean(perimeters)), 2) if perimeters else 0.0,
                "spot_contour_count": contour_count,
                "spot_density": round(100.0 * spot_area / leaf_area, 2) if leaf_area > 0 else 0.0,
                "spot_total_area": round(spot_area, 2),
                "spot_merged_count": merged_count,
                "spot_merged_area": round(merged_total_area, 2),
                "spot_merged_mean_area": round(float(np.mean(merged_areas)), 2) if merged_areas else 0.0,
                "spot_cluster_score": cluster_score,
        }


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

def _analyze_texture(img: np.ndarray, leaf_mask: np.ndarray, gray: np.ndarray) -> Dict[str, object]:
        """Analyse de texture : rugosité, variance locale, zones sèches/nécrotiques."""
        leaf_gray = cv2.bitwise_and(gray, gray, mask=leaf_mask)
        leaf_pixels = leaf_gray[leaf_mask == 255]
        if leaf_pixels.size == 0:
                return {
                        "texture_variance": 0.0,
                        "texture_roughness": 0.0,
                        "texture_local_variance_mean": 0.0,
                        "texture_label": "inconnue",
                        "percent_dry": 0.0,
                        "percent_necrotic": 0.0,
                }

        lap = cv2.Laplacian(leaf_gray, cv2.CV_64F)
        lap_leaf = lap[leaf_mask == 255]
        texture_variance = float(np.var(lap_leaf))
        texture_roughness = float(np.std(lap_leaf))

        # Variance locale par fenêtre glissante
        blur = cv2.GaussianBlur(leaf_gray.astype(np.float64), (0, 0), 3)
        sq_mean = cv2.GaussianBlur((leaf_gray.astype(np.float64)) ** 2, (0, 0), 3)
        local_var = sq_mean - blur ** 2
        local_var_leaf = local_var[leaf_mask == 255]
        local_var_mean = float(np.mean(local_var_leaf))

        # Classification lisse / rugueuse
        if texture_roughness < 12.0:
                texture_label = "lisse"
        elif texture_roughness < 28.0:
                texture_label = "moyenne"
        else:
                texture_label = "rugueuse"

        return {
                "texture_variance": round(texture_variance, 2),
                "texture_roughness": round(texture_roughness, 2),
                "texture_local_variance_mean": round(local_var_mean, 2),
                "texture_label": texture_label,
        }


# ---------------------------------------------------------------------------
# Analyse spatiale
# ---------------------------------------------------------------------------

def _spatial_spot_features(spot_mask: np.ndarray, leaf_mask: np.ndarray) -> Dict[str, object]:
        """Répartition spatiale des taches : bords, centre, quadrants, uniformité."""
        h, w = leaf_mask.shape
        border = max(1, int(min(h, w) * 0.12))

        border_mask = np.zeros_like(leaf_mask)
        border_mask[:border, :] = 255
        border_mask[-border:, :] = 255
        border_mask[:, :border] = 255
        border_mask[:, -border:] = 255
        border_mask = cv2.bitwise_and(border_mask, leaf_mask)
        center_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(border_mask))

        total_spot = float(np.count_nonzero(spot_mask))
        border_spot = float(np.count_nonzero(cv2.bitwise_and(spot_mask, border_mask)))
        center_spot = float(np.count_nonzero(cv2.bitwise_and(spot_mask, center_mask)))

        edge_pct = round(100.0 * border_spot / total_spot, 2) if total_spot > 0 else 0.0
        center_pct = round(100.0 * center_spot / total_spot, 2) if total_spot > 0 else 0.0

        if abs(edge_pct - center_pct) < 15.0:
                distribution = "uniforme"
        elif edge_pct > center_pct + 15.0:
                distribution = "concentrée_bords"
        else:
                distribution = "concentrée_centre"

        # Quadrants
        mid_y, mid_x = h // 2, w // 2
        quadrants = {
                "haut_gauche": spot_mask[:mid_y, :mid_x],
                "haut_droit": spot_mask[:mid_y, mid_x:],
                "bas_gauche": spot_mask[mid_y:, :mid_x],
                "bas_droit": spot_mask[mid_y:, mid_x:],
        }
        quad_counts = {k: float(np.count_nonzero(v)) for k, v in quadrants.items()}
        quad_values = list(quad_counts.values())
        quad_uniformity = 0.0
        if total_spot > 0 and quad_values:
                quad_uniformity = round(100.0 - (float(np.std(quad_values)) / (total_spot / 4.0 + 1e-6) * 100.0), 2)
                quad_uniformity = max(0.0, min(100.0, quad_uniformity))

        return {
                "spot_edge_percent": edge_pct,
                "spot_center_percent": center_pct,
                "spot_distribution": distribution,
                "spot_quadrant_haut_gauche": round(quad_counts.get("haut_gauche", 0), 2),
                "spot_quadrant_haut_droit": round(quad_counts.get("haut_droit", 0), 2),
                "spot_quadrant_bas_gauche": round(quad_counts.get("bas_gauche", 0), 2),
                "spot_quadrant_bas_droit": round(quad_counts.get("bas_droit", 0), 2),
                "spot_spatial_uniformity": quad_uniformity,
        }


def _vein_interaction(img: np.ndarray, leaf_mask: np.ndarray, spot_mask: np.ndarray) -> Dict[str, float]:
        """Estime le chevauchement des taches avec les nervures (top-hat morphologique)."""
        green = img[:, :, 1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
        _, vein_mask = cv2.threshold(tophat, 16, 255, cv2.THRESH_BINARY)
        vein_mask = cv2.bitwise_and(vein_mask, leaf_mask)
        vein_mask = _morph_mask(vein_mask, kernel_size=(7, 7), close_iter=1, open_iter=1)

        vein_area = float(np.count_nonzero(vein_mask))
        overlap = float(np.count_nonzero(cv2.bitwise_and(vein_mask, spot_mask)))
        spot_on_veins = round(100.0 * overlap / float(np.count_nonzero(spot_mask)), 2) if np.count_nonzero(spot_mask) else 0.0

        return {
                "vein_area": round(vein_area, 2),
                "vein_spot_overlap": round(overlap, 2),
                "vein_affected_percent": round(100.0 * overlap / vein_area, 2) if vein_area > 0 else 0.0,
                "spot_on_veins_percent": spot_on_veins,
        }


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def _detect_burned_edges(
        leaf_mask: np.ndarray, dark_mask: np.ndarray, brown_mask: np.ndarray
) -> Dict[str, float]:
        """Détecte les brûlures marginales (nécrose des bords de feuille)."""
        h, w = leaf_mask.shape
        border_thickness = max(3, int(min(h, w) * 0.08))
        kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (border_thickness * 2 + 1, border_thickness * 2 + 1)
        )
        eroded = cv2.erode(leaf_mask, kernel, iterations=1)
        edge_zone = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(eroded))
        edge_area = float(np.count_nonzero(edge_zone))
        if edge_area == 0:
                return {"percent_burned_edges": 0.0, "edge_burn_intensity": 0.0}
        burned = cv2.bitwise_or(dark_mask, brown_mask)
        burned_edge = cv2.bitwise_and(burned, edge_zone)
        pct_burned = round(100.0 * float(np.count_nonzero(burned_edge)) / edge_area, 2)
        return {
                "percent_burned_edges": pct_burned,
                "edge_burn_intensity": round(min(100.0, pct_burned * 1.4), 2),
        }


def _compute_lesion_symmetry(spot_mask: np.ndarray) -> float:
        """Score de symétrie de distribution des lésions (0=asymétrique, 100=symétrique)."""
        if np.count_nonzero(spot_mask) == 0:
                return 50.0
        h, w = spot_mask.shape
        mid_y, mid_x = h // 2, w // 2
        left = float(np.count_nonzero(spot_mask[:, :mid_x]))
        right = float(np.count_nonzero(spot_mask[:, mid_x:]))
        top = float(np.count_nonzero(spot_mask[:mid_y, :]))
        bot = float(np.count_nonzero(spot_mask[mid_y:, :]))
        lr = 1.0 - abs(left - right) / (left + right + 1e-6)
        tb = 1.0 - abs(top - bot) / (top + bot + 1e-6)
        return round(100.0 * (lr + tb) / 2.0, 2)


def _generate_heatmap(mask: np.ndarray, size: Tuple[int, int] = (48, 48)) -> List[List[int]]:
        """Produit une heatmap compacte (0-100) des zones malades, sérialisable en JSON."""
        heat = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        if heat.max() > 0:
                heat = (heat / heat.max()) * 100.0
        return np.clip(heat, 0, 100).astype(int).tolist()


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def analyze_leaf(image: object, debug: bool = False) -> Dict[str, object]:
        """Analyse une image de feuille et retourne un dictionnaire de caractéristiques.

        Args:
                image: chemin fichier ou tableau NumPy BGR
                debug: si True, inclut les masques binaires dans le résultat

        Retourne un dict avec couleur dominante, pourcentages, forme, texture,
        taches, répartition spatiale, heatmap, etc.
        """
        img = load_image(image)
        img = _ensure_small(img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Segmentation précise ---
        leaf_mask, seg_info = _segment_leaf_precise(img, hsv)
        leaf_area = int(np.count_nonzero(leaf_mask))
        if leaf_area == 0:
                raise ValueError("Impossible de détecter la feuille dans l'image (surface = 0).")

        # --- Forme de la feuille ---
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        leaf_aspect_ratio = 0.0
        leaf_width_height_ratio = 0.0
        leaf_compactness = 0.0
        leaf_contour_count = len(contours)
        leaf_shape = "inconnue"
        probable_plant = "Inconnu"

        if contours:
                largest = max(contours, key=cv2.contourArea)
                _, _, bw, bh = cv2.boundingRect(largest)
                if bh > 0:
                        leaf_aspect_ratio = float(bw) / float(bh)
                        leaf_width_height_ratio = leaf_aspect_ratio
                bounding_area = float(bw) * float(bh)
                if bounding_area > 0:
                        leaf_compactness = float(leaf_area) / bounding_area

                perimeter = float(cv2.arcLength(largest, True))
                rect = cv2.minAreaRect(largest)
                rw, rh = rect[1]
                shape_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else leaf_aspect_ratio or 1.0

                if shape_ratio >= 2.0:
                        leaf_shape = "feuille longue"
                elif shape_ratio >= 1.4:
                        leaf_shape = "feuille fine"
                else:
                        leaf_shape = "feuille large"

                if leaf_shape == "feuille longue":
                        probable_plant = "maïs"
                elif leaf_shape == "feuille fine":
                        probable_plant = "haricot"
                elif leaf_compactness >= 0.70:
                        probable_plant = "tomate"
                elif seg_info["percent_leaf_area"] > 20:
                        probable_plant = "manioc"
                else:
                        probable_plant = "pomme de terre"

                if leaf_shape == "feuille fine" and 0.30 <= leaf_compactness <= 0.60:
                        probable_plant = "riz"

        # --- Couleur dominante ---
        valid_leaf = leaf_mask == 255
        valid_color = valid_leaf & (sat > 30) & (val > 25)
        if np.count_nonzero(valid_color) == 0:
                valid_color = valid_leaf

        leaf_hues = hue[valid_color]
        if leaf_hues.size == 0:
                leaf_hues = hue[valid_leaf]
        dominant_hue = int(np.argmax(np.bincount(leaf_hues, minlength=180)))
        dominant_color_name = _map_hue_to_color_name(dominant_hue)

        # --- Masques symptômes ---
        masks = _build_symptom_masks(hsv, leaf_mask)
        yellow_mask = masks["yellow_mask"]
        brown_mask = masks["brown_mask"]
        dark_mask = masks["dark_mask"]
        pale_mask = masks["pale_mask"]
        dry_mask = masks["dry_mask"]
        necrotic_mask = masks["necrotic_mask"]

        yellow_area = int(np.count_nonzero(yellow_mask))
        brown_area = int(np.count_nonzero(brown_mask))
        dark_area = int(np.count_nonzero(dark_mask))
        pale_area = int(np.count_nonzero(pale_mask))
        dry_area = int(np.count_nonzero(dry_mask))
        necrotic_area = int(np.count_nonzero(necrotic_mask))

        # Masque global des taches (union des symptômes)
        spot_mask = np.clip(
                yellow_mask.astype(np.uint16)
                + brown_mask.astype(np.uint16)
                + dark_mask.astype(np.uint16)
                + pale_mask.astype(np.uint16)
                + necrotic_mask.astype(np.uint16),
                0, 255,
        ).astype(np.uint8)
        spot_mask = _morph_mask(spot_mask, kernel_size=(5, 5), close_iter=2, open_iter=1)
        spot_mask = cv2.bitwise_and(spot_mask, leaf_mask)
        unhealthy_mask = spot_mask

        # --- Nouvelles métriques avancées ---
        burned_edge_features = _detect_burned_edges(leaf_mask, dark_mask, brown_mask)
        lesion_symmetry = _compute_lesion_symmetry(spot_mask)

        # Trous/perforations : zones sombres profondément à l'intérieur de la feuille
        interior_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        leaf_interior = cv2.erode(leaf_mask, interior_kernel, iterations=1)
        holes_mask = cv2.bitwise_and(dark_mask, leaf_interior)
        pct_holes = round(100.0 * float(np.count_nonzero(holes_mask)) / float(leaf_area), 2) if leaf_area > 0 else 0.0

        # Humidité/aspect huileux : saturation élevée + teinte verte sombre
        humidity_mask = (
                ((hue >= 35) & (hue <= 85) & (sat > 100) & (val > 30) & (val < 90))
        ).astype(np.uint8) * 255
        humidity_mask = cv2.bitwise_and(humidity_mask, leaf_mask)
        pct_humid = round(100.0 * float(np.count_nonzero(humidity_mask)) / float(leaf_area), 2) if leaf_area > 0 else 0.0

        # --- Analyses dérivées ---
        spot_features = _analyze_spots(spot_mask, float(leaf_area))
        texture_features = _analyze_texture(img, leaf_mask, gray)
        spatial_features = _spatial_spot_features(spot_mask, leaf_mask)
        vein_features = _vein_interaction(img, leaf_mask, spot_mask)
        color_zones = _color_zone_percentages(hue, sat, val, valid_color)
        heatmap = _generate_heatmap(unhealthy_mask)

        # --- Pourcentages ---
        pct_yellow = 100.0 * yellow_area / leaf_area
        pct_brown = 100.0 * brown_area / leaf_area
        pct_unhealthy = 100.0 * int(np.count_nonzero(unhealthy_mask)) / leaf_area
        pct_dry = 100.0 * dry_area / leaf_area
        pct_necrotic = 100.0 * necrotic_area / leaf_area
        pct_dark = 100.0 * dark_area / leaf_area
        pct_pale = 100.0 * pale_area / leaf_area

        # --- Metrics addition: contrast and lesion-type heuristics (non-destructive)
        try:
                mean_leaf_brightness = float(np.mean(gray[leaf_mask == 255]))
        except Exception:
                mean_leaf_brightness = float(np.mean(gray))
        mean_spot_brightness = float(np.mean(gray[spot_mask == 255])) if np.count_nonzero(spot_mask) else mean_leaf_brightness
        contrast_score = round(max(0.0, mean_leaf_brightness - mean_spot_brightness), 2)

        lesion_types = []
        # Chlorosis: yellowing without many distinct spots
        if pct_yellow > 15 and spot_features.get('spot_count', 0) < 3:
                lesion_types.append('chlorose')
        # Mosaic / viral pattern: uniform mosaic of yellow/green
        if spatial_features.get('spot_distribution') == 'uniforme' and pct_yellow > 10 and spot_features.get('spot_count', 0) > 5:
                lesion_types.append('mosaïque')
        # Necrosis: notable necrotic area or brown dominance
        if pct_necrotic > 12 or pct_brown > 18:
                lesion_types.append('nécrose')
        # Circular spots: high circularity and moderate mean area
        if spot_features.get('spot_mean_circularity', 0) > 0.7 and spot_features.get('spot_mean_area', 0) > 20:
                lesion_types.append('taches circulaires')
        # Marginal burns: spots concentrated on leaf edges with dark regions
        if spatial_features.get('spot_edge_percent', 0) > 60 and pct_dark > 8:
                lesion_types.append('brûlures marginales')

        if not lesion_types:
                lesion_types.append('taches / anomalies indéterminées')

        if pct_yellow > 20 and dominant_color_name == "vert":
                dominant_color_name = "jaune"
        elif pct_brown > 15:
                dominant_color_name = "marron/orangé"

        if pct_unhealthy < 4:
                severity = "Faible"
        elif pct_unhealthy < 14:
                severity = "Modéré"
        elif pct_unhealthy < 35:
                severity = "Élevé"
        else:
                severity = "Sévère"

        # Distribution textuelle enrichie
        if spot_features["spot_cluster_score"] > 60:
                spatial_features["spot_distribution"] = "regroupée"
        elif spatial_features["spot_spatial_uniformity"] > 70:
                spatial_features["spot_distribution"] = "uniforme"

        result: Dict[str, object] = {
                # Couleur et santé globale
                "dominant_color_name": dominant_color_name,
                "dominant_hue": dominant_hue,
                "percent_leaf_area": seg_info["percent_leaf_area"],
                "percent_yellow": round(pct_yellow, 2),
                "percent_brown": round(pct_brown, 2),
                "percent_unhealthy": round(pct_unhealthy, 2),
                "percent_dark": round(pct_dark, 2),
                "percent_pale": round(pct_pale, 2),
                "percent_dry": round(pct_dry, 2),
                "percent_necrotic": round(pct_necrotic, 2),
                "severity": severity,
                # Forme
                "leaf_aspect_ratio": round(leaf_aspect_ratio, 2),
                "leaf_width_height_ratio": round(leaf_width_height_ratio, 2),
                "leaf_compactness": round(leaf_compactness, 2),
                "leaf_contour_count": leaf_contour_count,
                "leaf_shape": leaf_shape,
                "probable_plant": probable_plant,
                # Segmentation
                "leaf_segmentation_quality": seg_info["leaf_segmentation_quality"],
                "leaf_segmentation_overlap_percent": seg_info["leaf_segmentation_overlap_percent"],
                "refined_leaf_area": seg_info["refined_leaf_area"],
                # Couleurs avancées
                "percent_vert_clair": color_zones["percent_vert_clair"],
                "percent_vert_fonce": color_zones["percent_vert_fonce"],
                "percent_jaune_pale": color_zones["percent_jaune_pale"],
                "percent_brun_clair": color_zones["percent_brun_clair"],
                "percent_brun_fonce": color_zones["percent_brun_fonce"],
                "percent_noir": color_zones["percent_noir"],
                # Texture
                "texture_variance": texture_features["texture_variance"],
                "texture_roughness": texture_features["texture_roughness"],
                "texture_local_variance_mean": texture_features["texture_local_variance_mean"],
                "texture_label": texture_features["texture_label"],
                # Taches
                "spot_count": spot_features["spot_count"],
                "spot_mean_area": spot_features["spot_mean_area"],
                "spot_median_area": spot_features["spot_median_area"],
                "spot_area_std": spot_features["spot_area_std"],
                "spot_max_area": spot_features["spot_max_area"],
                "spot_min_area": spot_features["spot_min_area"],
                "spot_mean_circularity": spot_features["spot_mean_circularity"],
                "spot_circularity_std": spot_features["spot_circularity_std"],
                "spot_mean_perimeter": spot_features["spot_mean_perimeter"],
                "spot_contour_count": spot_features["spot_contour_count"],
                "spot_density": spot_features["spot_density"],
                "spot_total_area": spot_features["spot_total_area"],
                "spot_merged_count": spot_features["spot_merged_count"],
                "spot_merged_area": spot_features["spot_merged_area"],
                "spot_merged_mean_area": spot_features["spot_merged_mean_area"],
                "spot_cluster_score": spot_features["spot_cluster_score"],
                # Spatial
                "spot_edge_percent": spatial_features["spot_edge_percent"],
                "spot_center_percent": spatial_features["spot_center_percent"],
                "spot_distribution": spatial_features["spot_distribution"],
                "spot_quadrant_haut_gauche": spatial_features["spot_quadrant_haut_gauche"],
                "spot_quadrant_haut_droit": spatial_features["spot_quadrant_haut_droit"],
                "spot_quadrant_bas_gauche": spatial_features["spot_quadrant_bas_gauche"],
                "spot_quadrant_bas_droit": spatial_features["spot_quadrant_bas_droit"],
                "spot_spatial_uniformity": spatial_features["spot_spatial_uniformity"],
                # Nervures
                "vein_area": vein_features["vein_area"],
                "vein_spot_overlap": vein_features["vein_spot_overlap"],
                "vein_affected_percent": vein_features["vein_affected_percent"],
                "spot_on_veins_percent": vein_features["spot_on_veins_percent"],
                # Heatmap
                "heatmap": heatmap,
                "heatmap_width": len(heatmap[0]) if heatmap else 0,
                "heatmap_height": len(heatmap) if heatmap else 0,
                # Nouvelles métriques d'aide au diagnostic
                "contrast_score": contrast_score,
                "lesion_types": lesion_types,
                # Brûlures marginales
                "percent_burned_edges": burned_edge_features["percent_burned_edges"],
                "edge_burn_intensity": burned_edge_features["edge_burn_intensity"],
                # Symétrie des lésions (0=asymétrique, 100=symétrique)
                "lesion_symmetry": lesion_symmetry,
                # Trous / perforations internes
                "percent_holes": pct_holes,
                # Humidité / aspect huileux (indicateur mildiou, botrytis)
                "percent_humid": pct_humid,
        }

        # Alias rétrocompatibilité pour l'ancien champ spot_border_percent
        result["spot_border_percent"] = result["spot_edge_percent"]

        if debug:
                result["leaf_mask"] = leaf_mask
                result["yellow_mask"] = yellow_mask
                result["brown_mask"] = brown_mask
                result["dark_mask"] = dark_mask
                result["pale_mask"] = pale_mask
                result["dry_mask"] = dry_mask
                result["necrotic_mask"] = necrotic_mask
                result["unhealthy_mask"] = unhealthy_mask

        return result


def validate_leaf_image(image: object) -> Dict[str, object]:
        """Valide strictement si l'image contient une feuille/plante visible.

        Critères stricts :
        1. Couverture végétale suffisante (vert + jaune/brun pour feuilles malades)
        2. Indice ExG positif obligatoire sur zone significative
        3. Forme organique non-géométrique
        4. Texture naturelle (nervures, rugosité biologique)
        5. Absence de fond artificiel dominant
        6. Rejet formes géométriques parfaites (bâtiments, objets)

        Seuil de validation : confidence >= 70 (strict)

        Retourne:
                {
                        "is_leaf": True/False,
                        "confidence": 0-100,
                        "reason": "..."
                }
        """
        try:
                img = load_image(image)
                img = _ensure_small(img, max_dim=640)
        except Exception as exc:
                return {"is_leaf": False, "confidence": 0, "reason": f"Impossible de charger l'image : {exc}"}

        h, w = img.shape[:2]
        total_pixels = float(h * w)
        if total_pixels < 400:
                return {"is_leaf": False, "confidence": 0, "reason": "Image trop petite"}

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        score = 0
        reasons: List[str] = []
        reject_reasons: List[str] = []

        b_f = img[:, :, 0].astype(np.float32)
        g_f = img[:, :, 1].astype(np.float32)
        r_f = img[:, :, 2].astype(np.float32)

        # ---------------------------------------------------------------
        # Critère 1 : couverture végétale (vert + jaune + brun pour malades)
        # ---------------------------------------------------------------
        green_mask = ((hue >= 25) & (hue <= 90) & (sat > 30) & (val > 30)).astype(np.uint8)
        yellow_mask = ((hue >= 12) & (hue <= 35) & (sat > 35) & (val > 50)).astype(np.uint8)
        brown_mask = ((hue >= 4) & (hue <= 22) & (sat > 28) & (val > 22)).astype(np.uint8)
        veg_mask = np.clip(
                green_mask.astype(np.uint16) + yellow_mask.astype(np.uint16) + brown_mask.astype(np.uint16),
                0, 1,
        ).astype(np.uint8)
        pct_veg = 100.0 * float(np.count_nonzero(veg_mask)) / total_pixels
        pct_green_only = 100.0 * float(np.count_nonzero(green_mask)) / total_pixels

        if pct_veg >= 30.0:
                score += 40
                reasons.append(f"végétation dominante ({pct_veg:.0f}%)")
        elif pct_veg >= 18.0:
                score += 25
                reasons.append(f"végétation présente ({pct_veg:.0f}%)")
        elif pct_veg >= 8.0:
                score += 12
                reasons.append(f"végétation faible ({pct_veg:.0f}%)")
        else:
                reject_reasons.append(f"couleurs végétales insuffisantes ({pct_veg:.0f}%)")

        # ---------------------------------------------------------------
        # Critère 2 : Indice ExG (Excess Green Index) — obligatoire positif
        # ---------------------------------------------------------------
        exg = 2.0 * g_f - r_f - b_f
        mean_exg = float(np.mean(exg))
        pct_exg_pos = 100.0 * float(np.count_nonzero(exg > 8)) / total_pixels

        if mean_exg > 0 and pct_exg_pos >= 20.0:
                score += 28
                reasons.append(f"ExG positif ({pct_exg_pos:.0f}% pixels, moy={mean_exg:.1f})")
        elif mean_exg > 0 and pct_exg_pos >= 10.0:
                score += 18
                reasons.append(f"ExG partiel ({pct_exg_pos:.0f}%)")
        elif mean_exg > 0 and pct_exg_pos >= 4.0:
                score += 8
        else:
                reject_reasons.append(f"indice ExG non végétal (moy={mean_exg:.1f})")

        # ---------------------------------------------------------------
        # Critère 3 : forme organique (non-géométrique)
        # ---------------------------------------------------------------
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_v, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        organic_score = 0
        if contours_v:
                largest_v = max(contours_v, key=cv2.contourArea)
                c_area = float(cv2.contourArea(largest_v))
                c_perim = float(cv2.arcLength(largest_v, True))
                coverage = c_area / total_pixels
                if coverage >= 0.04 and c_perim > 0:
                        hull_v = cv2.convexHull(largest_v)
                        hull_area_v = float(cv2.contourArea(hull_v))
                        solidity_v = c_area / hull_area_v if hull_area_v > 0 else 0.0
                        compactness_v = (4.0 * np.pi * c_area) / (c_perim ** 2)
                        # Feuilles réelles : solidity 0.45-0.97, compactness 0.03-0.88
                        # Formes géométriques parfaites : compactness proche de 1.0 ou solidity > 0.99
                        if compactness_v > 0.92 and solidity_v > 0.97:
                                # Rectangle / cercle parfait = artificiel
                                organic_score = 0
                                reject_reasons.append(f"forme géométrique parfaite (compacité={compactness_v:.2f})")
                        elif 0.45 <= solidity_v <= 0.98 and 0.03 <= compactness_v <= 0.88:
                                organic_score = 20
                                reasons.append(f"forme organique (solidité={solidity_v:.2f}, compacité={compactness_v:.2f})")
                        elif coverage >= 0.06 and solidity_v > 0.40:
                                organic_score = 10
                                reasons.append("objet organique partiel")
                score += organic_score

        # ---------------------------------------------------------------
        # Critère 4 : texture naturelle (nervures, rugosité biologique)
        # ---------------------------------------------------------------
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Détection de nervures via top-hat morphologique (canal vert)
        green_ch = img[:, :, 1]
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(green_ch, cv2.MORPH_TOPHAT, tophat_kernel)
        _, vein_mask_v = cv2.threshold(tophat, 12, 255, cv2.THRESH_BINARY)
        pct_veins = 100.0 * float(np.count_nonzero(vein_mask_v)) / total_pixels

        if lap_var >= 80.0 and pct_veins >= 3.0:
                score += 12
                reasons.append(f"texture biologique (Lap={lap_var:.0f}, nervures={pct_veins:.0f}%)")
        elif lap_var >= 40.0:
                score += 6
                reasons.append(f"texture naturelle (Lap={lap_var:.0f})")
        elif lap_var < 15.0:
                score -= 10
                reject_reasons.append("surface trop uniforme/lisse (artificielle)")

        # ---------------------------------------------------------------
        # Critère 5 : rejet fond artificiel dominant
        # ---------------------------------------------------------------
        # Fond gris/blanc uniforme (mur, asphalte, papier)
        gray_mask = ((sat < 25) & (val > 40)).astype(np.uint8)
        pct_gray = 100.0 * float(np.count_nonzero(gray_mask)) / total_pixels
        if pct_gray > 65.0 and pct_veg < 15.0:
                score -= 20
                reject_reasons.append(f"fond uniforme gris/blanc ({pct_gray:.0f}%)")

        # Fond bleu artificiel (ciel, tissu)
        blue_mask = ((hue >= 95) & (hue <= 135) & (sat > 40) & (val > 50)).astype(np.uint8)
        pct_blue = 100.0 * float(np.count_nonzero(blue_mask)) / total_pixels
        if pct_blue > 40.0 and pct_veg < 15.0:
                score -= 20
                reject_reasons.append(f"fond bleu artificiel ({pct_blue:.0f}%)")

        # Teinte peau humaine dominante (main, visage) — hue 2-18, sat modérée
        skin_mask = ((hue >= 2) & (hue <= 18) & (sat > 28) & (sat < 160) & (val > 60)).astype(np.uint8)
        pct_skin = 100.0 * float(np.count_nonzero(skin_mask)) / total_pixels
        if pct_skin > 35.0 and pct_veg < 15.0:
                score -= 18
                reject_reasons.append(f"teinte peau humaine dominante ({pct_skin:.0f}%)")

        # ---------------------------------------------------------------
        # Critère 6 : rejet formes géométriques (détection de lignes droites)
        # ---------------------------------------------------------------
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=int(min(h, w) * 0.25), maxLineGap=10)
        n_straight_lines = len(lines) if lines is not None else 0
        if n_straight_lines >= 6 and pct_veg < 20.0:
                score -= 15
                reject_reasons.append(f"structure géométrique linéaire ({n_straight_lines} lignes droites)")

        # ---------------------------------------------------------------
        # Score final et décision
        # ---------------------------------------------------------------
        confidence = min(100, max(0, score))
        is_leaf = confidence >= 70

        if is_leaf:
                primary_reason = reasons[0] if reasons else "végétation confirmée"
        else:
                all_reasons = reject_reasons + [r for r in reasons if "insuffisant" in r or "faible" in r]
                primary_reason = (
                        "Image rejetée — " + "; ".join(all_reasons[:2])
                        if all_reasons else "Aucune feuille détectée"
                )

        return {"is_leaf": is_leaf, "confidence": confidence, "reason": primary_reason}


def validate_plant_match(analysis: Dict, plant_type: str) -> Dict[str, object]:
        """Vérifie la cohérence morphologique entre l'image analysée et le type de plante choisi.

        Utilise les caractéristiques de forme extraites par analyze_leaf :
        - leaf_aspect_ratio, leaf_compactness, leaf_shape, percent_leaf_area

        Retourne:
                {
                        "plant_match": True/False,
                        "confidence": 0-100,
                        "reason": "..."
                }
        """
        if not plant_type:
                return {"plant_match": True, "confidence": 100, "reason": "type de plante non précisé"}

        # Profils morphologiques par type de plante
        # aspect_ratio = rapport largeur/hauteur de la bounding box
        # compactness = aire feuille / aire bounding box (0=très étiré, 1=compact/rond)
        profiles: Dict[str, Dict] = {
                "maïs": {
                        "shapes": ["feuille longue", "feuille fine"],
                        "ar_min": 0.08, "ar_max": 0.65,     # feuille très longue → bounding box étroite
                        "comp_min": 0.20, "comp_max": 0.82,
                        "description": "feuille longue et étroite",
                },
                "riz": {
                        "shapes": ["feuille longue", "feuille fine"],
                        "ar_min": 0.06, "ar_max": 0.70,
                        "comp_min": 0.18, "comp_max": 0.80,
                        "description": "feuille fine et allongée",
                },
                "blé": {
                        "shapes": ["feuille longue", "feuille fine"],
                        "ar_min": 0.05, "ar_max": 0.55,
                        "comp_min": 0.15, "comp_max": 0.78,
                        "description": "feuille très fine et longue",
                },
                "sorgho": {
                        "shapes": ["feuille longue", "feuille fine"],
                        "ar_min": 0.07, "ar_max": 0.65,
                        "comp_min": 0.18, "comp_max": 0.82,
                        "description": "feuille longue type graminée",
                },
                "mil": {
                        "shapes": ["feuille longue", "feuille fine"],
                        "ar_min": 0.06, "ar_max": 0.60,
                        "comp_min": 0.15, "comp_max": 0.80,
                        "description": "feuille fine et allongée",
                },
                "tomate": {
                        "shapes": ["feuille large", "feuille fine"],
                        "ar_min": 0.50, "ar_max": 2.20,
                        "comp_min": 0.40, "comp_max": 0.95,
                        "description": "feuille large et composée",
                },
                "manioc": {
                        "shapes": ["feuille large", "feuille fine"],
                        "ar_min": 0.45, "ar_max": 2.00,
                        "comp_min": 0.28, "comp_max": 0.88,
                        "description": "feuille palmée aux lobes digitiformes",
                },
                "arachide": {
                        "shapes": ["feuille large", "feuille fine"],
                        "ar_min": 0.55, "ar_max": 1.90,
                        "comp_min": 0.50, "comp_max": 0.95,
                        "description": "petite feuille ovale de légumineuse",
                },
                "coton": {
                        "shapes": ["feuille large", "feuille fine"],
                        "ar_min": 0.45, "ar_max": 2.10,
                        "comp_min": 0.28, "comp_max": 0.88,
                        "description": "feuille large lobée",
                },
        }

        profile = profiles.get(plant_type)
        if profile is None:
                return {"plant_match": True, "confidence": 80, "reason": f"profil inconnu pour '{plant_type}'"}

        ar = float(analysis.get("leaf_aspect_ratio") or analysis.get("leaf_width_height_ratio") or 1.0)
        comp = float(analysis.get("leaf_compactness") or 0.5)
        shape = (analysis.get("leaf_shape") or "").lower()
        pct_leaf = float(analysis.get("percent_leaf_area") or 0.0)

        match_score = 0
        reasons_ok: List[str] = []
        reasons_ko: List[str] = []

        # -- Forme attendue --
        shape_ok = any(s in shape for s in profile["shapes"])
        if shape_ok:
                match_score += 40
                reasons_ok.append(f"forme compatible ({shape})")
        else:
                reasons_ko.append(f"forme incompatible : attendu '{profile['description']}', détecté '{shape}'")

        # -- Aspect ratio --
        ar_ok = profile["ar_min"] <= ar <= profile["ar_max"]
        if ar_ok:
                match_score += 35
                reasons_ok.append(f"ratio longueur/largeur compatible ({ar:.2f})")
        else:
                # Tolérance : pénalité progressive selon l'écart
                distance = min(abs(ar - profile["ar_min"]), abs(ar - profile["ar_max"]))
                if distance <= 0.25:
                        match_score += 15
                        reasons_ok.append(f"ratio limite acceptable ({ar:.2f})")
                else:
                        reasons_ko.append(
                                f"ratio incompatible ({ar:.2f}) — attendu [{profile['ar_min']:.2f}-{profile['ar_max']:.2f}]"
                        )

        # -- Compacité --
        comp_ok = profile["comp_min"] <= comp <= profile["comp_max"]
        if comp_ok:
                match_score += 25
                reasons_ok.append(f"compacité compatible ({comp:.2f})")
        else:
                distance_c = min(abs(comp - profile["comp_min"]), abs(comp - profile["comp_max"]))
                if distance_c <= 0.15:
                        match_score += 10

        plant_match = match_score >= 55
        confidence_match = min(100, match_score)

        if plant_match:
                reason = reasons_ok[0] if reasons_ok else "morphologie compatible"
        else:
                reason = reasons_ko[0] if reasons_ko else "morphologie incompatible avec la plante sélectionnée"

        return {
                "plant_match": plant_match,
                "confidence": confidence_match,
                "reason": reason,
                "expected": profile["description"],
                "detected_shape": shape,
                "detected_ar": round(ar, 3),
                "detected_compactness": round(comp, 3),
        }


def save_mask(mask: np.ndarray, path: str) -> None:
        """Sauvegarde un masque binaire (0/255) en PNG (utile en mode debug)."""
        cv2.imwrite(path, mask)


if __name__ == "__main__":
        import sys

        if len(sys.argv) < 2:
                print("Usage: python image_analysis.py <chemin_image>")
                sys.exit(1)

        res = analyze_leaf(sys.argv[1], debug=False)
        print("Analyse de l'image :")
        for key, value in res.items():
                if key.endswith("_mask"):
                        continue
                print(f"  {key}: {value}")
