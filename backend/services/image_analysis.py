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


def _detect_fabric_pattern(gray: np.ndarray, total_pixels: float) -> Tuple[bool, str]:
        """Détecte si l'image contient un tissu/vêtement via analyse de périodicité (FFT).

        Tissu = motif régulier → pics FFT intenses hors DC.
        Feuille = texture irrégulière → pas de périodicité.

        Retourne (is_fabric, reason).
        """
        f_img = np.float32(gray) / 255.0
        dft = np.fft.fft2(f_img)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        mask_radius = max(5, min(h, w) // 12)
        ys, xs = np.ogrid[:h, :w]
        dc_mask = ((ys - cy) ** 2 + (xs - cx) ** 2) <= mask_radius ** 2
        mag_no_dc = magnitude.copy()
        mag_no_dc[dc_mask] = 0.0

        flat = mag_no_dc.flatten()
        top_n = min(20, len(flat))
        top_vals = np.partition(flat, -top_n)[-top_n:]
        peak_ratio = float(np.mean(top_vals)) / (float(np.mean(magnitude)) + 1e-6)

        is_fabric = peak_ratio > 15.0
        reason = f"tissu détecté (FFT peak_ratio={peak_ratio:.1f})" if is_fabric else f"texture apériodique (peak_ratio={peak_ratio:.1f})"
        return is_fabric, reason


# ---------------------------------------------------------------------------
# Analyse couleur naturelle vs artificielle
# ---------------------------------------------------------------------------

def _analyze_color_naturalness(
        img: np.ndarray, hsv: np.ndarray, gray: np.ndarray, total_pixels: float
) -> Tuple[float, Dict[str, float]]:
        """Analyse la naturalité de la distribution des couleurs.

        Objets artificiels : couleurs séparées en blocs nets, transitions brutales,
        couleurs synthétiques (bleu vif, violet, rouge pur, plastique).
        Feuilles malades : transitions naturelles, dégradés biologiques,
        mélanges vert/jaune/brun organiques, variété chromatique modérée.

        Retourne (natural_color_score 0-100, details).
        """
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        details: Dict[str, float] = {}

        # 1. Couleurs synthétiques/artificielles (bleu vif, violet, rose neon, rouge pur)
        synthetic = (
                ((hue >= 100) & (hue <= 155) & (sat > 85))
                | ((hue >= 155) & (sat > 95))
                | ((hue >= 0)  & (hue <= 3)  & (sat > 165) & (val > 80))
        ).astype(np.uint8)
        pct_synthetic = 100.0 * float(np.count_nonzero(synthetic)) / total_pixels
        details["pct_synthetic"] = round(pct_synthetic, 2)

        # 2. Palette végétale naturelle large (vert, jaune, brun, beige, orange-brun rouille)
        natural_veg = (
                (hue >= 7) & (hue <= 102) & (sat > 8) & (val > 12) & (val < 248)
        ).astype(np.uint8)
        pct_natural = 100.0 * float(np.count_nonzero(natural_veg)) / total_pixels
        details["pct_natural"] = round(pct_natural, 2)

        # 3. Transitions de couleur — brutalité des changements de teinte
        hue_f = hue.astype(np.float32)
        grad_x = np.abs(np.diff(hue_f, axis=1))
        grad_y = np.abs(np.diff(hue_f, axis=0))
        grad_x = np.where(grad_x > 90, 180.0 - grad_x, grad_x)
        grad_y = np.where(grad_y > 90, 180.0 - grad_y, grad_y)
        pct_brutal = 100.0 * ((float(np.mean(grad_x > 40)) + float(np.mean(grad_y > 40))) / 2.0)
        details["pct_brutal_transitions"] = round(pct_brutal, 2)

        # 4. Variance locale de teinte — dégradés naturels vs blocs uniformes
        hue_blur = cv2.GaussianBlur(hue_f, (0, 0), 5)
        hue_local_var = float(np.mean((hue_f - hue_blur) ** 2))
        details["hue_local_var"] = round(hue_local_var, 2)

        # 5. Variété chromatique (nb pics > 2.5% → 1-2 = objet mono-couleur, ≥4 = vivant)
        hist_hue = np.bincount(hue.flatten(), minlength=180).astype(np.float32)
        hist_hue /= (hist_hue.sum() + 1e-6)
        hist_smooth = cv2.GaussianBlur(hist_hue.reshape(1, -1), (1, 9), 0).flatten()
        n_peaks = int(np.sum(hist_smooth > 0.025))
        details["n_color_peaks"] = float(n_peaks)

        # 6. Couleurs humaines/plastiques (ton chair uniformes sur grande surface)
        human_plastic = (
                ((hue >= 0) & (hue <= 20) & (sat > 15) & (sat < 180) & (val > 55))
        ).astype(np.uint8)
        pct_hp = 100.0 * float(np.count_nonzero(human_plastic)) / total_pixels
        details["pct_human_plastic"] = round(pct_hp, 2)

        # === Score de naturalité ===
        score = 42.0
        score += min(30.0, pct_natural * 0.48)
        score -= min(25.0, pct_synthetic * 2.2)
        score -= min(18.0, pct_brutal * 0.75)
        score += min(12.0, hue_local_var * 0.22)
        score += min(10.0, max(0.0, n_peaks - 2) * 2.2)

        return round(max(0.0, min(100.0, score)), 2), details


# ---------------------------------------------------------------------------
# Analyse texture biologique vs artificielle
# ---------------------------------------------------------------------------

def _analyze_material_texture(
        img: np.ndarray, hsv: np.ndarray, gray: np.ndarray, total_pixels: float
) -> Tuple[float, Dict[str, float]]:
        """Différencie texture végétale vs tissu vs plastique vs mur vs métal vs peau.

        Score élevé → texture biologique/organique.
        Score bas   → texture artificielle ou trop uniforme.

        Retourne (biological_texture_score 0-100, details).
        """
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        details: Dict[str, float] = {}

        # A. Rugosité globale (Laplacien)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(lap))
        details["lap_var"] = round(lap_var, 1)

        # B. Variance multi-échelle — un végétal varie différemment selon l'échelle
        scales: List[float] = []
        for ksize in [3, 7, 15]:
                bl = cv2.GaussianBlur(gray, (ksize, ksize), 0)
                scales.append(float(np.mean(np.abs(gray.astype(np.float32) - bl.astype(np.float32)))))
        multi_scale_var = float(np.std(scales))
        details["multi_scale_var"] = round(multi_scale_var, 2)

        # C. Micro-variations biologiques (gradient morphologique 3×3)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k3)
        bio_micro_var = float(np.std(gradient))
        details["bio_micro_var"] = round(bio_micro_var, 2)

        # D. Surface plate artificielle (plastique, écran, métal poli)
        val_std = float(np.std(val))
        pct_specular = 100.0 * float(np.count_nonzero(val > 248)) / total_pixels
        is_flat = val_std < 22.0 and lap_var < 40.0 and pct_specular < 1.5
        details["val_std"] = round(val_std, 1)
        details["is_flat"] = float(is_flat)

        # E. Détecteur peau — teinte chaude + texture lisse + ton uniforme
        skin = ((hue >= 2) & (hue <= 20) & (sat > 18) & (sat < 175) & (val > 55)).astype(np.uint8)
        pct_skin = 100.0 * float(np.count_nonzero(skin)) / total_pixels
        details["pct_skin"] = round(pct_skin, 2)

        # F. Variance locale de luminosité (structure complexe = organique)
        gray_d = gray.astype(np.float64)
        sq_mean = cv2.GaussianBlur(gray_d ** 2, (7, 7), 0)
        mean_sq = cv2.GaussianBlur(gray_d, (7, 7), 0) ** 2
        local_var = np.maximum(0.0, sq_mean - mean_sq)
        lv_mean = float(np.mean(local_var))
        lv_std  = float(np.std(local_var))
        lv_ratio = lv_std / (lv_mean + 1.0)
        details["lv_ratio"] = round(lv_ratio, 3)

        # === Score biologique ===
        bio = 28.0
        if   lap_var > 250: bio += 24.0
        elif lap_var > 100: bio += 17.0
        elif lap_var > 40:  bio += 10.0
        elif lap_var > 12:  bio += 4.0
        else:               bio -= 12.0

        bio += min(15.0, bio_micro_var * 0.48)
        bio += min(13.0, multi_scale_var * 6.5)
        bio += min(12.0, lv_ratio * 4.2)

        if is_flat:         bio -= 24.0
        bio -= min(13.0, pct_skin * 0.20)

        return round(max(0.0, min(100.0, bio)), 2), details


# ---------------------------------------------------------------------------
# Analyse contours organiques vs artificiels
# ---------------------------------------------------------------------------

def _analyze_organic_contours(
        img: np.ndarray, gray: np.ndarray, total_pixels: float
) -> Tuple[float, Dict[str, float]]:
        """Différencie contours organiques (feuille) vs artificiels (objet manufacturé).

        Feuille : bords irréguliers, dentelés, complexité élevée, asymétrie naturelle.
        Objet   : rectangles, lignes droites, formes géométriques répétées.

        Retourne (organic_contour_score 0-100, details).
        """
        details: Dict[str, float] = {}

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # A. Lignes droites (Hough) — signature d'objet artificiel
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=30, maxLineGap=10)
        n_straight = len(lines) if lines is not None else 0
        total_line_len = 0.0
        if lines is not None:
                for ln in lines:
                        x1, y1, x2, y2 = ln[0]
                        total_line_len += float(np.hypot(x2 - x1, y2 - y1))
        pct_straight = min(100.0, 100.0 * total_line_len / (float(np.sqrt(total_pixels)) * 8.0 + 1.0))
        details["n_straight_lines"] = float(n_straight)
        details["pct_straight"] = round(pct_straight, 2)

        # B. Contour principal — métriques d'irrégularité organique
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        score = 44.0
        if cnts:
                main = max(cnts, key=cv2.contourArea)
                c_area  = float(cv2.contourArea(main))
                c_perim = float(cv2.arcLength(main, True))
                if c_area > 0 and c_perim > 0:
                        compactness = (4.0 * np.pi * c_area) / (c_perim ** 2)
                        hull   = cv2.convexHull(main)
                        h_area = float(cv2.contourArea(hull))
                        solidity   = c_area / h_area if h_area > 0 else 0.5
                        complexity = c_perim / (float(np.sqrt(c_area)) + 1e-6)

                        details["compactness"] = round(compactness, 3)
                        details["solidity"]    = round(solidity, 3)
                        details["complexity"]  = round(complexity, 2)

                        # Forme géométrique parfaite → artificiel
                        if compactness > 0.82 and solidity > 0.96:
                                score -= 32.0
                        elif compactness > 0.72:
                                score -= 15.0

                        # Forme organique irrégulière
                        if 0.10 <= compactness <= 0.80 and 0.38 <= solidity <= 0.96:
                                score += 22.0

                        # Complexité élevée = dentelures = contour végétal
                        if   complexity > 18: score += 24.0
                        elif complexity > 12: score += 15.0
                        elif complexity > 7:  score += 7.0

        # Pénalité lignes droites
        score -= min(22.0, pct_straight * 0.42)

        # C. Asymétrie naturelle
        h, w = gray.shape
        left_m  = float(np.mean(gray[:, :w//2]))
        right_m = float(np.mean(gray[:, w//2:]))
        top_m   = float(np.mean(gray[:h//2, :]))
        bot_m   = float(np.mean(gray[h//2:, :]))
        asym = ((abs(left_m - right_m) + abs(top_m - bot_m)) /
                (left_m + right_m + top_m + bot_m + 1.0))
        details["asymmetry"] = round(asym, 3)
        if 0.015 <= asym <= 0.28:
                score += 9.0

        return round(max(0.0, min(100.0, score)), 2), details


def _compute_vein_structure_score(
        img: np.ndarray,
        gray: np.ndarray,
        total_pixels: float,
) -> Tuple[float, float, float, str]:
        """Calcule un score de structure vasculaire (nervures) multi-échelles.

        Retourne (vein_score, pct_primary_veins, pct_secondary_veins, note).
        - Nervure primaire  : kernel large (25×25) → nervure centrale
        - Nervures secondaires : kernel moyen (13×13)
        - Nervures tertiaires  : kernel petit (7×7)
        """
        green_ch = img[:, :, 1]

        def tophat_pct(ksize: int, thresh: int = 10) -> float:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                th = cv2.morphologyEx(green_ch, cv2.MORPH_TOPHAT, k)
                _, m = cv2.threshold(th, thresh, 255, cv2.THRESH_BINARY)
                return 100.0 * float(np.count_nonzero(m)) / total_pixels

        pct_primary = tophat_pct(25, thresh=12)    # nervure centrale forte
        pct_secondary = tophat_pct(13, thresh=10)  # ramifications 2nd ordre
        pct_tertiary = tophat_pct(7, thresh=8)     # nervures fines

        # Ratio ramification : si secondaires >> primaires → réseau complexe = feuille
        ramification = pct_secondary / (pct_primary + 0.5)

        if pct_primary < 0.5 and pct_secondary < 1.0:
                # Aucune nervure visible → tissu, peau, fond
                score = 0.0
                note = f"aucune nervure (prim={pct_primary:.1f}%, sec={pct_secondary:.1f}%)"
        elif pct_primary >= 1.5 and pct_secondary >= 3.0 and ramification >= 1.5:
                # Réseau veineux complet avec ramifications
                score = min(1.0, 0.70 + min(0.30, (pct_tertiary / 10.0)))
                note = f"réseau veineux complet (prim={pct_primary:.1f}%, sec={pct_secondary:.1f}%, ram={ramification:.1f})"
        elif pct_secondary >= 2.0:
                score = 0.50 + min(0.30, pct_secondary / 20.0)
                note = f"nervures partielles (sec={pct_secondary:.1f}%, ram={ramification:.1f})"
        else:
                score = 0.20
                note = f"nervures faibles (prim={pct_primary:.1f}%, sec={pct_secondary:.1f}%)"

        return round(score, 3), round(pct_primary, 2), round(pct_secondary, 2), note


# ---------------------------------------------------------------------------
# Détection des symptômes agricoles (boost score feuille malade)
# ---------------------------------------------------------------------------

def _compute_disease_pattern_score(
        img: np.ndarray,
        hsv: np.ndarray,
        gray: np.ndarray,
        total_pixels: float,
) -> Tuple[float, float, Dict[str, float]]:
        """Détecte les motifs de maladies agricoles et retourne un score 0-100.

        Reconnaît 13 symptômes : chlorose, rouille (pustules orange), nécrose,
        zones sèches, mildiou/oïdium, mosaïque virale, nervures, taches biologiques,
        brûlures, trous, jaunissement diffus, lésions irrégulières, texture bio.

        Rouille du maïs : détection avancée orange/brun + micro-pustules + densité.
        Objets artificiels n'ont aucun de ces signaux.

        Retourne (disease_pattern_score, vegetation_percent, symptom_details).
        """
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        symptoms: Dict[str, float] = {}

        # 1. Chlorose — jaunissement uniforme ou partiel (jaune-vert pâle, dégradé)
        chlorosis = (
                (hue >= 14) & (hue <= 48) & (sat > 20) & (val > 65)
        ).astype(np.uint8)
        symptoms["chlorose"] = 100.0 * float(np.count_nonzero(chlorosis)) / total_pixels

        # 2. Rouille — détection avancée orange-brun typique des pustules
        #    Plage principale : orange chaud (hue 3-20, sat élevée)
        rust_core = (
                (hue >= 3) & (hue <= 20) & (sat > 70) & (val > 55) & (val < 210)
        ).astype(np.uint8)
        #    Plage étendue brun-rouille : inclut les pustules plus sombres
        rust_ext = (
                (hue >= 0) & (hue <= 28) & (sat > 50) & (val > 40) & (val < 160)
        ).astype(np.uint8)
        rust_combined = np.clip(rust_core.astype(np.uint16) + rust_ext.astype(np.uint16), 0, 1).astype(np.uint8)
        pct_rust_color = 100.0 * float(np.count_nonzero(rust_combined)) / total_pixels
        symptoms["rouille"] = pct_rust_color

        # 2b. Rouille — micro-pustules dispersées (petites composantes rondes orange/brun)
        rust_mask_bin = (rust_combined * 255).astype(np.uint8)
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rust_mask_clean = cv2.morphologyEx(rust_mask_bin, cv2.MORPH_OPEN, k_dilate)
        n_rust, _, stats_rust, _ = cv2.connectedComponentsWithStats(rust_mask_clean, connectivity=8)
        rust_pustules = sum(
                1 for i in range(1, n_rust)
                if 4 < stats_rust[i, cv2.CC_STAT_AREA] < 2000
        )
        symptoms["rouille_pustules"] = min(100.0, rust_pustules * 5.0)

        # 3. Nécrose / brûlures — brun foncé à noir avec composante biologique
        necrosis = (
                ((hue >= 0) & (hue <= 25) & (sat > 22) & (val < 115))
                | (val < 32)
        ).astype(np.uint8)
        symptoms["necrose"] = 100.0 * float(np.count_nonzero(necrosis)) / total_pixels

        # 4. Zones sèches — beige/tan désaturé (feuilles brûlées, sèches)
        dry = (
                (sat < 70) & (val > 75) & (val < 225) & (hue >= 7) & (hue <= 55)
        ).astype(np.uint8)
        symptoms["sec"] = 100.0 * float(np.count_nonzero(dry)) / total_pixels

        # 5. Mildiou/Oïdium — zones blanchâtres, poudreuses, gris-vert
        mildew = (
                (sat < 55) & (val > 145) & (hue >= 22) & (hue <= 98)
        ).astype(np.uint8)
        symptoms["mildiou"] = 100.0 * float(np.count_nonzero(mildew)) / total_pixels

        # 6. Mosaïque virale — alternance vert/jaune irrégulière (variance locale élevée)
        mosaic_mask = (
                (hue >= 16) & (hue <= 92) & (sat > 16) & (val > 38)
        ).astype(np.uint8) * 255
        mosaic_lap = float(cv2.Laplacian(mosaic_mask, cv2.CV_64F).var())
        symptoms["mosaique"] = min(100.0, mosaic_lap / 75.0)

        # 7. Tons végétaux larges (inclut feuilles malades jaunes/brunes/sèches/rouillées)
        veg_broad = (
                ((hue >= 8)  & (hue <= 102) & (sat > 10) & (val > 12))
                | ((hue >= 2)  & (hue <= 8)  & (sat > 30) & (val > 28))
        ).astype(np.uint8)
        veg_percent = 100.0 * float(np.count_nonzero(veg_broad)) / total_pixels

        # 8. Nervures visibles — top-hat multi-échelles (structure vasculaire)
        green_ch = img[:, :, 1]
        k13 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        tophat13 = cv2.morphologyEx(green_ch, cv2.MORPH_TOPHAT, k13)
        _, vein_m = cv2.threshold(tophat13, 8, 255, cv2.THRESH_BINARY)
        vein_pct = 100.0 * float(np.count_nonzero(vein_m)) / total_pixels
        symptoms["nervures"] = vein_pct

        # 9. Taches biologiques irrégulières (composantes sombres de taille cohérente)
        dark_spots = (val < 88).astype(np.uint8) * 255
        n_comp, _, stats_spots, _ = cv2.connectedComponentsWithStats(dark_spots, connectivity=8)
        bio_spots = sum(
                1 for i in range(1, n_comp)
                if 15 < stats_spots[i, cv2.CC_STAT_AREA] < 7000
        )
        symptoms["taches_bio"] = min(100.0, bio_spots * 6.5)

        # 10. Texture biologique — irrégularité organique (std du gradient Laplacien)
        lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        bio_tex = float(np.std(lap_abs)) if lap_abs.size > 0 else 0.0
        symptoms["texture_bio"] = min(100.0, bio_tex / 1.4)

        # 11. Brûlures / lésions marginales (zones très sombres sur bords)
        dark_border = (val < 60).astype(np.uint8) * 255
        h_img, w_img = gray.shape
        border_w = max(3, min(h_img, w_img) // 10)
        border_zone = np.zeros_like(dark_border)
        border_zone[:border_w, :]  = 255
        border_zone[-border_w:, :] = 255
        border_zone[:, :border_w]  = 255
        border_zone[:, -border_w:] = 255
        burned_border = cv2.bitwise_and(dark_border, border_zone)
        symptoms["brulures"] = 100.0 * float(np.count_nonzero(burned_border)) / total_pixels

        # 12. Jaunissement diffus global (chlorose avancée ou sénescence)
        yellow_diffuse = (
                (hue >= 18) & (hue <= 42) & (sat > 35) & (val > 80)
        ).astype(np.uint8)
        symptoms["jaunissement"] = 100.0 * float(np.count_nonzero(yellow_diffuse)) / total_pixels

        # 13. Lésions irrégulières — composite brun+sombre+jaune (spectre maladie large)
        irregular_lesion = np.clip(
                (rust_core.astype(np.uint16)
                 + (((hue >= 0) & (hue <= 30) & (sat > 30) & (val < 130)).astype(np.uint16))
                 + (((hue >= 14) & (hue <= 50) & (sat > 25) & (val > 60) & (val < 160)).astype(np.uint16))),
                0, 1
        ).astype(np.uint8)
        symptoms["lesions"] = 100.0 * float(np.count_nonzero(irregular_lesion)) / total_pixels

        # ── Score de rouille spécifique (bonus fort si pustules ET couleur) ──
        rust_score_bonus = 0.0
        if pct_rust_color > 2.0 and rust_pustules >= 5:
                rust_score_bonus = min(18.0, pct_rust_color * 1.5 + rust_pustules * 0.8)
        elif pct_rust_color > 4.0:
                rust_score_bonus = min(12.0, pct_rust_color * 1.2)

        # === Score global — pondération par signification agricole ===
        sig_count = 0
        if symptoms["chlorose"]        >  5.0: sig_count += 1
        if symptoms["rouille"]         >  2.0: sig_count += 1
        if symptoms["rouille_pustules"]>  5.0: sig_count += 1   # pustules = signal fort
        if symptoms["necrose"]         >  3.5: sig_count += 1
        if symptoms["sec"]             >  7.0: sig_count += 1
        if symptoms["mildiou"]         >  3.5: sig_count += 1
        if symptoms["mosaique"]        >  8.0: sig_count += 1
        if symptoms["nervures"]        >  1.2: sig_count += 1
        if symptoms["taches_bio"]      >  0.0: sig_count += 1
        if symptoms["brulures"]        >  1.5: sig_count += 1
        if symptoms["jaunissement"]    >  4.0: sig_count += 1
        if symptoms["lesions"]         >  5.0: sig_count += 1
        if veg_percent                 > 10.0: sig_count += 1

        disease_pattern_score = min(100.0,
                sig_count * 11.0
                + min(20.0, veg_percent * 0.35)
                + min(14.0, symptoms["nervures"] * 1.5)
                + min(10.0, symptoms["texture_bio"] * 0.11)
                + rust_score_bonus
        )

        return (
                round(disease_pattern_score, 2),
                round(veg_percent, 2),
                {k: round(v, 2) for k, v in symptoms.items()},
        )


def validate_leaf_image(image: object, plant_type: Optional[str] = None) -> Dict[str, object]:
        """Validation feuille agricole — système simple et stable.

        Critères (aucun ne rejette seul) :
            texture organique : 50 %
            forme / contours  : 40 %
            indice couleur    : 10 %   ← le vert n'est PAS exigé

        Seuils :
            leaf_score < 40  → rejet
            40 ≤ score < 60  → feuille probable (warning)
            leaf_score ≥ 60  → feuille valide

        Seuls vrais rejets (signaux combinés, jamais un seul) :
            - peau humaine dominante + texture lisse
            - tissu / vêtement confirmé + texture uniforme
            - image quasi vide

        Formule : leaf_score = couleur*0.40 + texture*0.25 + forme*0.20 + contours*0.15

        Niveaux de décision :
            leaf_score < 45   → rejet définitif
            45 <= score < 65  → feuille probable (warning)
            leaf_score >= 65  → feuille valide

        AUCUN critère individuel ne provoque de rejet.
        Seul le score global détermine la décision.

        Tolérances intégrées :
        - Feuilles jaunes/malades : palette végétale large (pas uniquement vert)
        - Mauvaise lumière : bonus sous-exposition si végétation détectée
        - Flou caméra mobile : plancher texture relevé, jamais 0
        - Feuille partiellement visible : pénalité forme réduite

        Retourne:
                {
                        "is_leaf": bool,
                        "low_confidence_leaf": bool,
                        "leaf_score": float (0-100),
                        "color_score": float,
                        "texture_score": float,
                        "shape_score": float,
                        "contour_score": float,
                        "veg_percent": float,
                        "reason": str
                }
        """
        WARNING = 60.0

        def _empty(reason: str) -> Dict:
                return {
                        "is_leaf": False, "should_reject": True,
                        "low_confidence_leaf": False,
                        "leaf_score": 0.0, "texture_score": 0.0, "shape_score": 0.0,
                        "color_score": 0.0, "contour_score": 0.0, "veg_percent": 0.0,
                        "disease_pattern_score": 0.0, "vegetation_score": 0.0,
                        "confidence": 0, "vein_score": 0.0,
                        "symptom_details": {}, "reason": reason,
                }

        try:
                img = load_image(image)
                img = _ensure_small(img, max_dim=640)
        except Exception as exc:
                return _empty(f"Impossible de charger l'image : {exc}")

        h, w = img.shape[:2]
        total_pixels = float(h * w)
        if total_pixels < 400:
                return _empty("Image trop petite")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue  = hsv[:, :, 0]
        sat  = hsv[:, :, 1]
        val  = hsv[:, :, 2]

        # ═══════════════════════════════════════════════════════════════
        # A. TEXTURE BIOLOGIQUE (25%)
        #    Matière vivante = irrégulière, multi-échelle, entropie élevée.
        # ═══════════════════════════════════════════════════════════════
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        block = 16
        bh_b, bw_b = max(1, h // block), max(1, w // block)
        lv: List[float] = []
        for bi in range(bh_b):
                for bj in range(bw_b):
                        p = gray[bi*block:(bi+1)*block, bj*block:(bj+1)*block]
                        lv.append(float(p.var()))
        lv_mean  = float(np.mean(lv)) if lv else 0.0
        lv_std   = float(np.std(lv))  if lv else 0.0
        non_unif = lv_std / (lv_mean + 1.0)

        hist_g        = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        hist_n        = hist_g / (float(hist_g.sum()) + 1e-9)
        entropy       = float(-np.sum(hist_n * np.log2(hist_n + 1e-12)))
        entropy_score = min(100.0, (entropy / 5.5) * 100.0)

        blur_boost = 10.0 if lap_var < 50.0 else 0.0
        if   lap_var < 5:   lap_score = 15.0 + blur_boost
        elif lap_var < 25:  lap_score = 32.0 + min(10.0, non_unif * 8.0) + blur_boost
        elif lap_var < 70:  lap_score = 52.0 + min(15.0, non_unif * 10.0)
        elif lap_var < 180: lap_score = 68.0 + min(15.0, non_unif * 10.0)
        else:               lap_score = 80.0 + min(15.0, non_unif * 8.0)

        texture_score = 0.65 * lap_score + 0.35 * entropy_score

        is_fabric, fabric_reason = _detect_fabric_pattern(gray, total_pixels)
        fabric_uniform = non_unif < 0.35 and lap_var < 60.0
        if is_fabric and fabric_uniform:
                texture_score *= 0.50

        # Analyse texture biologique avancée (nouveau)
        texture_biological_score, tex_bio_details = _analyze_material_texture(
                img, hsv, gray, total_pixels
        )
        # Fusion : texture classique + texture biologique avancée
        texture_score_final = max(0.0, min(100.0,
                texture_score * 0.60 + texture_biological_score * 0.40
        ))

        # ═══════════════════════════════════════════════════════════════
        # B. CONTOURS ORGANIQUES (20%)
        #    Formes irrégulières = végétal. Géométriques = artificiel.
        # ═══════════════════════════════════════════════════════════════
        organic_contour_score, contour_details = _analyze_organic_contours(
                img, gray, total_pixels
        )

        # Score forme classique (conservé pour shape_note et compatibilité)
        blurred     = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh_b = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _     = cv2.findContours(thresh_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shape_score  = organic_contour_score   # utilise le nouveau score directement
        shape_note   = "forme indeterminee"
        coverage_val = 0.0

        if cnts:
                lc           = max(cnts, key=cv2.contourArea)
                c_area       = float(cv2.contourArea(lc))
                c_perim      = float(cv2.arcLength(lc, True))
                coverage_val = c_area / total_pixels

                if coverage_val < 0.02:
                        shape_note = f"objet tres petit ({coverage_val*100:.1f}%)"
                elif c_perim > 0:
                        hull   = cv2.convexHull(lc)
                        h_area = float(cv2.contourArea(hull))
                        sol    = c_area / h_area if h_area > 0 else 0.5
                        cpt    = (4.0 * np.pi * c_area) / (c_perim ** 2)
                        complexity = c_perim / (float(np.sqrt(c_area)) + 1e-6)

                        if sol > 0.97 and cpt > 0.75:
                                shape_note = f"forme rectangulaire (sol={sol:.2f})"
                        elif cpt > 0.88:
                                shape_note = f"forme circulaire (cpt={cpt:.2f})"
                        elif 0.40 <= sol <= 0.97 and cpt < 0.85:
                                shape_note = f"forme organique (sol={sol:.2f}, cpt={cpt:.2f})"
                        elif sol > 0.97 and cpt <= 0.75:
                                shape_note = f"forme allongee (sol={sol:.2f})"
                        else:
                                shape_note = f"forme ambigue (sol={sol:.2f}, cpt={cpt:.2f})"

        # ═══════════════════════════════════════════════════════════════
        # C. DISTRIBUTION COULEUR NATURELLE (10%)
        #    Couleurs végétales vs synthétiques/artificielles.
        # ═══════════════════════════════════════════════════════════════
        natural_color_score, color_nat_details = _analyze_color_naturalness(
                img, hsv, gray, total_pixels
        )

        # Score couleur classique (compatibilité)
        mask_natural = (
                ((hue >= 8) & (hue <= 96))
                | ((hue >= 0) & (hue <= 10) & (sat < 110))
                | ((hue >= 96) & (hue <= 112) & (sat < 75))
        ) & (val >= 18) & (val <= 245)

        mask_artificial = (
                ((hue >= 108) & (hue <= 145) & (sat > 55))
                | ((hue >= 145) & (sat > 50))
                | ((hue >= 0) & (hue <= 4) & (sat > 140) & (val > 100))
                | (val > 248) | (val < 10)
        )

        pct_natural    = 100.0 * float(np.count_nonzero(mask_natural))    / total_pixels
        pct_artificial = 100.0 * float(np.count_nonzero(mask_artificial)) / total_pixels

        color_score = max(0.0, min(100.0,
                pct_natural * (100.0 / 50.0) - pct_artificial * 1.5
        ))
        # Fusion couleur classique + naturalité avancée
        color_score_final = color_score * 0.45 + natural_color_score * 0.55

        pct_veg = 100.0 * float(np.count_nonzero(
                ((hue >= 22) & (hue <= 92) & (sat > 15) & (val > 15))
                | ((hue >= 15) & (hue <= 40) & (sat > 18) & (val > 30))
                | ((hue >= 4)  & (hue <= 25) & (sat > 15) & (val > 10))
        )) / total_pixels

        # ═══════════════════════════════════════════════════════════════
        # D. SYMPTÔMES AGRICOLES (35%)
        #    Chlorose, rouille (pustules), nécrose, mildiou, mosaïque,
        #    nervures, taches biologiques, brûlures — 13 signaux.
        #    Objet artificiel → score = 0.
        # ═══════════════════════════════════════════════════════════════
        disease_pattern_score, veg_percent_broad, symptom_details = _compute_disease_pattern_score(
                img, hsv, gray, total_pixels
        )

        # ═══════════════════════════════════════════════════════════════
        # E. STRUCTURE VÉGÉTALE (10%)
        #    Largeur de palette végétale (inclut feuilles malades).
        # ═══════════════════════════════════════════════════════════════
        vegetation_structure_score = min(100.0, veg_percent_broad * 1.8)

        # Boost shape si symptômes agricoles clairs (feuille sur fond blanc)
        if disease_pattern_score >= 30.0 and organic_contour_score < 35.0:
                shape_note = shape_note + " [corrige par symptomes agricoles]"

        # ═══════════════════════════════════════════════════════════════
        # SCORE FINAL PONDÉRÉ — 5 composantes
        # texture_bio   25%  : matière biologique vs artificielle
        # contour_org   20%  : forme organique vs géométrique
        # couleur_nat   10%  : palette végétale vs synthétique
        # maladie        35% : signaux agricoles (rouille, chlorose, etc.)
        # végétation     10% : spectre végétal large (feuilles malades)
        # ═══════════════════════════════════════════════════════════════
        leaf_score = round(
                texture_score_final      * 0.25 +
                organic_contour_score    * 0.20 +
                color_score_final        * 0.10 +
                disease_pattern_score    * 0.35 +
                vegetation_structure_score * 0.10,
                2,
        )

        # ═══════════════════════════════════════════════════════════════
        # REJET STRICT — 5 cas inutilisables (signaux COMBINÉS uniquement)
        # ═══════════════════════════════════════════════════════════════
        skin_mask = (
                (hue >= 2) & (hue <= 18) & (sat > 25) & (sat < 160) & (val > 60)
        ).astype(np.uint8)
        pct_skin = 100.0 * float(np.count_nonzero(skin_mask)) / total_pixels

        pct_synthetic = color_nat_details.get("pct_synthetic", 0.0)

        hard_reject = None

        # 1. Image vide / uniforme (rien à analyser)
        if lap_var < 2.0 and non_unif < 0.03:
                hard_reject = "image vide ou quasi uniforme"

        # 2. Peau humaine très dominante + zéro contenu agricole
        elif pct_skin > 75.0 and disease_pattern_score < 12.0 and texture_biological_score < 26.0:
                hard_reject = f"peau humaine dominante ({pct_skin:.0f}%) sans contenu agricole"

        # 3. Tissu/vêtement confirmé FFT + aucun signal biologique
        elif is_fabric and fabric_uniform and disease_pattern_score < 12.0 and veg_percent_broad < 5.0:
                hard_reject = f"vetement/tissu artificiel ({fabric_reason}) sans contenu biologique"

        # 4. Objet synthétique pur (plastique, écran, voiture) — couleur + texture plate + géom.
        elif (pct_synthetic > 55.0
              and tex_bio_details.get("is_flat", 0) > 0
              and disease_pattern_score < 10.0
              and organic_contour_score < 22.0):
                hard_reject = f"objet synthetique ({pct_synthetic:.0f}% couleurs artificielles, surface plate)"

        # 5. Forme géométrique pure + aucun signal biologique (mur, téléphone, papier)
        elif (contour_details.get("compactness", 0.5) > 0.85
              and contour_details.get("solidity", 0.5) > 0.97
              and disease_pattern_score < 10.0
              and veg_percent_broad < 4.0):
                hard_reject = "forme geometrique artificielle sans contenu biologique"

        if hard_reject:
                leaf_score = min(leaf_score, 14.0)

        leaf_score = round(leaf_score, 2)

        # ═══════════════════════════════════════════════════════════════
        # DÉCISION FINALE
        # ═══════════════════════════════════════════════════════════════
        should_reject       = hard_reject is not None
        is_leaf             = leaf_score >= 40.0
        low_confidence_leaf = not should_reject and not is_leaf
        # Feuille quasi-certaine si signaux agricoles clairs
        if disease_pattern_score >= 22.0 and not should_reject:
                is_leaf = True
                low_confidence_leaf = leaf_score < WARNING

        if should_reject:
                reason = f"Image rejetee — {hard_reject}"
        elif not is_leaf:
                parts: List[str] = []
                if disease_pattern_score < 12:    parts.append("aucun symptome agricole detecte")
                if texture_biological_score < 18: parts.append(f"texture non organique (Lap={lap_var:.0f})")
                if organic_contour_score < 20:    parts.append("contours geometriques/artificiels")
                if not parts:                     parts.append(f"score global faible ({leaf_score:.0f}/100)")
                reason = "Image non vegetale — " + "; ".join(parts)
        elif low_confidence_leaf:
                reason = f"Feuille probable avec confiance limitee — {shape_note}"
        else:
                reason = f"Feuille detectee — {shape_note}"

        # ── DEBUG LOGS (demandés) ──
        print(f"Disease pattern:              {disease_pattern_score:.1f}")
        print(f"Texture biological:           {texture_biological_score:.1f}")
        print(f"Organic contour:              {organic_contour_score:.1f}")
        print(f"Natural color distribution:   {natural_color_score:.1f}")
        print(f"Vegetation structure:         {vegetation_structure_score:.1f}")
        print(f"Final agricultural score:     {leaf_score:.1f}")
        print(
                f"[LEAF VALIDATION] score={leaf_score:.1f}/100 "
                f"tex_bio={texture_biological_score:.1f} org_cnt={organic_contour_score:.1f} "
                f"col_nat={natural_color_score:.1f} disease={disease_pattern_score:.1f} "
                f"veg={vegetation_structure_score:.1f} "
                f"is_leaf={is_leaf} should_reject={should_reject} reason='{reason}'"
        )

        return {
                "is_leaf":                   is_leaf,
                "should_reject":             should_reject,
                "low_confidence_leaf":       low_confidence_leaf,
                "leaf_score":                leaf_score,
                "confidence":                int(leaf_score),
                "texture_score":             round(texture_score_final,        1),
                "shape_score":               round(organic_contour_score,      1),
                "color_score":               round(color_score_final,          1),
                "contour_score":             round(organic_contour_score,      1),
                "disease_pattern_score":     round(disease_pattern_score,      1),
                "vegetation_score":          round(veg_percent_broad,          1),
                "vegetation_structure_score":round(vegetation_structure_score, 1),
                "texture_biological_score":  round(texture_biological_score,   1),
                "organic_contour_score":     round(organic_contour_score,      1),
                "natural_color_score":       round(natural_color_score,        1),
                "veg_percent":               round(pct_veg,                    1),
                "vein_score":                round(symptom_details.get("nervures", 0.0), 1),
                "symptom_details":           symptom_details,
                "reason":                    reason,
        }




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
