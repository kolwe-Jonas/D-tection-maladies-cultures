/**
 * Client API — connexion au serveur Flask.
 *
 * IMPORTANT :
 * - Utiliser http:// (PAS https://) — Flask tourne en HTTP sur le port 5000.
 * - Remplacer l'IP par celle du PC qui exécute `python app.py` (même réseau Wi-Fi).
 *   Exemple : ipconfig → Adresse IPv4 (ex. 192.168.137.3)
 */
// IP du PC qui lance "python app.py" (voir les logs Flask au demarrage)
const API_URL = 'http://192.168.137.3:5000';

function ensureHttpUrl(url) {
  const trimmed = (url || '').trim().replace(/\/$/, '');
  if (trimmed.startsWith('https://')) {
    console.warn('HTTPS detecte — remplacement par HTTP:', trimmed);
    return 'http://' + trimmed.slice(8);
  }
  if (!trimmed.startsWith('http://')) {
    return 'http://' + trimmed;
  }
  return trimmed;
}

const API_BASE = ensureHttpUrl(API_URL);

/**
 * Envoie une image au endpoint /detect (multipart/form-data).
 * @param {string} imageUri - URI locale de l'image (Expo ImagePicker)
 * @param {string} plantType - mais | manioc | tomate | riz | pomme de terre (optionnel)
 */
export const detectDisease = async (imageUri, plantType = 'tomate') => {
  const formData = new FormData();

  formData.append('image', {
    uri: imageUri,
    name: 'photo.jpg',
    type: 'image/jpeg',
  });

  const plantMap = {
    'maïs': 'mais',
    maïs: 'mais',
    mais: 'mais',
    manioc: 'manioc',
    tomate: 'tomate',
    riz: 'riz',
    'pomme de terre': 'pomme de terre',
  };
  const normalized = plantMap[plantType] || plantType || 'tomate';
  formData.append('plant_type', normalized);

  console.log('POST', `${API_BASE}/detect`);
  const response = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    body: formData,
  });

  const data = await response.json();

  if (!response.ok) {
    const msg = data?.error || data?.details || `Erreur HTTP ${response.status}`;
    throw new Error(msg);
  }

  console.log('SERVER RESPONSE:', data);
  return data;
};

export const getApiBaseUrl = () => API_BASE;
