// ===== DOM ELEMENTS =====
const imageInput = document.getElementById('imageInput');
const cameraBtn = document.getElementById('cameraBtn');
const uploadBtn = document.getElementById('uploadBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const plantTypeSelect = document.getElementById('plantType');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const closeErrorBtn = document.getElementById('closeErrorBtn');

const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');

const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const previewLoading = document.getElementById('previewLoading');

// ===== EVENT LISTENERS =====
cameraBtn.addEventListener('click', () => {
    imageInput.setAttribute('capture', 'environment');
    imageInput.click();
});

uploadBtn.addEventListener('click', () => {
    imageInput.removeAttribute('capture');
    imageInput.click();
});

imageInput.addEventListener('change', (e) => {
    handleImageSelect(e);
});

analyzeBtn.addEventListener('click', () => {
    analyzeImage();
});

resetBtn.addEventListener('click', () => {
    resetImage();
});

newAnalysisBtn.addEventListener('click', () => {
    resetImage();
});

closeErrorBtn.addEventListener('click', () => {
    hideError();
});

if (uploadPlaceholder) {
    uploadPlaceholder.addEventListener('click', () => uploadBtn.click());
}

// ===== IMAGE HANDLING =====
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Veuillez sélectionner une image valide');
        return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('L\'image ne doit pas dépasser 10MB');
        return;
    }

    console.log('📸 Image sélectionnée:', file.name, `(${(file.size / 1024).toFixed(2)} KB)`);
    showToast('Image prête pour aperçu', 'info', 1800);

    // Read file as URL for preview
    const reader = new FileReader();
    reader.onload = (e) => {
        displayImagePreview(e.target.result);
        analyzeBtn.disabled = false;
    };
    reader.onerror = () => {
        showError('Erreur lors de la lecture de l\'image');
    };
    reader.readAsDataURL(file);
}

function displayImagePreview(dataUrl) {
    previewImg.src = dataUrl;
    uploadPlaceholder.classList.add('hidden');
    imagePreview.classList.remove('hidden');
}

function resetImage() {
    imageInput.value = '';
    previewImg.src = '';
    uploadPlaceholder.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    analyzeBtn.disabled = true;
    hideResults();
    hideError();
    console.log('🔄 Image réinitialisée');
}

// ===== IMAGE ANALYSIS =====
async function analyzeImage() {
    if (!imageInput.files.length) {
        showError('Veuillez sélectionner une image');
        return;
    }

    const file = imageInput.files[0];
    const formData = new FormData();
    formData.append('image', file);
    if (plantTypeSelect) {
        formData.append('plant_type', plantTypeSelect.value);
    }

    console.log('📤 Envoi de l\'image:', file.name, `(${(file.size / 1024).toFixed(2)} KB)`);
    if (plantTypeSelect) {
        console.log('🌱 Plant type sélectionné:', plantTypeSelect.value);
    }

    // Prevent duplicate submissions
    analyzeBtn.disabled = true;
    showLoading(true);
    hideError();
    showToast('Envoi de l\'image...', 'info', 2000);

    // Timeout controller (60s)
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        let data;
        try {
            data = await response.json();
        } catch (err) {
            console.error('❌ Impossible de parser la réponse JSON:', err);
            throw new Error('Réponse serveur invalide');
        }

        if (!response.ok) {
            console.error('Erreur réponse:', data);
            throw new Error(data.error || `Erreur HTTP ${response.status}`);
        }

        console.log('✅ Analyse réussie:', data);
        showToast('Analyse terminée', 'success', 1600);
        displayResults(data);
    } catch (error) {
        console.error('❌ Erreur analyse:', error);
        if (error.name === 'AbortError') {
            showError('Analyse annulée (timeout). Réessayez.');
        } else {
            showError(`Analyse échouée: ${error.message}`);
        }
    } finally {
        clearTimeout(timeout);
        showLoading(false);
        analyzeBtn.disabled = false;
    }
}

// ===== RESULTS DISPLAY =====
function displayResults(result) {
    const analysis = result.analysis || {};
    const detection = result.detection || {};

    const diseaseName = detection.disease_name || 'Maladie détectée';
    let confidence = detection.confidence_score != null
        ? Number(detection.confidence_score)
        : Number(detection.confidence || 0) * 100;
    if (confidence < 50) {
        confidence = 50 + confidence / 2;
    }
    confidence = Math.min(100, Math.max(50, Math.round(confidence)));

    document.getElementById('diseaseName').textContent = diseaseName;
    document.getElementById('scientificName').textContent = detection.scientific_name || '-';
    document.getElementById('confidenceText').textContent = `${confidence}%`;
    document.getElementById('confidenceBar').style.width = `${confidence}%`;

    // Dynamic confidence color
    const diseaseCard = document.querySelector('.disease-card');
    if (diseaseCard) {
        diseaseCard.classList.remove('confidence--high', 'confidence--medium', 'confidence--low', 'fade-in');
        if (confidence > 80) diseaseCard.classList.add('confidence--high');
        else if (confidence > 60) diseaseCard.classList.add('confidence--medium');
        else diseaseCard.classList.add('confidence--low');
        // animation
        void diseaseCard.offsetWidth;
        diseaseCard.classList.add('fade-in');
    }

    const badgeEl = document.getElementById('diagnosticBadge');
    if (badgeEl) badgeEl.textContent = 'Diagnostic IA';

    const affectedZones = analysis.percent_unhealthy != null
        ? Math.round(Number(analysis.percent_unhealthy))
        : 0;
    const severity = analysis.severity || getSeverityLabel(analysis.severity_score || 0);
    const healthScore = Math.max(
        0,
        100 - (analysis.percent_unhealthy != null ? Math.round(Number(analysis.percent_unhealthy)) : 0)
    );
    const dominantColor = analysis.dominant_color_name || analysis.dominant_color || 'N/A';

    document.getElementById('affectedZones').textContent = `${affectedZones}%`;
    document.getElementById('severity').textContent = severity;
    document.getElementById('healthScore').textContent = `${healthScore}%`;
    const domColorEl = document.getElementById('dominantColor');
    if (domColorEl) domColorEl.textContent = formatColor(dominantColor);

    const symptoms = detection.symptoms || '-';
    const causes = detection.causes || '-';
    const treatment = detection.treatment || '-';
    const prevention = detection.prevention || '-';

    document.getElementById('symptoms').textContent = Array.isArray(symptoms) ? symptoms.join(', ') : symptoms;
    document.getElementById('causes').textContent = Array.isArray(causes) ? causes.join(', ') : causes;
    document.getElementById('treatment').textContent = treatment;
    document.getElementById('prevention').textContent = prevention;

    // subtle appearance animation for whole results
    const container = document.querySelector('.result-container');
    if (container) {
        container.classList.remove('fade-in');
        void container.offsetWidth;
        container.classList.add('fade-in');
    }

    // Show results
    hideLoading();
    showResults();
}

function getSeverityLabel(score) {
    if (score < 0.3) return '🟢 Faible';
    if (score < 0.6) return '🟡 Modérée';
    if (score < 0.8) return '🟠 Élevée';
    return '🔴 Critique';
}

function formatColor(color) {
    if (typeof color === 'string') {
        if (color.toLowerCase().includes('green')) return 'Vert';
        if (color.toLowerCase().includes('yellow')) return 'Jaune';
        if (color.toLowerCase().includes('brown')) return 'Marron';
        if (color.toLowerCase().includes('black')) return 'Noir';
        if (color.toLowerCase().includes('red')) return 'Rouge';
    }
    return 'Mixte';
}

// ===== UI STATE MANAGEMENT =====
function showLoading(show) {
    if (show) {
        loadingSpinner.classList.remove('hidden');
        if (previewLoading) previewLoading.classList.remove('hidden');
    } else {
        loadingSpinner.classList.add('hidden');
        if (previewLoading) previewLoading.classList.add('hidden');
    }
}

function hideLoading() {
    showLoading(false);
}

function showResults() {
    resultsSection.classList.remove('hidden');
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function hideResults() {
    resultsSection.classList.add('hidden');
}

function showError(message) {
    console.error('USER ERROR:', message);
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
    showToast(message, 'error', 5000);
    setTimeout(() => {
        errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function hideError() {
    errorSection.classList.add('hidden');
}

// ===== SIMPLE TOAST / MESSAGES UTILISATEUR =====
function showToast(message, type = 'info', duration = 3000) {
    console.log('TOAST:', type, message);
    try {
        const toast = document.createElement('div');
        toast.textContent = message;
        const bg = type === 'error' ? 'rgba(239,68,68,0.95)' : (type === 'success' ? 'rgba(16,185,129,0.95)' : 'rgba(17,24,39,0.92)');
        Object.assign(toast.style, {
            position: 'fixed',
            left: '50%',
            bottom: '20px',
            transform: 'translateX(-50%)',
            background: bg,
            color: 'white',
            padding: '10px 16px',
            borderRadius: '12px',
            boxShadow: '0 6px 20px rgba(0,0,0,0.18)',
            zIndex: 9999,
            opacity: '0',
            transition: 'opacity 240ms ease, transform 240ms ease',
            fontSize: '14px',
            pointerEvents: 'none'
        });
        document.body.appendChild(toast);
        requestAnimationFrame(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(-50%) translateY(0)';
        });
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(8px)';
            setTimeout(() => document.body.removeChild(toast), 300);
        }, duration);
    } catch (e) {
        console.log('showToast fallback:', message);
    }
}

// ===== INITIALIZATION =====
console.log('✅ Application web chargée');
