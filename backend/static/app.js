
const imageInput = document.getElementById('imageInput');
const chooseImageBtn = document.getElementById('chooseImageBtn');
const cameraBtn = document.getElementById('cameraBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const fileLabel = document.getElementById('fileLabel');
const loadingState = document.getElementById('loadingState');
const resultCard = document.getElementById('resultCard');
const resultName = document.getElementById('resultName');
const resultConfidence = document.getElementById('resultConfidence');
const resultDescription = document.getElementById('resultDescription');
const resultCauses = document.getElementById('resultCauses');
const resultTreatment = document.getElementById('resultTreatment');
const errorMessage = document.getElementById('errorMessage');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const plantTypeSelect = document.getElementById('plantTypeSelect');
const uploadZone = document.getElementById('uploadZone');

// Caméra vs Fichier selection trigger
if (cameraBtn) {
    cameraBtn.addEventListener('click', () => {
        imageInput.setAttribute('capture', 'environment');
        imageInput.click();
    });
}

if (chooseImageBtn) {
    chooseImageBtn.addEventListener('click', () => {
        imageInput.removeAttribute('capture');
        imageInput.click();
    });
}

if (uploadZone) {
    uploadZone.addEventListener('click', () => {
        imageInput.removeAttribute('capture');
        imageInput.click();
    });
}

imageInput.addEventListener('change', (event) => {
    handleImageSelect(event);
});

analyzeBtn.addEventListener('click', analyzeImage);
newAnalysisBtn.addEventListener('click', resetAnalysis);

function handleImageSelect(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    if (!file.type.startsWith('image/')) {
        showError('Veuillez sélectionner une image valide.');
        return;
    }

    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('L’image doit rester inférieure à 10 Mo.');
        return;
    }

    hideError();
    hideResults();

    // Afficher l'aperçu de l'image
    const reader = new FileReader();
    reader.onload = function(e) {
        const imagePreview = document.getElementById('imagePreview');
        const uploadPlaceholder = document.getElementById('uploadPlaceholder');
        if (imagePreview && uploadPlaceholder) {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            uploadPlaceholder.classList.add('hidden');
        }
    };
    reader.readAsDataURL(file);

    fileLabel.textContent = `Image sélectionnée : ${file.name}`;
    fileLabel.classList.remove('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.classList.add('active');
}

async function analyzeImage() {
    if (!imageInput.files.length) {
        showError('Veuillez choisir une image avant d’analyser.');
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.classList.remove('active');
    showError('');
    showLoading(true);

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    if (plantTypeSelect) {
        formData.append('plant_type', plantTypeSelect.value);
    }

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const message = errorData?.error || `Erreur serveur ${response.status}`;
            throw new Error(message);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError(`Analyse échouée : ${error.message}`);
    } finally {
        showLoading(false);
        analyzeBtn.disabled = false;
        analyzeBtn.classList.add('active');
    }
}

function displayResults(data) {
    const detection = data.detection || {};

    // Afficher nom de la maladie et nom scientifique si disponible
    if (detection.scientific_name) {
        resultName.innerHTML = `${detection.disease_name || 'Maladie détectée'} <span class="scientific-name">(${detection.scientific_name})</span>`;
    } else {
        resultName.textContent = detection.disease_name || 'Maladie détectée';
    }

    // Gestion intelligente du score de confiance
    const confidenceValue = detection.confidence_score != null
        ? Number(detection.confidence_score)
        : Number(detection.confidence || 0);
    
    let pct = confidenceValue;
    if (pct <= 1.0) {
        pct = pct * 100;
    }
    pct = Math.round(pct);
    
    resultConfidence.textContent = `${pct}%`;
    const confidenceBar = document.getElementById('confidenceBar');
    if (confidenceBar) {
        confidenceBar.style.width = `${pct}%`;
    }

    // Badge de sévérité si disponible
    const severityBadge = document.getElementById('severityBadge');
    if (severityBadge) {
        const severity = data.analysis?.severity || detection.severity || 'Diagnostic IA';
        severityBadge.textContent = `🧬 Sévérité : ${severity}`;
    }

    // Afficher descriptions/symptômes, causes et traitements
    resultDescription.textContent = detection.symptoms || detection.description || 'Symptômes non disponibles.';
    resultCauses.textContent = detection.causes || 'Informations non disponibles.';
    
    let treatmentText = '';
    if (detection.treatment) {
        treatmentText += `Traitement : ${detection.treatment}`;
    }
    if (detection.prevention) {
        if (treatmentText) treatmentText += '\n\n';
        treatmentText += `Prévention : ${detection.prevention}`;
    }
    resultTreatment.textContent = treatmentText || 'Aucun traitement ou mesure de prévention spécifié.';

    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetAnalysis() {
    imageInput.value = '';
    fileLabel.textContent = '';
    fileLabel.classList.add('hidden');
    
    const imagePreview = document.getElementById('imagePreview');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    if (imagePreview && uploadPlaceholder) {
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
    }

    analyzeBtn.disabled = true;
    analyzeBtn.classList.remove('active');
    hideResults();
    showError('');
}

function showLoading(visible) {
    if (loadingState) {
        loadingState.classList.toggle('hidden', !visible);
        if (visible) {
            loadingState.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
}

function hideResults() {
    if (resultCard) {
        resultCard.classList.add('hidden');
    }
}

function showError(message) {
    if (!message) {
        if (errorMessage) {
            errorMessage.classList.add('hidden');
            errorMessage.textContent = '';
        }
        return;
    }

    if (errorMessage) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// Initialisation
if (loadingState) loadingState.classList.add('hidden');
if (resultCard) resultCard.classList.add('hidden');
if (errorMessage) errorMessage.classList.add('hidden');
if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.classList.remove('active');
}
console.log('✅ UI app initialized');
