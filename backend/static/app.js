
const imageInput = document.getElementById('imageInput');
const chooseImageBtn = document.getElementById('chooseImageBtn');
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

chooseImageBtn.addEventListener('click', () => {
    imageInput.click();
});

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
    fileLabel.textContent = `Fichier sélectionné : ${file.name}`;
    fileLabel.classList.remove('hidden');
    analyzeBtn.disabled = false;
}

async function analyzeImage() {
    if (!imageInput.files.length) {
        showError('Veuillez choisir une image avant d’analyser.');
        return;
    }

    analyzeBtn.disabled = true;
    showError('');
    showLoading(true);

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

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
    }
}

function displayResults(data) {
    const detection = data.detection || {};

    resultName.textContent = detection.disease_name || 'Maladie détectée';
    const confidenceValue = detection.confidence_score != null
        ? Number(detection.confidence_score)
        : Number(detection.confidence || 0);
    resultConfidence.textContent = `${Math.round(confidenceValue * 100)} %`;
    resultDescription.textContent = detection.description || 'Description non disponible.';
    resultCauses.textContent = detection.causes || 'Informations non disponibles.';
    resultTreatment.textContent = detection.treatment || 'Aucun traitement spécifié.';

    resultCard.classList.remove('hidden');
}

function resetAnalysis() {
    imageInput.value = '';
    fileLabel.textContent = '';
    fileLabel.classList.add('hidden');
    analyzeBtn.disabled = true;
    hideResults();
    showError('');
}

function showLoading(visible) {
    loadingState.classList.toggle('hidden', !visible);
}

function hideResults() {
    resultCard.classList.add('hidden');
}

function showError(message) {
    if (!message) {
        errorMessage.classList.add('hidden');
        errorMessage.textContent = '';
        return;
    }

    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

showLoading(false);
hideResults();
showError('');
analyzeBtn.disabled = true;
