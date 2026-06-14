
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

// Event Handlers helper for buttons to prevent default actions
function bindClick(btn, handler) {
    if (btn) {
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            handler(event);
        });
    }
}

// Caméra vs Fichier selection triggers
bindClick(cameraBtn, () => {
    imageInput.setAttribute('capture', 'environment');
    imageInput.click();
});

bindClick(chooseImageBtn, () => {
    imageInput.removeAttribute('capture');
    imageInput.click();
});

bindClick(uploadZone, () => {
    imageInput.removeAttribute('capture');
    imageInput.click();
});

// Drag and drop event listeners on uploadZone
if (uploadZone) {
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.remove('dragover');
        }, false);
    });

    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            // Assign the files to file input
            try {
                imageInput.files = files;
            } catch (err) {
                console.warn('Assignment to file input files failed: ', err);
            }
            handleImageSelect(files[0]);
        }
    }, false);
}

// Bind change handler on file input
imageInput.addEventListener('change', (event) => {
    if (event.target.files && event.target.files.length > 0) {
        handleImageSelect(event.target.files[0]);
    }
});

bindClick(analyzeBtn, analyzeImage);
bindClick(newAnalysisBtn, resetAnalysis);

// Process and display selected image preview
function handleImageSelect(file) {
    if (!file) {
        return;
    }

    // Validate that file is indeed an image
    if (!file.type.startsWith('image/')) {
        showError('Veuillez sélectionner une image valide.');
        // Reset preview if invalid file selected
        resetImagePreview();
        return;
    }

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showError('L’image doit rester inférieure à 10 Mo.');
        resetImagePreview();
        return;
    }

    hideError();
    hideResults();

    // Use FileReader to display preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const imagePreview = document.getElementById('imagePreview');
        const uploadPlaceholder = document.getElementById('uploadPlaceholder');
        if (imagePreview && uploadPlaceholder) {
            // Smoothly swap classes for transition
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            uploadPlaceholder.classList.add('hidden');
        }
    };
    reader.onerror = function() {
        showError('Impossible de lire le fichier image.');
        resetImagePreview();
    };
    reader.readAsDataURL(file);

    if (fileLabel) {
        fileLabel.textContent = `Image sélectionnée : ${file.name}`;
        fileLabel.classList.remove('hidden');
    }

    // Enable the analyze button and activate glowing pulse animation
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.add('active');
    }
}

// Helper to reset preview area to default empty card
function resetImagePreview() {
    const imagePreview = document.getElementById('imagePreview');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    if (imagePreview && uploadPlaceholder) {
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
    }
    if (fileLabel) {
        fileLabel.textContent = '';
        fileLabel.classList.add('hidden');
    }
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.classList.remove('active');
    }
}

async function analyzeImage() {
    const file = imageInput.files[0];
    if (!file) {
        showError('Veuillez choisir une image avant d’analyser.');
        return;
    }

    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.classList.remove('active');
    }
    showError('');
    showLoading(true);

    const formData = new FormData();
    formData.append('image', file);
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
        if (imageInput.files.length > 0) {
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
                analyzeBtn.classList.add('active');
            }
        }
    }
}

function displayResults(data) {
    const detection = data.detection || {};

    // Afficher nom de la maladie et nom scientifique si disponible
    if (resultName) {
        if (detection.scientific_name) {
            resultName.innerHTML = `${detection.disease_name || 'Maladie détectée'} <span class="scientific-name">(${detection.scientific_name})</span>`;
        } else {
            resultName.textContent = detection.disease_name || 'Maladie détectée';
        }
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
    
    if (resultConfidence) {
        resultConfidence.textContent = `${pct}%`;
    }
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
    if (resultDescription) {
        resultDescription.textContent = detection.symptoms || detection.description || 'Symptômes non disponibles.';
    }
    if (resultCauses) {
        resultCauses.textContent = detection.causes || 'Informations non disponibles.';
    }
    
    if (resultTreatment) {
        let treatmentText = '';
        if (detection.treatment) {
            treatmentText += `Traitement : ${detection.treatment}`;
        }
        if (detection.prevention) {
            if (treatmentText) treatmentText += '\n\n';
            treatmentText += `Prévention : ${detection.prevention}`;
        }
        resultTreatment.textContent = treatmentText || 'Aucun traitement ou mesure de prévention spécifié.';
    }

    if (resultCard) {
        resultCard.classList.remove('hidden');
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function resetAnalysis() {
    imageInput.value = '';
    resetImagePreview();
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

function hideError() {
    showError('');
}

// Initialisation
if (loadingState) loadingState.classList.add('hidden');
if (resultCard) resultCard.classList.add('hidden');
if (errorMessage) errorMessage.classList.add('hidden');
if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.classList.remove('active');
}
console.log('✅ UI application events loaded');

// ---- PWA: Service Worker registration and Install prompt handling ----
let deferredInstallPrompt = null;
const installBtn = document.getElementById('installBtn');

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredInstallPrompt = e;
    if (installBtn) {
        installBtn.classList.remove('hidden');
    }
});

if (installBtn) {
    installBtn.addEventListener('click', async () => {
        if (!deferredInstallPrompt) return;
        installBtn.disabled = true;
        try {
            deferredInstallPrompt.prompt();
            const choice = await deferredInstallPrompt.userChoice;
            if (choice.outcome === 'accepted') {
                console.log('PWA install accepted');
            } else {
                console.log('PWA install dismissed');
            }
        } catch (err) {
            console.warn('Install prompt failed:', err);
        } finally {
            installBtn.classList.add('hidden');
            deferredInstallPrompt = null;
            installBtn.disabled = false;
        }
    });
}

// Service worker registration is performed in the template using Flask `url_for`.
if ('serviceWorker' in navigator) {
    // placeholder: registration handled in server-rendered template
}

