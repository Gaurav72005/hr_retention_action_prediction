document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
});

// Clear input textarea
function clearInput() {
    document.getElementById('feedbackText').value = '';
    document.getElementById('emptyState').classList.remove('hidden');
    document.getElementById('loader').classList.add('hidden');
    document.getElementById('resultsContent').classList.add('hidden');
}

// Handle Form Submission & API Request
async function handlePrediction(event) {
    event.preventDefault();

    const textInput = document.getElementById('feedbackText').value.trim();
    if (!textInput) {
        alert('Please enter employee feedback or exit reason text.');
        return;
    }

    // UI Elements
    const emptyState = document.getElementById('emptyState');
    const loader = document.getElementById('loader');
    const resultsContent = document.getElementById('resultsContent');
    const predictBtn = document.getElementById('predictBtn');

    // Show Loader
    emptyState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loader.classList.remove('hidden');
    predictBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: textInput })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Prediction request failed.');
        }

        // Render Results
        renderResults(data);

    } catch (error) {
        console.error('Error during prediction:', error);
        alert(`Prediction Error: ${error.message}`);
        emptyState.classList.remove('hidden');
    } finally {
        loader.classList.add('hidden');
        predictBtn.disabled = false;
    }
}

// Render Prediction Result Data
function renderResults(data) {
    const resultsContent = document.getElementById('resultsContent');
    const actionBadge = document.getElementById('actionBadge');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const strategyTitle = document.getElementById('strategyTitle');
    const strategyText = document.getElementById('strategyText');
    const urgencyTag = document.getElementById('urgencyTag');
    const strategyIcon = document.getElementById('strategyIcon');

    // Set Action Badge
    const prediction = data.prediction;
    const strategy = data.strategy_info || {};

    actionBadge.textContent = prediction;
    actionBadge.className = `action-badge ${strategy.badge_class || 'badge-na'}`;

    // Set Confidence Score
    const confidence = data.confidence || 0;
    confidenceValue.textContent = `${confidence}%`;
    confidenceFill.style.width = `${confidence}%`;

    // Set Strategy Details
    strategyTitle.textContent = strategy.title || 'HR Retention Strategy';
    strategyText.textContent = strategy.strategy || 'No specific strategy recommendations available.';
    urgencyTag.textContent = `Urgency Level: ${strategy.urgency || 'Medium'}`;

    if (strategy.icon) {
        strategyIcon.className = `fa-solid ${strategy.icon}`;
    }

    // Unhide Results
    resultsContent.classList.remove('hidden');
}

// Fetch Analytics Stats from Backend API
async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        if (response.ok) {
            const stats = await response.json();
            if (stats.total_records) {
                document.getElementById('statTotal').textContent = stats.total_records.toLocaleString();
            }
            if (stats.attrition_count) {
                document.getElementById('statAttrition').textContent = stats.attrition_count.toLocaleString();
            }
        }
    } catch (err) {
        console.log('Stats endpoint fetch info:', err);
    }
}
