// 04_outputs/static/app.js

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const form = document.getElementById("analysis-form");
    const sentenceInput = document.getElementById("sentence-input");
    const daysSlider = document.getElementById("days-slider");
    const daysVal = document.getElementById("days-val");
    const formatSelect = document.getElementById("format-select");
    const sessionSelect = document.getElementById("session-select");
    const clientSelect = document.getElementById("client-select");
    const presetSelect = document.getElementById("preset-select");
    
    const heatmapContainer = document.getElementById("heatmap-container");
    const detailsContent = document.getElementById("details-content");
    
    let activePredictions = [];

    // Predefined mock datasets matching sample_exercises.json
    const PRESETS = [
        {
            sentence: "He works in a bank.",
            format: "listen",
            session: "practice",
            client: "android",
            days: 10
        },
        {
            sentence: "She plays soccer.",
            format: "reverse_translate",
            session: "lesson",
            client: "web",
            days: 1
        },
        {
            sentence: "They run.",
            format: "reverse_tap",
            session: "test",
            client: "ios",
            days: 20
        }
    ];

    // Update Days val text dynamically
    daysSlider.addEventListener("input", (e) => {
        daysVal.textContent = `${e.target.value} days`;
    });

    // Preset selection change listener
    presetSelect.addEventListener("change", (e) => {
        const val = e.target.value;
        if (val !== "custom") {
            const index = parseInt(val, 10);
            const data = PRESETS[index];
            
            sentenceInput.value = data.sentence;
            formatSelect.value = data.format;
            sessionSelect.value = data.session;
            clientSelect.value = data.client;
            daysSlider.value = data.days;
            daysVal.textContent = `${data.days} days`;
            
            // Auto submit
            submitAnalysis();
        }
    });

    // Form submission listener
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        presetSelect.value = "custom"; // Shift back to custom if manually submitting
        submitAnalysis();
    });

    // Core Submit prediction method
    async function submitAnalysis() {
        const sentence = sentenceInput.value.trim();
        if (!sentence) return;

        const requestData = {
            sentence: sentence,
            exercise_format: formatSelect.value,
            session_type: sessionSelect.value,
            client: clientSelect.value,
            days_in_course: parseFloat(daysSlider.value)
        };

        try {
            heatmapContainer.innerHTML = '<p class="empty-state">Analyzing sentence features...</p>';
            
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`Server returned error: ${response.statusText}`);
            }

            const data = await response.json();
            activePredictions = data.predictions;
            
            renderHeatmap(activePredictions);
            
            // Auto-select first token
            if (activePredictions.length > 0) {
                selectToken(0);
            }

        } catch (error) {
            console.error(error);
            heatmapContainer.innerHTML = `<p class="empty-state" style="color: var(--risk-high)">Error running inference: ${error.message}</p>`;
        }
    }

    function getRiskClass(prob) {
        if (prob < 0.10) return "risk-low";
        if (prob < 0.20) return "risk-med";
        return "risk-high";
    }

    function renderHeatmap(predictions) {
        heatmapContainer.innerHTML = "";
        
        predictions.forEach((pred, idx) => {
            const tokenSpan = document.createElement("span");
            tokenSpan.textContent = pred.token;
            tokenSpan.className = `heatmap-token ${getRiskClass(pred.lr_prob)}`;
            tokenSpan.id = `token-${idx}`;
            
            tokenSpan.addEventListener("click", () => {
                selectToken(idx);
            });
            
            heatmapContainer.appendChild(tokenSpan);
        });
    }

    function selectToken(index) {
        // Clear active classes
        document.querySelectorAll(".heatmap-token").forEach(span => {
            span.classList.remove("active");
        });
        
        // Highlight chosen token
        const selectedSpan = document.getElementById(`token-${index}`);
        if (selectedSpan) {
            selectedSpan.classList.add("active");
        }
        
        const pred = activePredictions[index];
        if (!pred) return;
        
        // Render detailed diagnostics
        const lrRiskClass = getRiskClass(pred.lr_prob);
        const dtRiskClass = getRiskClass(pred.dt_prob);
        const lrRiskLabel = pred.lr_prob < 0.10 ? "LOW" : pred.lr_prob < 0.20 ? "MED" : "HIGH";
        
        // Build diagnostic explainers
        let explHtml = "";
        if (pred.explanations.length > 0) {
            pred.explanations.forEach(ex => {
                explHtml += `<div class="expl-pill">${ex}</div>`;
            });
        } else {
            explHtml = '<div class="expl-pill" style="border-left-color: var(--text-muted)">Baseline acquisition risk (No structural L1 transfer clashes triggered).</div>';
        }

        detailsContent.innerHTML = `
            <div class="details-grid">
                <div>
                    <div class="details-header">
                        <div class="selected-token-text">${pred.token}</div>
                        <div class="pos-tag">${pred.pos.toUpperCase()} &bull; INDEX ${pred.order}</div>
                    </div>
                    
                    <div class="prob-metrics">
                        <div class="prob-row">
                            <span class="prob-title">Logistic Regression Prob</span>
                            <span class="prob-value ${lrRiskClass}">${(pred.lr_prob * 100).toFixed(1)}%</span>
                        </div>
                        <div class="prob-row">
                            <span class="prob-title">Decision Tree Prob</span>
                            <span class="prob-value ${dtRiskClass}">${(pred.dt_prob * 100).toFixed(1)}%</span>
                        </div>
                        <div class="prob-row">
                            <span class="prob-title">Acquisition Risk Level</span>
                            <span class="prob-value ${lrRiskClass}" style="font-weight: 700; letter-spacing: 0.5px;">${lrRiskLabel}</span>
                        </div>
                    </div>
                </div>
                
                <div class="detail-explanations">
                    <h4 class="explanation-title">Pedagogical Explanations</h4>
                    ${explHtml}
                </div>
            </div>
        `;
    }
    
    // Run initial analysis automatically on load
    submitAnalysis();
});
