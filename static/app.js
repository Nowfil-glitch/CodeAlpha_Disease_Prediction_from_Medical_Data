document.getElementById('medical-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const payload = {
        age: parseFloat(document.getElementById('age').value),
        gender: parseInt(document.getElementById('gender').value),
        chest_pain_type: parseInt(document.getElementById('chest_pain_type').value),
        resting_bp: parseFloat(document.getElementById('resting_bp').value),
        cholesterol: parseFloat(document.getElementById('cholesterol').value),
        fasting_bs: parseInt(document.getElementById('fasting_bs').value),
        max_hr: parseFloat(document.getElementById('max_hr').value),
        exercise_angina: parseInt(document.getElementById('exercise_angina').value),
        oldpeak: parseFloat(document.getElementById('oldpeak').value),
        glucose: parseFloat(document.getElementById('glucose').value),
        bmi: parseFloat(document.getElementById('bmi').value)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.status === 'success') {
            // Heart Disease Update
            const heartPct = data.heart_disease.risk_percentage;
            document.getElementById('heart-bar').style.width = `${heartPct}%`;
            document.getElementById('heart-val').innerText = `${heartPct}% Risk Probability`;
            
            const heartPill = document.getElementById('heart-pill');
            heartPill.innerText = data.heart_disease.risk_level;
            if (heartPct >= 50) {
                heartPill.style.background = 'rgba(239, 68, 68, 0.2)';
                heartPill.style.color = '#f87171';
                heartPill.style.border = '1px solid #ef4444';
            } else {
                heartPill.style.background = 'rgba(16, 185, 129, 0.2)';
                heartPill.style.color = '#34d399';
                heartPill.style.border = '1px solid #10b981';
            }

            // Diabetes Update
            const diabetesPct = data.diabetes.risk_percentage;
            document.getElementById('diabetes-bar').style.width = `${diabetesPct}%`;
            document.getElementById('diabetes-val').innerText = `${diabetesPct}% Risk Probability`;
            
            const diabetesPill = document.getElementById('diabetes-pill');
            diabetesPill.innerText = data.diabetes.risk_level;
            if (diabetesPct >= 50) {
                diabetesPill.style.background = 'rgba(239, 68, 68, 0.2)';
                diabetesPill.style.color = '#f87171';
                diabetesPill.style.border = '1px solid #ef4444';
            } else {
                diabetesPill.style.background = 'rgba(16, 185, 129, 0.2)';
                diabetesPill.style.color = '#34d399';
                diabetesPill.style.border = '1px solid #10b981';
            }
        }
    } catch (err) {
        console.error("Error predicting disease risk:", err);
    }
});
