import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')
base_dir = os.path.dirname(os.path.abspath(__file__))

heart_model_path = os.path.join(base_dir, 'heart_disease_model.pkl')
if not os.path.exists(heart_model_path):
    from train_model import train_medical_models
    train_medical_models()

heart_model = joblib.load(os.path.join(base_dir, 'heart_disease_model.pkl'))
diabetes_model = joblib.load(os.path.join(base_dir, 'diabetes_model.pkl'))
scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
feature_cols = joblib.load(os.path.join(base_dir, 'feature_cols.pkl'))

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_dict = {
            'age': float(data.get('age', 45)),
            'gender': int(data.get('gender', 1)),
            'chest_pain_type': int(data.get('chest_pain_type', 0)),
            'resting_bp': float(data.get('resting_bp', 130)),
            'cholesterol': float(data.get('cholesterol', 220)),
            'fasting_bs': int(data.get('fasting_bs', 0)),
            'max_hr': float(data.get('max_hr', 150)),
            'exercise_angina': int(data.get('exercise_angina', 0)),
            'oldpeak': float(data.get('oldpeak', 1.0)),
            'glucose': float(data.get('glucose', 110)),
            'bmi': float(data.get('bmi', 26.5))
        }
        
        df = pd.DataFrame([input_dict])[feature_cols]
        scaled_df = scaler.transform(df)
        
        heart_prob = float(heart_model.predict_proba(scaled_df)[0][1]) * 100.0
        heart_pred = int(heart_prob >= 50.0)
        
        diabetes_prob = float(diabetes_model.predict_proba(scaled_df)[0][1]) * 100.0
        diabetes_pred = int(diabetes_prob >= 50.0)
        
        return jsonify({
            'status': 'success',
            'heart_disease': {
                'prediction': heart_pred,
                'risk_percentage': round(heart_prob, 2),
                'risk_level': 'High Risk' if heart_prob >= 50 else 'Low Risk'
            },
            'diabetes': {
                'prediction': diabetes_pred,
                'risk_percentage': round(diabetes_prob, 2),
                'risk_level': 'High Risk' if diabetes_prob >= 50 else 'Low Risk'
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
