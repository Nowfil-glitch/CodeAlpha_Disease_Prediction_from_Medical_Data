import numpy as np
import pandas as pd
import os

def generate_medical_dataset(n_samples=2000, seed=42):
    np.random.seed(seed)
    
    age = np.random.randint(25, 80, size=n_samples)
    gender = np.random.choice([0, 1], size=n_samples) # 0: Female, 1: Male
    chest_pain_type = np.random.choice([0, 1, 2, 3], size=n_samples) # 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic
    resting_bp = np.random.normal(130, 18, size=n_samples).clip(90, 200)
    cholesterol = np.random.normal(240, 45, size=n_samples).clip(120, 450)
    fasting_bs = np.random.choice([0, 1], p=[0.8, 0.2], size=n_samples) # >120 mg/dl
    max_hr = np.random.normal(150, 22, size=n_samples).clip(70, 210)
    exercise_angina = np.random.choice([0, 1], p=[0.7, 0.3], size=n_samples)
    oldpeak = np.random.exponential(scale=1.0, size=n_samples).clip(0, 6.0)
    glucose = np.random.normal(115, 35, size=n_samples).clip(70, 300)
    bmi = np.random.normal(28, 6, size=n_samples).clip(16, 50)
    
    # Calculate Heart Disease Risk score
    heart_risk = (
        (age / 100) * 1.5 +
        (resting_bp / 120) * 1.0 +
        (cholesterol / 200) * 1.2 +
        exercise_angina * 2.0 +
        oldpeak * 1.5 +
        fasting_bs * 1.2 -
        (max_hr / 200) * 1.0 +
        np.random.normal(0, 0.8, size=n_samples)
    )
    heart_disease = (heart_risk > 3.2).astype(int)
    
    # Calculate Diabetes Risk score
    diabetes_risk = (
        (glucose / 100) * 2.5 +
        (bmi / 25) * 1.8 +
        (age / 50) * 0.8 +
        np.random.normal(0, 0.6, size=n_samples)
    )
    diabetes = (diabetes_risk > 4.1).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'chest_pain_type': chest_pain_type,
        'resting_bp': np.round(resting_bp, 1),
        'cholesterol': np.round(cholesterol, 1),
        'fasting_bs': fasting_bs,
        'max_hr': np.round(max_hr, 1),
        'exercise_angina': exercise_angina,
        'oldpeak': np.round(oldpeak, 2),
        'glucose': np.round(glucose, 1),
        'bmi': np.round(bmi, 1),
        'heart_disease': heart_disease,
        'diabetes': diabetes
    })
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(out_dir, 'medical_data.csv')
    df.to_csv(file_path, index=False)
    print(f"Medical dataset with {n_samples} samples generated to {file_path}")
    return df

if __name__ == '__main__':
    generate_medical_dataset()
