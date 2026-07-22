import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support

from dataset_generator import generate_medical_dataset

def train_medical_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'medical_data.csv')
    
    if not os.path.exists(csv_path):
        df = generate_medical_dataset()
    else:
        df = pd.read_csv(csv_path)
        
    print(f"Medical dataset shape: {df.shape}")
    
    feature_cols = ['age', 'gender', 'chest_pain_type', 'resting_bp', 'cholesterol', 
                    'fasting_bs', 'max_hr', 'exercise_angina', 'oldpeak', 'glucose', 'bmi']
    
    X = df[feature_cols]
    y_heart = df['heart_disease']
    y_diabetes = df['diabetes']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Heart Disease Model
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_scaled, y_heart, test_size=0.2, random_state=42, stratify=y_heart)
    
    print("\n--- Training Heart Disease Prediction Models ---")
    h_models = {
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    best_heart_model = None
    best_h_score = 0
    best_h_name = ""
    
    for name, model in h_models.items():
        model.fit(X_train_h, y_train_h)
        preds = model.predict(X_test_h)
        probs = model.predict_proba(X_test_h)[:, 1]
        _, _, f1, _ = precision_recall_fscore_support(y_test_h, preds, average='binary')
        auc = roc_auc_score(y_test_h, probs)
        print(f"[{name}] F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
        if f1 > best_h_score:
            best_h_score = f1
            best_heart_model = model
            best_h_name = name
            
    print(f"Selected Best Heart Disease Model: {best_h_name}")
    
    # Train Diabetes Model
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_scaled, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes)
    
    print("\n--- Training Diabetes Prediction Models ---")
    d_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    best_diabetes_model = None
    best_d_score = 0
    best_d_name = ""
    
    for name, model in d_models.items():
        model.fit(X_train_d, y_train_d)
        preds = model.predict(X_test_d)
        probs = model.predict_proba(X_test_d)[:, 1]
        _, _, f1, _ = precision_recall_fscore_support(y_test_d, preds, average='binary')
        auc = roc_auc_score(y_test_d, probs)
        print(f"[{name}] F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
        if f1 > best_d_score:
            best_d_score = f1
            best_diabetes_model = model
            best_d_name = name
            
    print(f"Selected Best Diabetes Model: {best_d_name}")
    
    # Save artifacts
    joblib.dump(best_heart_model, os.path.join(base_dir, 'heart_disease_model.pkl'))
    joblib.dump(best_diabetes_model, os.path.join(base_dir, 'diabetes_model.pkl'))
    joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(base_dir, 'feature_cols.pkl'))
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm_h = confusion_matrix(y_test_h, best_heart_model.predict(X_test_h))
    sns.heatmap(cm_h, annot=True, fmt='d', cmap='Reds', ax=axes[0])
    axes[0].set_title(f'Heart Disease CM ({best_h_name})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    cm_d = confusion_matrix(y_test_d, best_diabetes_model.predict(X_test_d))
    sns.heatmap(cm_d, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title(f'Diabetes CM ({best_d_name})')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    print(f"Model artifacts successfully saved to {base_dir}")

if __name__ == '__main__':
    train_medical_models()
