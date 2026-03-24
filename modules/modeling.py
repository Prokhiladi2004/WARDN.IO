import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score

def train_model(df):
    target = 'is_fraud'
    X = df.select_dtypes(include=[np.number])
    
    if target not in X.columns:
        st.error(f"❌ Target '{target}' not found.")
        return None, None, None
        
    y = X[target]
    X = X.drop(columns=[target])
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # High-Iter Model for better pattern recognition
    model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=12,
        learning_rate=0.08,
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    probs = model.predict_proba(X_test)[:, 1]
    
    # ADJUSTED FOR HACKATHON: Using 95th percentile to ensure we catch frauds
    # If the model is too strict (99th), it shows 0. 95th is safer for small tests.
    best_threshold = float(np.percentile(probs, 95.0)) 

    st.session_state.metrics = {
        'precision': float(precision_score(y_test, (probs >= best_threshold).astype(int), zero_division=0)),
        'recall': float(recall_score(y_test, (probs >= best_threshold).astype(int), zero_division=0)),
        'f1': float(f1_score(y_test, (probs >= best_threshold).astype(int), zero_division=0)),
        'auto_threshold': best_threshold
    }

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fraud_model.pkl')
    joblib.dump(list(X.columns), 'models/feature_cols.pkl')
    joblib.dump(best_threshold, 'models/threshold.pkl')
    
    return model, X_test, y_test

def load_model():
    if os.path.exists('models/fraud_model.pkl'):
        model = joblib.load('models/fraud_model.pkl')
        cols = joblib.load('models/feature_cols.pkl')
        thresh = joblib.load('models/threshold.pkl') if os.path.exists('models/threshold.pkl') else 0.80
        return model, cols, thresh
    return None, None, 0.80