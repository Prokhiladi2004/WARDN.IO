import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate(model, X_test, y_test):
    st.subheader("📊 Model Performance Analysis")
    
    # 1. Get Predictions
    y_pred = model.predict(X_test)
    
    # 2. Display Metrics (Precision, Recall, F1-Score)
    st.text("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'))

    # 3. Plot Confusion Matrix
    # Inside the evaluate function...
    st.write("### 🔍 What is the AI looking at?")
    importances = model.feature_importances_
    feature_names = X_test.columns
    
    # Create a DataFrame for plotting
    feat_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False).head(10)
    
    # Plotting
    fig2, ax2 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feat_importance_df, palette='viridis')
    st.pyplot(fig2)