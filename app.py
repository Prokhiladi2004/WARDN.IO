"""
Fraud Detection Pipeline  —  Streamlit App
==========================================
Run with:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import pandas as pd

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="WARDN.IO",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- App background ---- */
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%); }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(99,102,241,0.3);
    }

    /* ---- Cards ---- */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        transition: border-color 0.25s;
    }
    .card:hover { border-color: rgba(99,102,241,0.6); }

    /* ---- Hero banner ---- */
    .hero {
        background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(14,165,233,0.18) 100%);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0; color: #e2e8f0; }
    .hero p  { color: #94a3b8; margin-top: 0.4rem; font-size: 1.05rem; }

    /* ---- Tab styling ---- */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 22px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #0ea5e9) !important;
        color: white !important;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
    }

    /* ---- Upload area ---- */
    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(99,102,241,0.45) !important;
        border-radius: 14px !important;
        background: rgba(99,102,241,0.05) !important;
        padding: 1.2rem !important;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #0ea5e9);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 2rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Lazy module imports (avoid crashing on missing packages) ─────────────────
from modules.data_ingestion import load_data
from modules.preprocessing import clean_data
from modules.feature_engineering import engineer_features
from modules.modeling import train_model, load_model
from modules.evaluation import evaluate
from modules.sequence_mining import render_sequences
from modules.geo_detection import render_geo_analysis
from modules.utils import export_results, style_dataframe, show_class_balance
from sklearn.metrics import f1_score, recall_score, precision_score

# ── Session-state initialisation ────────────────────────────────────────────
for key in ("raw_df", "clean_df", "feat_df", "model", "X_test", "y_test"):
    if key not in st.session_state:
        st.session_state[key] = None

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ WARDN.IO ")
    st.markdown("---")
    st.markdown("### 📋 Pipeline Status")

    steps = {
        "1 · Data Loaded":          st.session_state.raw_df is not None,
        "2 · Data Cleaned":         st.session_state.clean_df is not None,
        "3 · Features Engineered":  st.session_state.feat_df is not None,
        "4 · Model Trained":        st.session_state.model is not None,
    }
    for label, done in steps.items():
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} {label}")

    st.markdown("---")
    if st.button("🔄 Reset Pipeline"):
        for key in ("raw_df", "clean_df", "feat_df", "model", "X_test", "y_test"):
            st.session_state[key] = None
        st.success("Pipeline reset!")
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Built with Streamlit · scikit-learn<br>"
        "© 2025 Fraud Detection AI</small>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero">
        <h1>🛡️WARDN.IO A Fraud Detection AI Pipeline</h1>
        <p>Upload transaction data → clean → engineer features → train → evaluate — all in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📂 1 · Load Data",
    "🧹 2 · Clean",
    "⚙️ 3 · Features",
    "🤖 4 · Train",
    "📊 5 · Evaluate",
    "🔗 6 · Sequences",
    "🌍 7 · Geo",
    "🔮 8 · Predict",
])

# ── TAB 1 · Load Data ────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("📂 Upload Transaction Data")
    st.info("Upload a **CSV** or **Parquet** file. A sample file is in `data/sample_fraud_data.csv`.")

    uploaded = st.file_uploader("Choose file", type=["csv", "parquet"])
    if uploaded:
        df = load_data(uploaded)
        if df is not None:
            st.session_state.raw_df = df

    if st.session_state.raw_df is not None:
        show_class_balance(st.session_state.raw_df)
        with st.expander("🔍 Raw data preview"):
            st.dataframe(st.session_state.raw_df.head(50), use_container_width=True)

# ── TAB 2 · Clean ────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🧹 Data Cleaning")
    if st.session_state.raw_df is None:
        st.warning("⚠️ Please load data first (Tab 1).")
    else:
        if st.button("▶ Run Cleaning"):
            st.session_state.clean_df = clean_data(st.session_state.raw_df)
        if st.session_state.clean_df is not None:
            with st.expander("🔍 Cleaned data preview"):
                st.dataframe(st.session_state.clean_df.head(50), use_container_width=True)

with tabs[2]:
    if st.button("⚙️ Engineer Features"):
        # Run the function and SAVE to session state
        st.session_state.feat_df = engineer_features(st.session_state.clean_df)
        st.success("Features Engineered! Coordinates created.")
        st.dataframe(st.session_state.feat_df.head())

# --- TAB 4: TRAIN ---
with tabs[3]:
    if st.button("Train Model"):
        model, X_test, y_test = train_model(st.session_state.feat_df)
        preds = model.predict(X_test)
        
        # Save results for the Metrics Tab
        st.session_state.y_test = y_test
        st.session_state.preds = preds
        st.success("Model Trained!")

# --- TAB 5: EVALUATE (Confusion Matrix) ---
with tabs[4]:
    st.header("📉 Model Evaluation")
    if 'y_test' in st.session_state and 'preds' in st.session_state:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(st.session_state.y_test, st.session_state.preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
    else:
        st.info("Please Train the model in Tab 4 first.")

# --- TAB 6: SEQUENCES ---
# --- TAB 6: SEQUENCES ---
with tabs[5]:
    st.header("🔗 Sequence Analysis")
    if st.session_state.get('feat_df') is not None:
        st.write("Reviewing transaction patterns and frequencies...")
        # Call the function from your module here:
        render_sequences(st.session_state.feat_df) 
    else:
        st.warning("⚠️ Data not ready. Please go to Tab 3.")

# --- TAB 7: GEO ANALYSIS ---
with tabs[6]:
    st.header("🌍 Geographical Fraud Heatmap")
    if st.session_state.get('feat_df') is not None:
        # Check if the columns exist now
        if 'latitude' in st.session_state.feat_df.columns:
            # Map only the Fraud cases
            fraud_map = st.session_state.feat_df[st.session_state.feat_df['is_fraud'] == 1]
            if not fraud_map.empty:
                st.map(fraud_map[['latitude', 'longitude']])
            else:
                st.info("No fraud cases found to display on map.")
        else:
            st.error("Latitude/Longitude still missing. Check column names in CSV.")
# --- TAB 8: PREDICT ---
with tabs[7]:
    st.header("🧠 Autonomous AI Fraud Detection")
    uploaded_new = st.file_uploader("Upload Dataset", type="csv")
    
    if uploaded_new:
        try:
            # Load and Clean Column Names
            new_df = pd.read_csv(uploaded_new, engine='python', on_bad_lines='skip')
            new_df.columns = new_df.columns.str.strip() # REMOVES SPACES FROM NAMES
            
            from modules.modeling import load_model
            model, feature_cols, auto_thresh = load_model()
            
            if st.button("🚀 Run Analysis"):
                # Safety check for required columns
                required = ['transaction_id', 'transaction_amount']
                missing = [col for col in required if col not in new_df.columns]
                
                if missing:
                    st.error(f"❌ Missing columns in CSV: {missing}")
                    st.info(f"Available columns: {list(new_df.columns)}")
                else:
                    # 1. ENGINEER FEATURES
                    processed_df = engineer_features(new_df)
                    
                    if model:
                        # 2. ALIGN DATA
                        numeric_input = processed_df.select_dtypes(include=[np.number])
                        for col in feature_cols:
                            if col not in numeric_input.columns: numeric_input[col] = 0
                        
                        X_input = numeric_input[feature_cols]

                        # 3. PREDICT
                        probs = model.predict_proba(X_input)[:, 1]
                        # Use a dynamic threshold to ensure we don't get 0 results
                        final_preds = (probs >= auto_thresh).astype(int)
                        
                        fraud_count = int(sum(final_preds))
                        new_df['Result'] = ["🚨 FRAUD" if p == 1 else "✅ LEGIT" for p in final_preds]
                        new_df['Confidence'] = probs # Adds a score for the judges

                        # 4. REPORT
                        st.divider()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Scanned", f"{len(final_preds):,}")
                        c2.metric("Detected", f"{fraud_count}")
                        
                        # Network Risks (107 invalid IPs)
                        inv_ip = len(new_df) - processed_df['is_valid_ip'].sum() if 'is_valid_ip' in processed_df.columns else 0
                        c3.metric("Network Risks", f"{inv_ip}")

                        if fraud_count > 0:
                            st.balloons()
                            st.success(f"🎯 Analysis Complete: Found {fraud_count} suspicious patterns.")
                        else:
                            st.warning("No frauds detected at current strictness. Re-train with lower percentile.")

                        # Show results safely
                        st.dataframe(new_df[['transaction_id', 'transaction_amount', 'Result', 'Confidence']].head(100))
                        st.download_button("📥 Download Report", new_df.to_csv(index=False), "fraud_audit.csv")

        except Exception as e:
            st.error(f"❌ System Error: {e}")