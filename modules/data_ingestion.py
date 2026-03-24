import pandas as pd
import streamlit as st


def load_data(uploaded_file) -> pd.DataFrame | None:
    """Load a CSV or Parquet file into a DataFrame."""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("❌ Only CSV or Parquet files are supported.")
            return None

        st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(5), use_container_width=True)
        return df
    except Exception as exc:
        st.error(f"❌ Failed to load file: {exc}")
        return None
