import io
import pandas as pd
import streamlit as st


def export_results(df: pd.DataFrame, label: str = "results") -> None:
    """Provide a CSV download button for any DataFrame."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(
        label=f"⬇️ Download {label} (CSV)",
        data=buf,
        file_name=f"{label}.csv",
        mime="text/csv",
    )


def style_dataframe(df, highlight_fraud=True):
    """Perfect styling for fraud results"""
    def highlight_fraud(row):
        if highlight_fraud and 'Result' in row and 'FRAUD' in str(row['Result']):
            return ['background-color: #fee2e2; color: #dc2626; font-weight: bold'] * len(row)
        elif highlight_fraud and 'Result' in row and 'LEGIT' in str(row['Result']):
            return ['background-color: #ecfdf5; color: #059669'] * len(row)
        return [''] * len(row)
    
    return df.style.apply(highlight_fraud, axis=1).format({
        'fraud_probability': '{:.3f}',
        'transaction_amount': '${:,.2f}'
    })


def show_class_balance(df: pd.DataFrame) -> None:
    """Render a small class-balance summary."""
    if "is_fraud" not in df.columns:
        return
    total = len(df)
    fraud = df["is_fraud"].sum()
    legit = total - fraud
    st.info(
        f"📊 Dataset — Total: {total:,} | "
        f"🟢 Legit: {legit:,} ({legit/total:.1%}) | "
        f"🔴 Fraud: {fraud:,} ({fraud/total:.1%})"
    )
