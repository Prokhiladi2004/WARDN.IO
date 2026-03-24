import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def render_geo_analysis(df: pd.DataFrame) -> None:
    """
    Render geographic transaction analysis.
    Requires 'latitude' and 'longitude' columns.
    Optional: 'is_fraud', 'geo_distance'.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.warning("⚠️ 'latitude' and 'longitude' columns required for geo analysis.")
        return

    st.subheader("🌍 Geographic Transaction Analysis")

    # ── Map all transactions ─────────────────────────────────────────────────
    color_col = "is_fraud" if "is_fraud" in df.columns else None
    fig = px.scatter_mapbox(
        df.dropna(subset=["latitude", "longitude"]),
        lat="latitude",
        lon="longitude",
        color=color_col,
        color_discrete_map={0: "#22d3ee", 1: "#ef4444"},
        hover_data=["amount"] if "amount" in df.columns else None,
        title="Transaction Heatmap",
        mapbox_style="carto-darkmatter",
        zoom=2,
        height=500,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Geo-distance histogram ───────────────────────────────────────────────
    if "geo_distance" in df.columns:
        st.subheader("📏 Geographic Distance from User Home Location")
        fig2 = px.histogram(
            df,
            x="geo_distance",
            color=color_col,
            nbins=60,
            title="Distribution of Geo Distance",
            color_discrete_map={0: "#22d3ee", 1: "#ef4444"},
            barmode="overlay",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # High-distance flags
        threshold = st.slider("Distance alert threshold (km / degrees)", 0.0, 5000.0, 1000.0, step=50.0)
        flagged = df[df["geo_distance"] > threshold]
        st.info(
            f"🚩 {len(flagged):,} transactions exceed the distance threshold of {threshold:.0f}."
        )
        if len(flagged) and "is_fraud" in flagged.columns:
            fraud_ratio = flagged["is_fraud"].mean()
            st.metric("Fraud rate among high-distance transactions", f"{fraud_ratio:.1%}")
