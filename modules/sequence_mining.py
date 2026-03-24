from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
# import plotly.express as px
import pandas as pd
import streamlit as st



def mine_sequences(df: pd.DataFrame, top_n: int = 15) -> list[tuple]:
    """
    Mine the most common consecutive transaction-category pairs
    per user and return the top-N as a list of ((cat_a, cat_b), count).
    """
    if "category" not in df.columns or "user_id" not in df.columns:
        st.warning("⚠️ Columns 'category' and 'user_id' are required for sequence mining.")
        return []

    sort_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    seq = (
        df.sort_values(["user_id", sort_col])
        .groupby("user_id")["category"]
        .apply(list)
    )

    pairs: list[tuple[str, str]] = []
    for user_seq in seq:
        for i in range(len(user_seq) - 1):
            pairs.append((user_seq[i], user_seq[i + 1]))

    common = Counter(pairs).most_common(top_n)
    return common


def render_sequences(df: pd.DataFrame) -> None:
    """Run sequence mining and render results in Streamlit."""
    st.subheader("🔗 Transaction Sequence Mining")

    top_n = st.slider("Top N sequences", min_value=5, max_value=30, value=15)
    common = mine_sequences(df, top_n=top_n)

    if not common:
        return

    rows = [
        {"From Category": a, "To Category": b, "Count": cnt}
        for (a, b), cnt in common
    ]
    seq_df = pd.DataFrame(rows)
    st.dataframe(seq_df, use_container_width=True)

    # ── Sankey-style bar chart ──────────────────────────────────────────────
    seq_df["Sequence"] = seq_df["From Category"] + " → " + seq_df["To Category"]
    fig = px.bar(
        seq_df,
        x="Count",
        y="Sequence",
        orientation="h",
        title=f"Top {top_n} Category Transition Sequences",
        color="Count",
        color_continuous_scale="Plasma",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Fraud rate per sequence ─────────────────────────────────────────────
    if "is_fraud" in df.columns:
        st.subheader("🚨 Fraud Rate by Category")
        fraud_by_cat = (
            df.groupby("category")["is_fraud"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "Fraud Rate", "sum": "Fraud Count", "count": "Total"})
            .sort_values("Fraud Rate", ascending=False)
            .reset_index()
        )
        fraud_by_cat["Fraud Rate (%)"] = (fraud_by_cat["Fraud Rate"] * 100).round(2)
        fig2 = px.bar(
            fraud_by_cat,
            x="category",
            y="Fraud Rate (%)",
            title="Fraud Rate per Transaction Category",
            color="Fraud Rate (%)",
            color_continuous_scale="Reds",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig2, use_container_width=True)
