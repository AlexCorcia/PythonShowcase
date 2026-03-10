import streamlit as st

def render_metrics(metrics: dict):
    st.markdown("#### Metrics")
    if not metrics:
        st.info("No metrics available for future-only forecast (no ground truth).")
        return
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
