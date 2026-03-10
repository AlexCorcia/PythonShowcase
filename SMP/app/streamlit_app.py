import streamlit as st

from config import DEFAULT_SYMBOLS, DEFAULT_MODELS
from services.data_service import load_price_data
from services.model_registry import list_available_models, load_model, get_trained_models_inventory
from services.inference import run_backtest, run_forecast
from services.training_service import train_lgbm_for_symbol
from ui.plots import plot_predictions
from ui.metrics import render_metrics

st.set_page_config(page_title="Stock Forecast Showcase", layout="wide")

st.title("📈 Stock Forecast Showcase")

page = st.sidebar.radio(
    "Page",
    ["Forecast", "Train Models", "Model Registry"]
)

if page == "Train Models":
    st.subheader("Train Models")

    train_symbols = st.multiselect(
        "Symbols to train",
        DEFAULT_SYMBOLS,
        default=DEFAULT_SYMBOLS[:2],
    )

    train_freq = st.selectbox("Frequency", ["daily", "weekly"], index=1)
    lookback = st.slider("Lookback", min_value=5, max_value=60, value=20, step=1)

    if st.button("Train LightGBM models"):
        if not train_symbols:
            st.warning("Select at least one symbol.")
        else:
            results = []
            progress = st.progress(0)

            for idx, symbol in enumerate(train_symbols):
                df = load_price_data(symbol=symbol, freq=train_freq)
                result = train_lgbm_for_symbol(symbol=symbol, df=df, lookback=lookback)
                results.append(result)
                progress.progress((idx + 1) / len(train_symbols))

            st.success("Training completed.")
            st.dataframe(results, use_container_width=True)

    st.divider()
    st.subheader("Available trained models")

    inventory = get_trained_models_inventory()

    if not inventory:
        st.info("No trained models found yet.")
    else:
        st.dataframe(inventory, use_container_width=True)

        selected_symbol = st.selectbox(
            "Inspect models for symbol",
            sorted(set(row["symbol"] for row in inventory))
        )

        symbol_models = [row for row in inventory if row["symbol"] == selected_symbol]
        st.write(symbol_models)

    st.stop()

# ---------- Forecast page ----------
st.caption("Compare trained models, backtests, and future forecasts.")

st.sidebar.header("Forecast Controls")

symbols = DEFAULT_SYMBOLS
symbol = st.sidebar.selectbox("Symbol", symbols, index=0)

available_models = list_available_models(symbol=symbol, defaults=DEFAULT_MODELS)
model_name = st.sidebar.selectbox("Model", available_models, index=0)

mode = st.sidebar.radio("Mode", ["Backtest", "Forecast"], index=0)
freq = st.sidebar.selectbox("Frequency", ["daily", "weekly"], index=1)

chart_height = st.sidebar.slider("Chart height", min_value=350, max_value=1000, value=520, step=10)

past_options = {
    "3 months": 3,
    "6 months": 6,
    "1 year": 12,
    "3 years": 36,
    "5 years": 60,
    "Max": None,
}
past_label = st.sidebar.selectbox("Show past history", list(past_options.keys()), index=2)
past_months = past_options[past_label]

future_options = {
    "1 month": 1,
    "3 months": 3,
    "6 months": 6,
    "12 months": 12,
}
future_label = st.sidebar.selectbox("Forecast into future", list(future_options.keys()), index=1)
future_months = future_options[future_label]

with st.spinner("Loading data..."):
    df = load_price_data(symbol=symbol, freq=freq)

with st.spinner("Loading model..."):
    model = load_model(symbol=symbol, model_name=model_name)

if mode == "Backtest":
    st.subheader("Backtest")

    with st.spinner("Running backtest..."):
        result = run_backtest(df=df, model=model, model_name=model_name)

    left, right = st.columns([2, 1], gap="large")
    with left:
        fig = plot_predictions(
            df=result["plot_df"],
            title=f"{symbol} — {model_name} backtest",
            forecast_start_date=None,
            chart_height=chart_height,
            past_months=past_months,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        render_metrics(result["metrics"])
        st.markdown("#### Notes")
        st.write(result.get("notes", "—"))

else:
    st.subheader("Forecast")

    with st.spinner("Forecasting..."):
        result = run_forecast(
            df=df,
            model=model,
            model_name=model_name,
            freq=freq,
            future_months=future_months,
        )

    left, right = st.columns([2, 1], gap="large")
    with left:
        fig = plot_predictions(
            df=result["plot_df"],
            title=f"{symbol} — {model_name} forecast",
            forecast_start_date=result.get("forecast_start_date"),
            chart_height=chart_height,
            past_months=past_months,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        render_metrics(result.get("metrics", {}))
        st.markdown("#### Forecast settings")
        st.write(result.get("settings", {}))