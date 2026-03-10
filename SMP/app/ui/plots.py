import pandas as pd
import plotly.graph_objects as go


def plot_predictions(
    df,
    title: str,
    forecast_start_date=None,
    chart_height: int = 520,
    past_months=None,
):
    fig = go.Figure()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Visual filter only: does NOT affect model inference
    if past_months is not None:
        if forecast_start_date is not None:
            forecast_start_date = pd.to_datetime(forecast_start_date)
            cutoff = forecast_start_date - pd.DateOffset(months=past_months)
            df = df[df["date"] >= cutoff].copy()
        else:
            max_date = df["date"].max()
            cutoff = max_date - pd.DateOffset(months=past_months)
            df = df[df["date"] >= cutoff].copy()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="lines",
            name="Actual",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["pred"],
            mode="lines",
            name="Prediction",
        )
    )

    if forecast_start_date is not None:
        y_min = df[["close", "pred"]].min(numeric_only=True).min()
        y_max = df[["close", "pred"]].max(numeric_only=True).max()

        fig.add_trace(
            go.Scatter(
                x=[forecast_start_date, forecast_start_date],
                y=[y_min, y_max],
                mode="lines",
                name="Forecast starts",
                line=dict(dash="dash"),
                hoverinfo="skip",
                showlegend=True,
            )
        )

        fig.add_annotation(
            x=forecast_start_date,
            y=y_max,
            text="Forecast starts",
            showarrow=True,
            arrowhead=1,
            yshift=10,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Series",
        height=chart_height,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig