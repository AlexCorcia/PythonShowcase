import os

from stockapp.pipelines.forecast_2026 import run_forecast_2026_hgb_direct_weekly
from stockapp.pipelines.backtest import run_backtest_hgb_direct_weekly

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "S8HGL5NJILISUR66")
DEFAULT_SYMBOL = "NVDA"

# Cambia esto sin tocar nada más:
# MODE = "forecast_2026"
MODE = "backtest"


def main():
    symbol = DEFAULT_SYMBOL

    if MODE == "forecast_2026":
        run_forecast_2026_hgb_direct_weekly(symbol=symbol, api_key=API_KEY, forecast_weeks=52)

    elif MODE == "backtest":
        run_backtest_hgb_direct_weekly(symbol=symbol, api_key=API_KEY, start_year=2016, end_year=2025)

    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
