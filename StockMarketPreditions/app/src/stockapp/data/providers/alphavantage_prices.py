import pandas as pd
from alpha_vantage.timeseries import TimeSeries


def fetch_daily_data_alpha_vantage(symbol: str, api_key: str, outputsize: str = "compact") -> pd.DataFrame:
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, _meta = ts.get_daily(symbol=symbol, outputsize=outputsize)
    data.columns = ["open", "high", "low", "close", "volume"]
    return data.sort_index()


def fetch_weekly_data_alpha_vantage(symbol: str, api_key: str) -> pd.DataFrame:
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, _meta = ts.get_weekly(symbol=symbol)
    data.columns = ["open", "high", "low", "close", "volume"]
    return data.sort_index()
