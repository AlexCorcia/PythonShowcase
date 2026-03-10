import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def load_price_data(symbol: str, freq: str = "weekly") -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_{freq}.csv"

    if path.exists():
        df = pd.read_csv(path)
    else:
        interval = "1wk" if freq == "weekly" else "1d"

        df = yf.download(
            symbol,
            start="2015-01-01",
            auto_adjust=True,
            progress=False,
            interval=interval,
        )

        if df is None or df.empty:
            raise ValueError(f"No data downloaded for symbol={symbol}")

        df = df.reset_index()

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(x) for x in col if x not in ("", None)]).strip("_")
                for col in df.columns.values
            ]
        else:
            df.columns = [str(col) for col in df.columns]

        # Find date column
        date_col = None
        for col in df.columns:
            if col.lower() == "date" or "date" in col.lower():
                date_col = col
                break

        # Find close column
        close_col = None
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == "close" or col_lower.startswith("close_") or "close" in col_lower:
                close_col = col
                break

        if date_col is None or close_col is None:
            raise ValueError(
                f"Downloaded data for {symbol} does not contain recognizable 'date' and 'close' columns. "
                f"Columns found: {list(df.columns)}"
            )

        df = df.rename(columns={date_col: "date", close_col: "close"})
        df = df[["date", "close"]]
        df.to_csv(path, index=False)

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    return df