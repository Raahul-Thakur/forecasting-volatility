import io

import numpy as np
import pandas as pd


def validate_date_range(start_date, end_date) -> None:
    if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
        raise ValueError("Start date must be before end date.")


def validate_price_frame(df: pd.DataFrame, date_col: str, price_col: str, min_rows: int = 30) -> pd.DataFrame:
    missing = [col for col in [date_col, price_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    out = df[[date_col, price_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna().sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    if len(out) < min_rows:
        raise ValueError(f"Need at least {min_rows} valid price rows; found {len(out)}.")
    if (out[price_col] <= 0).any():
        raise ValueError("Prices must be strictly positive to compute log returns.")
    return out.set_index(date_col)[[price_col]]


def to_log_returns_from_prices(df: pd.DataFrame, price_col: str) -> pd.Series:
    px = pd.to_numeric(df[price_col], errors="coerce")
    if (px <= 0).any():
        raise ValueError("Prices must be strictly positive to compute log returns.")
    r = np.log(px).diff().dropna()
    r.name = "logret"
    return r


def sample_daily_data(n_days: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    sigma = np.zeros(n_days)
    rets = np.zeros(n_days)
    sigma[0] = 0.012
    for i in range(1, n_days):
        sigma[i] = np.sqrt(0.000004 + 0.08 * rets[i - 1] ** 2 + 0.90 * sigma[i - 1] ** 2)
        rets[i] = sigma[i] * rng.standard_t(df=6)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"Date": dates, "Close": close})


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    if not data:
        raise ValueError("Uploaded file is empty.")
    return pd.read_csv(io.BytesIO(data))


def fetch_yfinance_daily(ticker: str, start_date, end_date) -> pd.DataFrame:
    validate_date_range(start_date, end_date)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance with: pip install yfinance") from exc

    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No Yahoo Finance data returned for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    if close_col not in data.columns:
        raise ValueError(f"Yahoo Finance response for {ticker} did not include a close column.")
    out = data[[close_col]].reset_index().rename(columns={close_col: "Close"})
    return out[["Date", "Close"]].dropna()


def fetch_yfinance_intraday(ticker: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance with: pip install yfinance") from exc

    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No Yahoo Finance intraday data returned for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Yahoo Finance intraday response missing columns: {missing}")
    out = data[required].reset_index()
    datetime_col = "Datetime" if "Datetime" in out.columns else out.columns[0]
    out = out.rename(columns={datetime_col: "Datetime"})
    return out[["Datetime", "Open", "High", "Low", "Close"]].dropna()
