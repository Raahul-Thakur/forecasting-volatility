import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from . import TRADING_DAYS


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    r2 = returns.dropna() ** 2
    if r2.empty:
        raise ValueError("Need returns before computing EWMA volatility.")
    sigma2 = np.zeros(len(r2))
    sigma2[0] = float(r2.iloc[:20].mean() if len(r2) > 20 else r2.mean())
    for i in range(1, len(r2)):
        sigma2[i] = lam * sigma2[i - 1] + (1 - lam) * float(r2.iloc[i - 1])
    return pd.Series(np.sqrt(sigma2) * np.sqrt(TRADING_DAYS), index=r2.index, name=f"EWMA_{lam}")


def rolling_garch_forecast(
    returns: pd.Series,
    train_window: int = 756,
    horizon: int = 1,
    variant: str = "GARCH",
    dist: str = "t",
    expanding: bool = False,
    max_forecasts: int | None = None,
    progress_callback=None,
) -> pd.Series:
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("Install arch with: pip install arch") from exc

    if len(returns) < train_window + horizon + 10:
        raise ValueError("Not enough data for selected GARCH window.")
    r = returns.dropna() * 100
    ends = list(range(train_window, len(r) - horizon + 1))
    if max_forecasts and len(ends) > max_forecasts:
        ends = ends[-max_forecasts:]

    vols = []
    idx = []
    for pos, end in enumerate(ends, start=1):
        train = r.iloc[:end] if expanding else r.iloc[end - train_window : end]
        if variant == "GARCH":
            am = arch_model(train, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
        elif variant == "EGARCH":
            am = arch_model(train, mean="Zero", vol="EGARCH", p=1, q=1, dist=dist)
        elif variant == "GJR":
            am = arch_model(train, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=dist)
        else:
            raise ValueError("variant must be GARCH, EGARCH, or GJR")
        try:
            res = am.fit(disp="off")
            forecast = res.forecast(horizon=horizon, reindex=False)
            var_h = forecast.variance.values[-1, horizon - 1]
        except Exception as exc:
            raise RuntimeError(f"{variant} failed to converge at forecast {pos}/{len(ends)}: {exc}") from exc
        vols.append((np.sqrt(var_h) / 100) * np.sqrt(TRADING_DAYS))
        idx.append(returns.index[end + horizon - 1])
        if progress_callback:
            progress_callback(pos, len(ends))
    return pd.Series(vols, index=pd.DatetimeIndex(idx), name=f"{variant}11_h{horizon}")


def har_rv_forecast(
    realized_vol_annual: pd.Series,
    h: int = 1,
    ridge_alpha: float = 0.0,
    train_window: int = 252,
    expanding: bool = False,
) -> tuple[pd.Series, LinearRegression | Ridge]:
    rv = realized_vol_annual.dropna()
    df = pd.DataFrame({"RV_d": rv, "RV_w": rv.rolling(5).mean(), "RV_m": rv.rolling(22).mean()}).dropna()
    features = df[["RV_d", "RV_w", "RV_m"]]
    target = df["RV_d"]

    rows = []
    for i in range(0, len(df) - h):
        # Forecast made with information through date i and labelled at the target date i+h.
        rows.append((df.index[i + h], features.iloc[i], target.iloc[i + h]))
    if len(rows) < train_window + 5:
        raise ValueError("Not enough data for selected HAR train window.")

    x = pd.DataFrame([row[1] for row in rows], index=[row[0] for row in rows])
    y = pd.Series([row[2] for row in rows], index=[row[0] for row in rows], name="target")
    model = Ridge(alpha=ridge_alpha) if ridge_alpha and ridge_alpha > 0 else LinearRegression()
    preds = []
    idx = []
    for i in range(train_window, len(y)):
        x_train = x.iloc[:i] if expanding else x.iloc[i - train_window : i]
        y_train = y.iloc[:i] if expanding else y.iloc[i - train_window : i]
        model.fit(x_train, y_train)
        preds.append(max(0.0, float(model.predict(x.iloc[i : i + 1])[0])))
        idx.append(y.index[i])
    return pd.Series(preds, index=pd.DatetimeIndex(idx), name=f"HAR_RV_h{h}"), model


def stacked_hybrid_forecast(
    target_realized_vol: pd.Series, base_forecasts: dict, train_window: int = 252, ridge_alpha: float = 1.0
) -> pd.Series:
    y = target_realized_vol.dropna()
    x = pd.DataFrame(base_forecasts).dropna()
    y, x = y.align(x, join="inner", axis=0)
    if len(y) < train_window + 20:
        raise ValueError("Not enough data for hybrid stacking window.")
    preds = []
    idx = []
    model = Ridge(alpha=ridge_alpha)
    for i in range(train_window, len(y)):
        model.fit(x.iloc[i - train_window : i], y.iloc[i - train_window : i])
        preds.append(max(0.0, float(model.predict(x.iloc[i : i + 1])[0])))
        idx.append(y.index[i])
    return pd.Series(preds, index=pd.DatetimeIndex(idx), name="HYBRID_STACKED")
