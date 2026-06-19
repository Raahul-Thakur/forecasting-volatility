import numpy as np
import pandas as pd

from . import TRADING_DAYS


def vol_target_positions(
    returns: pd.Series, forecast_vol_annual: pd.Series, target_vol_annual: float, max_leverage: float, lag: int = 1
) -> pd.Series:
    fv = forecast_vol_annual.shift(lag).dropna()
    weights = (target_vol_annual / fv.replace(0, np.nan)).clip(lower=0, upper=max_leverage).dropna()
    weights.name = "weight"
    return weights


def portfolio_pnl(returns: pd.Series, weights: pd.Series) -> pd.Series:
    r, w = returns.align(weights, join="inner")
    pnl = w * r
    pnl.name = "pnl"
    return pnl


def parametric_var_1d(forecast_vol_annual: pd.Series, z: float = 2.33) -> pd.Series:
    return z * forecast_vol_annual / np.sqrt(TRADING_DAYS)


def make_multi_asset_returns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Missing multi-asset date column: {date_col}")
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x = x.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    prices = x.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if prices.shape[1] < 2:
        raise ValueError("Multi-asset file needs at least two numeric asset price columns.")
    if (prices <= 0).any().any():
        raise ValueError("Multi-asset prices must be strictly positive.")
    return np.log(prices.astype(float)).diff().dropna(how="all")


def inverse_vol_weights(vols: pd.Series, max_weight: float) -> pd.Series:
    w = (1 / vols.replace(0, np.nan)).dropna()
    w = w / w.sum()
    w = w.clip(upper=max_weight)
    return w / w.sum()


def minimum_variance_weights(cov: pd.DataFrame, max_weight: float) -> pd.Series:
    assets = cov.columns
    inv = np.linalg.pinv(cov.values)
    ones = np.ones(len(assets))
    w = pd.Series((inv @ ones) / (ones @ inv @ ones), index=assets)
    w = w.clip(lower=0, upper=max_weight)
    return w / w.sum()


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    w = weights.reindex(cov.columns).fillna(0).values
    port_var = float(w @ cov.values @ w)
    if port_var <= 0:
        return pd.Series(index=cov.columns, dtype=float)
    return pd.Series(w * (cov.values @ w) / port_var, index=cov.columns, name="risk_contribution")


def portfolio_summary(returns: pd.DataFrame, weights: pd.Series, target_vol: float, var_z: float):
    w = weights.reindex(returns.columns).fillna(0)
    raw = returns.mul(w, axis=1).sum(axis=1).dropna()
    raw_vol = raw.std() * np.sqrt(TRADING_DAYS)
    scale = min(3.0, target_vol / raw_vol) if raw_vol and np.isfinite(raw_vol) and raw_vol > 0 else 1
    pnl = raw * scale
    var_1d = var_z * pnl.std()
    cvar_1d = pnl[pnl <= -var_1d].mean() if len(pnl[pnl <= -var_1d]) else np.nan
    return pnl, {
        "scale": scale,
        "ann_return": (1 + pnl).prod() ** (TRADING_DAYS / len(pnl)) - 1 if len(pnl) else np.nan,
        "ann_vol": pnl.std() * np.sqrt(TRADING_DAYS),
        "sharpe": pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS) if pnl.std() > 0 else np.nan,
        "var_1d": var_1d,
        "cvar_1d": cvar_1d,
    }
