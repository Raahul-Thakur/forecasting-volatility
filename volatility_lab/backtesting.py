import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import TRADING_DAYS


def annualized_vol_to_daily_var(vol_annual: pd.Series, eps: float = 1e-12) -> pd.Series:
    return ((vol_annual**2) / TRADING_DAYS).clip(lower=eps)


def qlike_series(realized_var: pd.Series, forecast_var: pd.Series, eps: float = 1e-12) -> pd.Series:
    rv, fv = realized_var.align(forecast_var, join="inner")
    fv = fv.clip(lower=eps)
    return np.log(fv) + rv / fv


def qlike_loss(realized_var: pd.Series, forecast_var: pd.Series) -> float:
    loss = qlike_series(realized_var, forecast_var)
    return float(loss.mean()) if len(loss) else np.nan


def eval_models(realized_vol: pd.Series, realized_var: pd.Series, model_vols: dict, test_start=None) -> pd.DataFrame:
    rows = []
    for name, vol in model_vols.items():
        y_vol, p_vol = realized_vol.align(vol, join="inner")
        if test_start is not None:
            y_vol = y_vol.loc[y_vol.index >= pd.Timestamp(test_start)]
            p_vol = p_vol.reindex(y_vol.index).dropna()
            y_vol = y_vol.reindex(p_vol.index)
        if len(y_vol) < 2:
            continue
        y_var = realized_var.reindex(y_vol.index).dropna()
        p_var = annualized_vol_to_daily_var(p_vol).reindex(y_var.index).dropna()
        y_var = y_var.reindex(p_var.index)
        rows.append(
            {
                "model": name,
                "MAE(vol)": mean_absolute_error(y_vol, p_vol),
                "RMSE(vol)": np.sqrt(mean_squared_error(y_vol, p_vol)),
                "MSE(var)": mean_squared_error(y_var, p_var) if len(y_var) else np.nan,
                "QLIKE(var)": qlike_loss(y_var, p_var),
                "n": len(y_vol),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["QLIKE(var)", "RMSE(vol)"]).reset_index(drop=True) if rows else pd.DataFrame()
    )


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def diebold_mariano_table(realized_var: pd.Series, model_vols: dict) -> pd.DataFrame:
    losses = {name: qlike_series(realized_var, annualized_vol_to_daily_var(vol)) for name, vol in model_vols.items()}
    rows = []
    names = list(losses)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            la, lb = losses[a].align(losses[b], join="inner")
            d = (la - lb).dropna()
            if len(d) < 5 or d.std(ddof=1) == 0:
                stat, p_value = np.nan, np.nan
            else:
                stat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
                p_value = float(2 * (1 - normal_cdf(abs(stat))))
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "mean_loss_diff_a_minus_b": float(d.mean()) if len(d) else np.nan,
                    "DM_stat": stat,
                    "p_value": p_value,
                    "n": len(d),
                }
            )
    return pd.DataFrame(rows)


def forecast_error_frame(realized_vol: pd.Series, model_vols: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: realized_vol.align(vol, join="inner")[0] - realized_vol.align(vol, join="inner")[1]
            for name, vol in model_vols.items()
        }
    )


def confidence_band(realized_vol: pd.Series, forecast_vol: pd.Series, window: int = 63, z: float = 1.96):
    y, f = realized_vol.align(forecast_vol, join="inner")
    err_std = (y - f).rolling(window).std().bfill()
    return (f - z * err_std).clip(lower=0), f + z * err_std


def rolling_model_ranking(realized_var: pd.Series, model_vols: dict, window: int = 63) -> pd.DataFrame:
    losses = pd.DataFrame(
        {name: qlike_series(realized_var, annualized_vol_to_daily_var(vol)) for name, vol in model_vols.items()}
    )
    return losses.rolling(window).mean().rank(axis=1, method="min").dropna(how="all")


def var_hit_series(returns: pd.Series, var_1d: pd.Series) -> pd.Series:
    r, v = returns.align(var_1d, join="inner")
    return (r < -v).astype(int)


def kupiec_var_test(returns: pd.Series, var_1d: pd.Series, alpha: float) -> dict:
    hits = var_hit_series(returns, var_1d).dropna()
    n = len(hits)
    x = int(hits.sum())
    if n == 0:
        return {"exceptions": 0, "observations": 0, "hit_ratio": np.nan, "LR_uc": np.nan, "p_value": np.nan}
    p_hat = np.clip(x / n, 1e-12, 1 - 1e-12)
    alpha = np.clip(alpha, 1e-12, 1 - 1e-12)
    lr = -2 * (((n - x) * np.log(1 - alpha) + x * np.log(alpha)) - ((n - x) * np.log(1 - p_hat) + x * np.log(p_hat)))
    p_value = 1 - math.erf(math.sqrt(max(lr, 0) / 2))
    return {"exceptions": x, "observations": n, "hit_ratio": x / n, "LR_uc": float(lr), "p_value": float(p_value)}
