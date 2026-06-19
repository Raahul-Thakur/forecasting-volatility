import numpy as np
import pandas as pd

from volatility_lab.backtesting import annualized_vol_to_daily_var, qlike_loss
from volatility_lab.models import har_rv_forecast
from volatility_lab.portfolio import vol_target_positions


def test_har_forecast_index_is_target_date_for_horizon():
    idx = pd.date_range("2024-01-01", periods=90, freq="B")
    rv = pd.Series(np.linspace(0.10, 0.30, len(idx)), index=idx)
    preds, _ = har_rv_forecast(rv, h=3, train_window=30)
    assert len(preds) > 0
    assert preds.index.min() in rv.index
    first_target_pos = rv.index.get_loc(preds.index[0])
    first_feature_pos = first_target_pos - 3
    assert first_feature_pos >= 30


def test_qlike_is_finite_for_positive_variances():
    idx = pd.date_range("2024-01-01", periods=5)
    realized = pd.Series([0.01, 0.02, 0.015, 0.018, 0.012], index=idx)
    forecast_vol = pd.Series([0.20, 0.22, 0.19, 0.21, 0.18], index=idx)
    loss = qlike_loss(realized, annualized_vol_to_daily_var(forecast_vol))
    assert np.isfinite(loss)


def test_portfolio_weights_respect_max_leverage():
    idx = pd.date_range("2024-01-01", periods=10)
    returns = pd.Series(np.repeat(0.001, 10), index=idx)
    forecast_vol = pd.Series(np.repeat(0.02, 10), index=idx)
    weights = vol_target_positions(returns, forecast_vol, target_vol_annual=0.20, max_leverage=1.5)
    assert (weights <= 1.5).all()
