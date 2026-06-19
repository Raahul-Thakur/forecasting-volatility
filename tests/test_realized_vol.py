import numpy as np
import pandas as pd

from volatility_lab.data import to_log_returns_from_prices, validate_price_frame
from volatility_lab.realized_vol import (
    aggregate_intraday_realized_vol,
    realized_var_from_daily_returns,
    realized_vol_from_daily_returns,
)


def test_log_return_calculation():
    df = pd.DataFrame({"Close": [100.0, 105.0, 102.0]}, index=pd.date_range("2024-01-01", periods=3))
    returns = to_log_returns_from_prices(df, "Close")
    assert np.isclose(returns.iloc[0], np.log(105.0 / 100.0))
    assert returns.name == "logret"


def test_non_positive_prices_are_rejected():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=30), "Close": [100.0] * 29 + [0.0]})
    try:
        validate_price_frame(df, "Date", "Close")
    except ValueError as exc:
        assert "strictly positive" in str(exc)
    else:
        raise AssertionError("Expected non-positive prices to be rejected")


def test_realized_vol_and_var_alignment():
    returns = pd.Series(np.linspace(-0.02, 0.02, 30), index=pd.date_range("2024-01-01", periods=30))
    vol = realized_vol_from_daily_returns(returns, window=5)
    var = realized_var_from_daily_returns(returns, window=5)
    assert vol.index.equals(var.index)
    assert len(vol) == 26
    assert (vol >= 0).all()


def test_intraday_realized_index_is_timezone_naive():
    intraday = pd.DataFrame(
        {
            "Datetime": [
                "2024-01-02 09:30:00-05:00",
                "2024-01-02 09:32:00-05:00",
                "2024-01-02 09:34:00-05:00",
            ],
            "Open": [100.0, 100.1, 100.2],
            "High": [100.2, 100.3, 100.4],
            "Low": [99.9, 100.0, 100.1],
            "Close": [100.1, 100.2, 100.3],
        }
    )
    out = aggregate_intraday_realized_vol(intraday, "Datetime", "Open", "High", "Low", "Close", "Realized variance")
    assert out.index.tz is None
