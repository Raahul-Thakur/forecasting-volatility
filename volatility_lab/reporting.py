import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def download_dataframe(st, label: str, df: pd.DataFrame, filename: str):
    st.download_button(label, df.to_csv(index=True).encode("utf-8"), file_name=filename, mime="text/csv")


def html_report(metrics: pd.DataFrame, best_model: str, rv_source: str, settings: dict, dataset_summary: dict) -> bytes:
    settings_df = pd.DataFrame(sorted(settings.items()), columns=["setting", "value"])
    dataset_df = pd.DataFrame(sorted(dataset_summary.items()), columns=["field", "value"])
    html = f"""
    <html><head><title>Volatility Lab Report</title></head><body>
    <h1>Volatility Forecasting Lab Report</h1>
    <p><strong>Generated:</strong> {datetime.now().isoformat(timespec="seconds")}</p>
    <p><strong>Realized volatility target:</strong> {rv_source}</p>
    <p><strong>Best model by QLIKE:</strong> {best_model}</p>
    <h2>Dataset Summary</h2>
    {dataset_df.to_html(index=False)}
    <h2>Model Settings</h2>
    {settings_df.to_html(index=False)}
    <h2>Metrics</h2>
    {metrics.to_html(index=False)}
    </body></html>
    """
    return html.encode("utf-8")


def make_run_output_dir(base_dir: str | Path, ticker: str, run_time: datetime | None = None) -> Path:
    run_time = run_time or datetime.now()
    safe_ticker = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker or "run").strip("_") or "run"
    run_dir = Path(base_dir) / f"{run_time:%Y%m%d_%H%M%S}_{safe_ticker}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    return run_dir


def save_run_tables(
    run_dir: Path,
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    settings: dict,
    dataset_summary: dict,
    best_model: str,
    rv_source: str,
) -> None:
    data_dir = run_dir / "data"
    metrics.to_csv(data_dir / "metrics.csv", index=False)
    forecasts.to_csv(data_dir / "forecasts.csv", index=True)
    pd.DataFrame(sorted(settings.items()), columns=["setting", "value"]).to_csv(data_dir / "settings.csv", index=False)
    pd.DataFrame(sorted(dataset_summary.items()), columns=["field", "value"]).to_csv(
        data_dir / "dataset_summary.csv", index=False
    )
    (run_dir / "report.html").write_bytes(html_report(metrics, best_model, rv_source, settings, dataset_summary))
