import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def plot_series(title: str, series_dict: dict, height: int = 4, save_path=None):
    plt.figure(figsize=(12, height))
    for name, s in series_dict.items():
        s = s.dropna()
        if len(s):
            plt.plot(s.index, s.values, label=name)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    st.pyplot(plt.gcf())
    plt.close()


def correlation_heatmap(corr: pd.DataFrame, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Correlation")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    st.pyplot(plt.gcf())
    plt.close()


def realized_forecast_scatter(
    realized: pd.Series,
    forecast: pd.Series,
    title: str = "Realized vs Forecast",
    save_path=None,
):
    data = pd.DataFrame({"Realized": realized, "Forecast": forecast}).dropna()
    if data.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(data["Realized"], data["Forecast"], alpha=0.6)
    lo = min(data["Realized"].min(), data["Forecast"].min())
    hi = max(data["Realized"].max(), data["Forecast"].max())
    plt.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="45-degree line")
    plt.xlabel("Realized annualized volatility")
    plt.ylabel("Forecast annualized volatility")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    st.pyplot(plt.gcf())
    plt.close()
