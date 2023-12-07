"""
This module provides visualization utilities, particularly for plotting
target time series and forecasted values.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def plot_timeseries(
    y: pd.DataFrame,
    y_hat: pd.DataFrame,
    quantiles: tuple = None,
    title: str = None,
    save_path: Path = None,
) -> go.Figure:
    """Plot the target values and predictions using Plotly.

    Args:
        y (pd.DataFrame): A DataFrame containing the target values
            with a datetime index and a single target column.
        y_hat (pd.DataFrame): A DataFrame containing the predictions,
            with a datetime index and columns for prediction.
        title (str, optional): The title for the plot.
            If not provided, no title will be displayed.
        show_quantiles (tuple, optional): Quantiles to display for probabilistic forecasts.
            Defaults 1 and 99 quantile.

    Returns:
        go.Figure: The plotly figure object containing the plot.
    """

    fig = go.Figure()
    target = y.columns[0]

    # target series
    fig.add_trace(
        go.Scatter(
            x=y.index,
            y=y[target],
            mode="lines",
            line=dict(color="rgb(166, 119, 180)"),
            name="target",
        )
    )

    # forecast
    pred_mean = np.mean(y_hat, axis=1)
    fig.add_trace(
        go.Scatter(
            x=y_hat.index,
            y=pred_mean,
            mode="lines",
            line=dict(color="rgb(31, 119, 180)"),
            name="prediction",
        )
    )
    if quantiles:
        # low quantile
        low_q = np.quantile(y_hat, q=quantiles[0], axis=1)
        fig.add_trace(
            go.Scatter(
                x=y_hat.index,
                y=low_q,
                mode="lines",
                line=dict(color="rgb(31, 119, 180)", width=0),
                name="",
            )
        )
        # high quantile
        high_q = np.quantile(y_hat, q=quantiles[1], axis=1)
        fig.add_trace(
            go.Scatter(
                x=y_hat.index,
                y=high_q,
                mode="lines",
                line=dict(color="rgb(31, 119, 180)", width=0),
                fill="tonexty",
                name="1% - 99% quantile",
            )
        )
    if title:
        fig.update_layout(title=title)

    fig.update_yaxes(title=target)
    if save_path is not None:
        # fig.write_html(model_dir.joinpath(f"{title.split(' = ')[0]}.html"))
        pio.write_image(fig, save_path)

    return fig
