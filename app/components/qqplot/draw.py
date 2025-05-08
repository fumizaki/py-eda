import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from typing import Optional

def draw(
    df: pd.DataFrame,
    col: str,
    dist: Optional[str] = 'norm',
    title: Optional[str] = None,
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draw a QQ plot of the specified column in the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The column to plot.
        dist (Optional[str]): The theoretical distribution to compare against.
                             Defaults to 'norm' (normal distribution).
                             See scipy.stats for other options (e.g., 'uniform', 'expon').
        title (Optional[str]): Title of the plot. If None, uses column name.
        height (Optional[int]): Height of the plot in inches. Defaults to 15.
        width (Optional[int]): Width of the plot in inches. Defaults to 30.
    """
    # Set context and style
    sns.set_context('poster')
    sns.set_style('whitegrid')

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(width, height))

    # Generate QQ plot
    stats.probplot(df[col], dist=dist, plot=ax)

    # Set title
    if title is None:
        title = f"QQ Plot of {col} (vs {dist})"
    ax.set_title(title, fontsize=16)

    # Set labels
    ax.set_xlabel(f"Theoretical Quantiles ({dist})", fontsize=12)
    ax.set_ylabel("Sample Quantiles", fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)