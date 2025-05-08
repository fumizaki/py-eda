import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List


def cmap_options() -> list[str]:
    return [
        "Blues",
        "Reds",
        "Greens",
        "Purples",
        "YlOrBr",
        "YlGnBu",
        "RdBu",
        "coolwarm"
    ]

def fmt_options() -> list[str]:
    return [
        ".2f",
        ".3f",
        ".1f",
        ".0f"
    ]


def draw(
    df: pd.DataFrame,
    title: Optional[str] = "Correlation Heatmap",
    cmap: Optional[str] = "Blues",
    annot: Optional[bool] = True,
    fmt: Optional[str] = '.2f',
    linewidth: Optional[float] = 0.5,
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draws a heatmap of the correlation matrix of the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (List[str]): The columns to include in the heatmap.
        title (Optional[str]): The title of the plot. Defaults to "Correlation Heatmap".
        cmap (Optional[str]): The colormap to use. Defaults to "Blues".
        annot (Optional[bool]): Whether to annotate the heatmap with values. Defaults to True.
        fmt (Optional[str]): Format string for annotations. Defaults to '.2f'.
        linewidth (Optional[float]): Width of the lines that divide cells. Defaults to 0.5.
        height (Optional[int]): The height of the figure in pixels. Defaults to 15.
        width (Optional[int]): The width of the figure in pixels. Defaults to 30.
    """
    # Set context and style
    sns.set_context('poster')
    
    
    # Create figure and axes with specified size
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Create the heatmap
    heatmap_kwargs = {
        "data": df,
        "annot": annot,
        "fmt": fmt,
        "linewidths": linewidth,
        "cmap": cmap,
        "ax": ax
    }
    
    sns.heatmap(**heatmap_kwargs)
    
    # Set title
    ax.set_title(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

