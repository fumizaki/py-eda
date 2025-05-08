import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional


def draw(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    title: Optional[str] = "Scatter Plot",
    alpha: Optional[float] = 0.7,
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draws a scatter plot using the given DataFrame and columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column to use for the x-axis.
        y_col (str): The column to use for the y-axis.
        color_col (Optional[str]): The column to use for point colors. Defaults to None.
        size_col (Optional[str]): The column to use for point sizes. Defaults to None.
        title (Optional[str]): The title of the plot. Defaults to "Scatter Plot".
        alpha (Optional[float]): The transparency of the points. Defaults to 0.7.
        height (Optional[int]): The height of the figure in pixels. Defaults to 15.
        width (Optional[int]): The width of the figure in pixels. Defaults to 30.
    """
    # Set context and style
    sns.set_context('poster')
    sns.set_style('whitegrid')
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Create the scatter plot
    scatter_kwargs = {
        "data": df,
        "x": x_col,
        "y": y_col,
        "alpha": alpha,
        "ax": ax
    }
    
    if color_col:
        scatter_kwargs["hue"] = color_col
    
    if size_col:
        scatter_kwargs["size"] = size_col
    
    sns.scatterplot(**scatter_kwargs)
    
    # Set title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
