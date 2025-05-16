import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List
from pandas.plotting import parallel_coordinates


def draw_paralellcoor(
    df: pd.DataFrame,
    columns: List[str],
    class_col: str,
    alpha: Optional[float] = 0.5,
    color_palette: Optional[str] = "tab10",
    title: Optional[str] = None,
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draw a parallel coordinates plot of the specified columns in the dataframe.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of columns to include in the plot.
        class_col (str): Column to use for coloring lines.
        alpha (Optional[float]): Transparency of lines. Defaults to 0.5.
        color_palette (Optional[str]): Color palette. Defaults to "tab10".
        title (Optional[str]): Title of the plot. If None, generates a title.
        height (Optional[int]): Height of the plot in pixels. Defaults to 15.
        width (Optional[int]): Width of the plot in pixels. Defaults to 30.
    """
    # Set context and style
    plt.figure(figsize=(width, height))
    
    # Select columns to plot
    df_subset = df[columns + [class_col]]
    
    # Create custom colormap
    colors = sns.color_palette(color_palette, len(df_subset[class_col].unique()))
    
    # Draw parallel coordinates plot
    parallel_coordinates(
        df_subset,
        class_col,
        cols=columns,
        alpha=alpha,
        color=colors
    )
    
    # Set title
    if title is None:
        title = f"Parallel Coordinates Plot (colored by {class_col})"
    plt.title(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt.gcf())