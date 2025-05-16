import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List


def draw_lineplot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: Optional[str] = None,
    markers: Optional[bool] = True,
    palette: Optional[str] = "tab10",
    height: Optional[int] = 15,
    width: Optional[int] = 30
) -> None:
    """
    Draw a line plot of the specified columns in the dataframe.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column for the x-axis.
        y_cols (List[str]): List of columns for the y-axis.
        title (Optional[str]): Title of the plot. If None, generates a title.
        markers (Optional[bool]): Whether to show markers. Defaults to True.
        palette (Optional[str]): Color palette. Defaults to "tab10".
        height (Optional[int]): Height of the plot in pixels. Defaults to 15.
        width (Optional[int]): Width of the plot in pixels. Defaults to 30.
    """
    # Set context and style
    sns.set_context('poster')
    sns.set_style('whitegrid')
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Draw line for each y column
    for i, y_col in enumerate(y_cols):
        plt.plot(
            df[x_col],
            df[y_col],
            marker='o' if markers else None,
            linewidth=2,
            alpha=0.7,
            label=y_col
        )
    
    # Set title
    if title is None:
        title = f"Line Plot of {', '.join(y_cols)} vs {x_col}"
    ax.set_title(title, fontsize=16)
    
    # Set labels
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)